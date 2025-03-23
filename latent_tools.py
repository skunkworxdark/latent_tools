# 2025 skunkworxdark (https://github.com/skunkworxdark)

import io
import math
from typing import Literal

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")  # Use the Agg backend to prevent popup
from matplotlib import pyplot as plt
from matplotlib import ticker
from PIL import Image
from scipy import stats

from invokeai.invocation_api import (
    BaseInvocation,
    FieldDescriptions,
    ImageOutput,
    Input,
    InputField,
    InvocationContext,
    LatentsField,
    LatentsOutput,
    WithBoard,
    WithMetadata,
    invocation,
)


@invocation(
    "latent_average",
    title="Latent Average",
    tags=["latents"],
    category="latents",
    version="1.0.0",
)
class LatentAverageInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Average two latents"""

    latentA: LatentsField = InputField(
        default=None,
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    latentB: LatentsField = InputField(
        default=None,
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latent_a = context.tensors.load(self.latentA.latents_name)
        latent_b = context.tensors.load(self.latentB.latents_name)

        assert latent_a.shape == latent_b.shape, "inputs must be same size"

        latent_out = (latent_a + latent_b) / 2

        name = context.tensors.save(tensor=latent_out)
        return LatentsOutput.build(latents_name=name, latents=latent_out, seed=None)


BLEND_MODES = Literal[
    "average",
    "add",
    "subtract",
    "difference",
    "maximum",
    "minimum",
    "multiply",
    "frequency_blend",
    "screen",
    "dodge",
    "burn",
    "overlay",
    "soft_light",
    "hard_light",
    "color_dodge",
    "color_burn",
    "linear_dodge",
    "linear_burn",
    # Add more blend modes here...
]


@invocation(
    "latent_combine",
    title="Latent Combine",
    tags=["latents", "combine"],
    category="latents",
    version="1.0.0",
)
class LatentCombineInvocation(BaseInvocation):
    """Combines two latent tensors using various methods, including frequency domain blending."""

    latentA: LatentsField = InputField(
        default=None,
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    latentB: LatentsField = InputField(
        default=None,
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    method: BLEND_MODES = InputField(
        default="average",
        description="Combination method",
    )
    weighted_combine: bool = InputField(default=False, description="Alter input latents by the provided weights")
    weight_a: float = InputField(
        default=0.5,
        description="Weight for latent A",
        ge=0,
        le=1,
    )
    weight_b: float = InputField(
        default=0.5,
        description="Weight for latent B",
        ge=0,
        le=1,
    )
    scale_to_input_ranges: bool = InputField(default=False, description="Scale output to input ranges")

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latent_in_a = context.tensors.load(self.latentA.latents_name)
        latent_in_b = context.tensors.load(self.latentB.latents_name)
        assert latent_in_a.shape == latent_in_b.shape, "inputs must be same size"

        if self.weighted_combine:
            latent_a = latent_in_a * self.weight_a
            latent_b = latent_in_b * self.weight_b
        else:
            latent_a = latent_in_a
            latent_b = latent_in_b

        max_a = torch.max(latent_a)
        max_b = torch.max(latent_b)
        min_val = min(torch.min(latent_a).item(), torch.min(latent_b).item())
        max_val = max(torch.max(latent_a).item(), torch.max(latent_b).item())

        # Normalize to [0, 1]
        normalized_a = (latent_a - min_val) / (max_val - min_val)
        normalized_b = (latent_b - min_val) / (max_val - min_val)

        # Blend operations (using standard formulas)
        if self.method == "average":
            blended_latent = (normalized_a + normalized_b) / 2
        elif self.method == "add":
            blended_latent = normalized_a + normalized_b
        elif self.method == "subtract":
            blended_latent = normalized_a - normalized_b
        elif self.method == "difference":
            blended_latent = torch.abs(normalized_a - normalized_b)
        elif self.method == "maximum":
            blended_latent = torch.maximum(normalized_a, normalized_b)
        elif self.method == "minimum":
            blended_latent = torch.minimum(normalized_a, normalized_b)
        elif self.method == "multiply":
            blended_latent = normalized_a * normalized_b
        elif self.method == "screen":
            blended_latent = 1 - (1 - normalized_a) * (1 - normalized_b)
        elif self.method == "dodge":
            blended_latent = torch.clamp(normalized_a / (1 - normalized_b + 1e-7), 0, 1)
        elif self.method == "burn":
            blended_latent = torch.clamp(1 - (1 - normalized_a) / (normalized_b + 1e-7), 0, 1)
        elif self.method == "overlay":
            blended_latent = torch.where(
                normalized_a < 0.5, 2 * normalized_a * normalized_b, 1 - 2 * (1 - normalized_a) * (1 - normalized_b)
            )
        elif self.method == "soft_light":
            blended_latent = torch.where(
                normalized_a < 0.5,
                normalized_b - (1 - 2 * normalized_a) * normalized_b * (1 - normalized_b),
                normalized_b + (2 * normalized_a - 1) * (torch.sqrt(normalized_b) - normalized_b),
            )
        elif self.method == "hard_light":
            blended_latent = torch.where(
                normalized_b < 0.5, 2 * normalized_a * normalized_b, 1 - 2 * (1 - normalized_a) * (1 - normalized_b)
            )
        elif self.method == "color_dodge":
            blended_latent = torch.where(
                normalized_b == 1, torch.ones_like(normalized_a), torch.clamp(normalized_a / (1 - normalized_b), 0, 1)
            )
        elif self.method == "color_burn":
            blended_latent = torch.where(
                normalized_b == 0, torch.zeros_like(normalized_a), torch.clamp(1 - (1 - normalized_a) / normalized_b, 0, 1)
            )
        elif self.method == "linear_dodge":
            blended_latent = torch.clamp(normalized_a + normalized_b, 0, 1)
        elif self.method == "linear_burn":
            blended_latent = torch.clamp(normalized_a + normalized_b - 1, 0, 1)
        elif self.method == "frequency_blend":
            blended_latent = self._frequency_blend(normalized_a, normalized_b)
        else:
            raise ValueError(f"Invalid method: {self.method}")

        # Re-expand to original range
        blended_latent = blended_latent * (max_val - min_val) + min_val

        if self.scale_to_input_ranges:
            blended_latent = self._normalize_latent(latent_a, latent_b, blended_latent)

        name = context.tensors.save(tensor=blended_latent)
        return LatentsOutput.build(latents_name=name, latents=blended_latent, seed=None)

    def _frequency_blend(self, latent_a: torch.Tensor, latent_b: torch.Tensor) -> torch.Tensor:
        """Blends two latent tensors in the frequency domain with weights."""
        original_dtype = latent_a.dtype
        latent_a_float = latent_a.float()
        latent_b_float = latent_b.float()

        latent_a_fft = torch.fft.fftn(latent_a_float, dim=(-2, -1))
        latent_b_fft = torch.fft.fftn(latent_b_float, dim=(-2, -1))

        blended_fft = latent_a_fft + latent_b_fft

        blended_latent = torch.fft.ifftn(blended_fft, dim=(-2, -1)).real

        if original_dtype != torch.float32:
            blended_latent = blended_latent.to(original_dtype)

        return blended_latent

    def _normalize_latent(self, latent_a: torch.Tensor, latent_b: torch.Tensor, latent_out: torch.Tensor) -> torch.Tensor:
        """Normalizes the output latent to the range of the input latents."""
        normalized_latent = torch.zeros_like(latent_out)
        for i in range(latent_out.shape[1]):
            channel_a = latent_a[:, i, :, :]
            channel_b = latent_b[:, i, :, :]
            channel_out = latent_out[:, i, :, :]

            min_val_in = min(torch.min(channel_a).item(), torch.min(channel_b).item())
            max_val_in = max(torch.max(channel_a).item(), torch.max(channel_b).item())

            min_val_out = torch.min(channel_out).item()
            max_val_out = torch.max(channel_out).item()

            if max_val_in == min_val_in:
                normalized_channel = torch.zeros_like(channel_out)
            elif max_val_out == min_val_out:
                normalized_channel = torch.full_like(channel_out, min_val_in)
            else:
                normalized_channel = (channel_out - min_val_out) / (max_val_out - min_val_out) * (max_val_in - min_val_in) + min_val_in

            normalized_latent[:, i, :, :] = normalized_channel

        return normalized_latent


@invocation(
    "latent_plot",
    title="Latent Plot",
    tags=["latents", "image", "flux"],
    category="latents",
    version="1.0.0",
)
class LatentPlotInvocation(BaseInvocation, WithMetadata, WithBoard):
    latent: LatentsField = InputField(
        default=None,
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    """Generates plots from latent channels and display in a grid."""
    cell_x_multiplier: float = InputField(default=4.0, description="x size multiplier for grid cells")
    cell_y_multiplier: float = InputField(default=3.0, description="y size multiplier for grid cells")
    histogram_plot: bool = InputField(default=True, description="Plot histogram")
    histogram_bins: int = InputField(default=100, description="Number of bins for the histogram")
    box_plot: bool = InputField(default=True, description="Plot box and whisker")
    stats_plot: bool = InputField(default=True, description="Plot distribution data (mean, std, mode, min, max, dtype)")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        latent = context.tensors.load(self.latent.latents_name)

        if len(latent.shape) != 4:
            raise ValueError("Latent tensor must have 4 dimensions (batch, channels, height, width).")

        channels = latent[0].cpu().float().numpy()
        num_channels = channels.shape[0]
        grid_size = int(math.ceil(math.sqrt(num_channels)))

        fig_plot, axes_plot = plt.subplots(
            grid_size, grid_size, figsize=(grid_size * self.cell_x_multiplier, grid_size * self.cell_y_multiplier)
        )
        axes_plot = axes_plot.flatten()

        for i, channel in enumerate(channels):
            if i < num_channels:
                axes_plot[i].set_title(f"{i}")
                channel_flat = channel.flatten()

                if self.histogram_plot:
                    n, bins, patches = axes_plot[i].hist(channel_flat, bins=self.histogram_bins, alpha=0.6)
                else:
                    n = [0]  # If no histogram, set n to [0] to avoid errors

                if self.box_plot:
                    box_data = axes_plot[i].boxplot(
                        channel_flat,
                        vert=False,
                        positions=[np.max(n) * 0.6 if self.histogram_plot else 0],  # Adjust position based on histogram
                        widths=np.max(n) * 0.08 if self.histogram_plot else 0.08,  # Adjust width based on histogram
                        patch_artist=True,
                        showfliers=False,  # Hide outliers
                        # whis=3.0,  # Adjust whisker length (increase to 2.0)
                    )

                    for box in box_data["boxes"]:
                        box.set_facecolor("lightblue")

                if self.stats_plot:
                    min_val = np.min(channel_flat)
                    max_val = np.max(channel_flat)
                    mean = np.mean(channel_flat)
                    median = np.median(channel_flat)
                    std = np.std(channel_flat)
                    mode = stats.mode(channel_flat, keepdims=True).mode[0]
                    axes_plot[i].annotate(
                        f"min: {min_val:.3f}\nmax: {max_val:.3f}\nmean: {mean:.3f}\nmedian: {median:.3f}\nmode: {mode:.3f}\nstd: {std:.3f}",
                        xy=(0.05, 0.95),
                        xycoords="axes fraction",
                        verticalalignment="top",
                        fontsize=8,
                    )

                    data_type = latent[:, i, :, :].dtype
                    axes_plot[i].annotate(
                        f"type: {data_type}",
                        xy=(0.55, 0.95),
                        xycoords="axes fraction",
                        verticalalignment="top",
                        fontsize=8,
                    )

                # Force y-axis to show integer ticks
                max_y = int(max(n))  # Get the max y-value (histogram height)
                axes_plot[i].yaxis.set_ticks(np.linspace(0, max_y, min(6, max_y + 1)))  # Set up to 6 ticks
                axes_plot[i].yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
            else:
                fig_plot.delaxes(axes_plot[i])

        plt.tight_layout()  # pad=2.0)
        buffer = io.BytesIO()
        fig_plot.savefig(buffer, format="png")
        buffer.seek(0)
        hist_image = Image.open(buffer).convert("RGB")
        plt.close(fig_plot)

        image_dto = context.images.save(image=hist_image)
        return ImageOutput.build(image_dto)


@invocation(
    "latent_channels_to_grid",
    title="Latent channels to grid",
    tags=["latents", "image", "flux"],
    category="latents",
    version="1.0.0",
)
class LatentChannelsToGridInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Generates a scaled grid Images from latent channels"""

    latent: LatentsField = InputField(
        default=None,
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    scale: float = InputField(default=1.0, description="Overall scale factor for the grid.")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        latent = context.tensors.load(self.latent.latents_name)

        if len(latent.shape) != 4:
            raise ValueError("Latent tensor must have 4 dimensions (batch, channels, height, width).")

        channels = latent[0].cpu().float().numpy()
        num_channels = channels.shape[0]
        grid_size = int(math.ceil(math.sqrt(num_channels)))

        original_height, original_width = channels.shape[1], channels.shape[2]
        scaled_height = int(original_height * self.scale)
        scaled_width = int(original_width * self.scale)

        channel_images = []
        for channel in channels:
            min_val = np.min(channel)
            max_val = np.max(channel)
            normalized_channel = (channel - min_val) / (max_val - min_val) if max_val != min_val else np.zeros_like(channel)
            normalized_channel = (normalized_channel * 255).astype(np.uint8)  # Scale to 0-255
            channel_image = Image.fromarray(normalized_channel, mode="L").resize((scaled_width, scaled_height), resample=Image.NEAREST)
            channel_images.append(channel_image)

        grid_width = grid_size * scaled_width
        grid_height = grid_size * scaled_height
        grid_image = Image.new("L", (grid_width, grid_height))

        for i, channel_image in enumerate(channel_images):
            row = i // grid_size
            col = i % grid_size
            grid_image.paste(channel_image, (col * scaled_width, row * scaled_height))

        grid_image_rgb = grid_image.convert("RGB")
        image_dto = context.images.save(image=grid_image_rgb)
        return ImageOutput.build(image_dto)


CONVERT_DTYPE_OPTIONS = Literal["float32", "float16", "bfloat16"]


@invocation(
    "latent_dtype_convert",
    title="Latent Dtype Convert",
    tags=["latents", "dtype", "convert"],
    category="latents",
    version="1.0.0",
)
class LatentDtypeConvertInvocation(BaseInvocation):
    """Converts the dtype of a latent tensor."""

    latent: LatentsField = InputField(
        default=None,
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    dtype: CONVERT_DTYPE_OPTIONS = InputField(
        default="bfloat16",
        description="The target dtype for the latent tensor.",
    )

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        """Converts the dtype of a latent tensor."""
        latent = context.tensors.load(self.latent.latents_name)

        target_dtype = getattr(torch, self.dtype)
        converted_latent = latent.to(target_dtype)

        name = context.tensors.save(tensor=converted_latent)
        return LatentsOutput.build(latents_name=name, latents=converted_latent, seed=None)


@invocation(
    "latent_modify_channels",
    title="Latent Modify Channels",
    tags=["latents", "modify", "channels"],
    category="latents",
    version="1.0.0",
)
class LatentModifyChannelsInvocation(BaseInvocation):
    """Modifies selected channels of a latent tensor using scale and shift."""

    latent: LatentsField = InputField(
        default=None,
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    channels: str = InputField(
        default="all",
        description="Comma-separated list of channel indices to modify. Use 'all' for all channels.",
    )
    scale: float = InputField(
        default=1.0,
        description="Scale factor to apply to the selected channels.",
    )
    shift: float = InputField(
        default=0.0,
        description="Shift value to add to the selected channels.",
    )

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        """Modifies selected channels of a latent tensor."""
        latent = context.tensors.load(self.latent.latents_name)

        if len(latent.shape) != 4:
            raise ValueError("Latent tensor must have 4 dimensions (batch, channels, height, width).")

        num_channels = latent.shape[1]

        if self.channels.lower() == "all":
            channel_indices = list(range(num_channels))
        else:
            try:
                channel_indices = [int(c.strip()) for c in self.channels.split(",")]
            except ValueError:
                raise ValueError("Invalid channel list. Please provide a comma-separated list of integers or 'all'.")

            # Validate channel indices
            invalid_indices = [c for c in channel_indices if not (0 <= c < num_channels)]
            if invalid_indices:
                raise ValueError(f"Invalid channel indices: {invalid_indices}. Valid range: 0 to {num_channels - 1}.")

        modified_latent = latent.clone()  # Create a copy to avoid modifying the original

        for channel_index in channel_indices:
            modified_latent[:, channel_index, :, :] = modified_latent[:, channel_index, :, :] * self.scale + self.shift

        name = context.tensors.save(tensor=modified_latent)
        return LatentsOutput.build(latents_name=name, latents=modified_latent, seed=None)
