# 2025 skunkworxdark (https://github.com/skunkworxdark)

import io
import math
from typing import List, Literal

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


def validate_channel_indices(channels_str: str, num_channels: int) -> List[int]:
    """Validates and parses channel indices from a comma-separated string."""
    if channels_str.lower() == "all":
        return list(range(num_channels))
    else:
        try:
            channel_indices = [int(c.strip()) for c in channels_str.split(",")]
        except ValueError:
            raise ValueError("Invalid channel list. Please provide a comma-separated list of integers or 'all'.")

        invalid_indices = [c for c in channel_indices if not (0 <= c < num_channels)]
        if invalid_indices:
            raise ValueError(f"Invalid channel indices: {invalid_indices}. Valid range: 0 to {num_channels - 1}.")

        return list(set(channel_indices))


@invocation(
    "latent_average",
    title="Latent Average",
    tags=["latents"],
    category="latents",
    version="1.1.0",
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
    channels: str = InputField(
        default="all",
        description="Comma-separated list of channel indices to average. Use 'all' for all channels.",
    )

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        """Averages two latents"""
        latent_a = context.tensors.load(self.latentA.latents_name)
        latent_b = context.tensors.load(self.latentB.latents_name)
        assert latent_a.shape == latent_b.shape, "inputs must be same size"

        num_channels = latent_a.shape[1]
        channel_indices = validate_channel_indices(self.channels, num_channels)

        # Create a copy of latent_a to modify
        averaged_latent = latent_a.clone()
        for i in channel_indices:
            averaged_latent[:, i, :, :] = (latent_a[:, i, :, :] + latent_b[:, i, :, :]) / 2

        name = context.tensors.save(tensor=averaged_latent)
        return LatentsOutput.build(latents_name=name, latents=averaged_latent, seed=None)


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
    version="1.1.0",
)
class LatentCombineInvocation(BaseInvocation):
    """Combines two latent tensors using various methods"""

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
    channels: str = InputField(
        default="all",
        description="Comma-separated list of channel indices to combine. Use 'all' for all channels.",
    )

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latent_in_a = context.tensors.load(self.latentA.latents_name)
        latent_in_b = context.tensors.load(self.latentB.latents_name)
        assert latent_in_a.shape == latent_in_b.shape, "inputs must be same size"

        num_channels = latent_in_a.shape[1]
        channel_indices = validate_channel_indices(self.channels, num_channels)

        if self.weighted_combine:
            latent_a = latent_in_a * self.weight_a
            latent_b = latent_in_b * self.weight_b
        else:
            latent_a = latent_in_a
            latent_b = latent_in_b

        # Create a copy of latent_in_a to modify
        blended_latent = latent_in_a.clone()

        for i in channel_indices:
            min_val = min(torch.min(latent_a[:, i, :, :]).item(), torch.min(latent_b[:, i, :, :]).item())
            max_val = max(torch.max(latent_a[:, i, :, :]).item(), torch.max(latent_b[:, i, :, :]).item())

            # Normalize to [0, 1]
            normalized_a = (latent_a[:, i, :, :] - min_val) / (max_val - min_val)
            normalized_b = (latent_b[:, i, :, :] - min_val) / (max_val - min_val)

            # Blend operations (using standard formulas)
            if self.method == "average":
                blended_channel = (normalized_a + normalized_b) / 2
            elif self.method == "add":
                blended_channel = normalized_a + normalized_b
            elif self.method == "subtract":
                blended_channel = normalized_a - normalized_b
            elif self.method == "difference":
                blended_channel = torch.abs(normalized_a - normalized_b)
            elif self.method == "maximum":
                blended_channel = torch.maximum(normalized_a, normalized_b)
            elif self.method == "minimum":
                blended_channel = torch.minimum(normalized_a, normalized_b)
            elif self.method == "multiply":
                blended_channel = normalized_a * normalized_b
            elif self.method == "screen":
                blended_channel = 1 - (1 - normalized_a) * (1 - normalized_b)
            elif self.method == "dodge":
                blended_channel = torch.clamp(normalized_a / (1 - normalized_b + 1e-7), 0, 1)
            elif self.method == "burn":
                blended_channel = torch.clamp(1 - (1 - normalized_a) / (normalized_b + 1e-7), 0, 1)
            elif self.method == "overlay":
                blended_channel = torch.where(
                    normalized_a < 0.5, 2 * normalized_a * normalized_b, 1 - 2 * (1 - normalized_a) * (1 - normalized_b)
                )
            elif self.method == "soft_light":
                blended_channel = torch.where(
                    normalized_a < 0.5,
                    normalized_b - (1 - 2 * normalized_a) * normalized_b * (1 - normalized_b),
                    normalized_b + (2 * normalized_a - 1) * (torch.sqrt(normalized_b) - normalized_b),
                )
            elif self.method == "hard_light":
                blended_channel = torch.where(
                    normalized_b < 0.5, 2 * normalized_a * normalized_b, 1 - 2 * (1 - normalized_a) * (1 - normalized_b)
                )
            elif self.method == "color_dodge":
                blended_channel = torch.where(
                    normalized_b == 1, torch.ones_like(normalized_a), torch.clamp(normalized_a / (1 - normalized_b), 0, 1)
                )
            elif self.method == "color_burn":
                blended_channel = torch.where(
                    normalized_b == 0, torch.zeros_like(normalized_a), torch.clamp(1 - (1 - normalized_a) / normalized_b, 0, 1)
                )
            elif self.method == "linear_dodge":
                blended_channel = torch.clamp(normalized_a + normalized_b, 0, 1)
            elif self.method == "linear_burn":
                blended_channel = torch.clamp(normalized_a + normalized_b - 1, 0, 1)
            elif self.method == "frequency_blend":
                blended_channel = self._frequency_blend(normalized_a, normalized_b)
            else:
                raise ValueError(f"Invalid method: {self.method}")

            # Re-expand to original range
            blended_channel = blended_channel * (max_val - min_val) + min_val

            if self.scale_to_input_ranges:
                blended_channel = self._normalize_latent(latent_a[:, i, :, :], latent_b[:, i, :, :], blended_channel)

            blended_latent[:, i, :, :] = blended_channel

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
        min_val_in = min(torch.min(latent_a).item(), torch.min(latent_b).item())
        max_val_in = max(torch.max(latent_a).item(), torch.max(latent_b).item())

        min_val_out = torch.min(latent_out).item()
        max_val_out = torch.max(latent_out).item()

        if max_val_in == min_val_in:
            normalized_channel = torch.zeros_like(latent_out)
        elif max_val_out == min_val_out:
            normalized_channel = torch.full_like(latent_out, min_val_in)
        else:
            normalized_channel = (latent_out - min_val_out) / (max_val_out - min_val_out) * (max_val_in - min_val_in) + min_val_in

        return normalized_channel


@invocation(
    "latent_plot",
    title="Latent Plot",
    tags=["latents", "image", "flux"],
    category="latents",
    version="1.4.0",
)
class LatentPlotInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Generates plots from latent channels and display in a grid."""

    latent: LatentsField = InputField(
        default=None,
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    cell_x_multiplier: float = InputField(default=4.0, description="x size multiplier for grid cells")
    cell_y_multiplier: float = InputField(default=3.0, description="y size multiplier for grid cells")
    histogram_plot: bool = InputField(default=True, description="Plot histogram")
    histogram_bins: int = InputField(default=256, description="Number of bins for the histogram")
    histogram_log_scale: bool = InputField(default=False, description="Use log scale for histogram y-axis")
    box_plot: bool = InputField(default=True, description="Plot box and whisker")
    stats_plot: bool = InputField(default=True, description="Plot distribution data (mean, std, mode, min, max, dtype)")
    title: str = InputField(default="Latent Channel Plots", description="Title to display on the image")
    channels: str = InputField(
        default="all",
        description="Comma-separated list of channel indices to plot. Use 'all' for all channels.",
    )
    common_axis: bool = InputField(default=False, description="Use a common axis scales for all plots")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        latent = context.tensors.load(self.latent.latents_name)

        if len(latent.shape) != 4:
            raise ValueError("Latent tensor must have 4 dimensions (batch, channels, height, width).")

        num_channels = latent.shape[1]
        channel_indices = validate_channel_indices(self.channels, num_channels)
        channels_to_plot = latent[0, channel_indices].cpu().float().numpy()
        num_channels_to_plot = len(channel_indices)

        # Calculate grid size based on the actual number of channels to plot
        grid_size = int(math.ceil(math.sqrt(num_channels_to_plot)))

        # Adjust the number of rows and columns to fit the exact number of plots
        if num_channels_to_plot <= grid_size:
            rows = num_channels_to_plot
            cols = 1
        else:
            rows = (num_channels_to_plot + grid_size - 1) // grid_size
            cols = grid_size

        fig_plot, axes_plot = plt.subplots(rows, cols, figsize=(cols * self.cell_x_multiplier, rows * self.cell_y_multiplier))

        # Flatten the axes array for easier iteration
        if num_channels_to_plot == 1:
            axes_plot = [axes_plot]
        else:
            axes_plot = axes_plot.flatten()

        # Determine common axis limits if requested
        if self.common_axis:
            all_min_val = float("inf")
            all_max_val = float("-inf")
            all_max_y = 0
            for channel in channels_to_plot:
                channel_flat = channel.flatten()
                all_min_val = min(all_min_val, np.min(channel_flat))
                all_max_val = max(all_max_val, np.max(channel_flat))
                if self.histogram_plot:
                    n, bins = np.histogram(channel_flat, bins=self.histogram_bins)
                    all_max_y = max(all_max_y, np.max(n))

            if self.histogram_plot:
                common_max_y = all_max_y * 1.1  # May Y plus a little bit
            else:
                common_max_y = 0

            common_min_x = math.floor(all_min_val)
            common_max_x = math.ceil(all_max_val)

        for i, channel in enumerate(channels_to_plot):
            axes_plot[i].set_title(f"{channel_indices[i]}")
            channel_flat = channel.flatten()

            if self.histogram_plot:
                n, bins, patches = axes_plot[i].hist(channel_flat, bins=self.histogram_bins, alpha=0.6)
                if self.histogram_log_scale:
                    axes_plot[i].set_yscale("log")
                if self.common_axis:
                    axes_plot[i].set_ylim([0, common_max_y])
                    axes_plot[i].set_xlim([common_min_x, common_max_x])
                else:
                    axes_plot[i].set_ylim([0, np.max(n) * 1.1])
            else:
                n = [0]  # If no histogram, set n to [0] to avoid errors

            if self.box_plot:
                # Use a secondary Y axis for the boxplot to avoid scaling issues with histogram
                ax_box = axes_plot[i].twinx()
                ax_box.set_yticks([])  # Hide ticks of the secondary axis
                ax_box.set_ylim(0, 1)  # Define a simple relative scale for positioning

                box_data = ax_box.boxplot(
                    channel_flat,
                    vert=False,
                    positions=[0.4],  # Position in the middle of the secondary Y axis (0 to 1)
                    widths=0.2,  # Relative width on the secondary Y axis
                    patch_artist=True,
                    showfliers=False,  # Hide outliers
                    manage_ticks=False,  # Don't let boxplot manage original X ticks
                )

                for item in ["boxes", "whiskers", "caps", "medians"]:
                    plt.setp(box_data[item], alpha=0.5)

                for box in box_data["boxes"]:
                    box.set_facecolor("lightblue")

            min_val = np.min(channel_flat)
            max_val = np.max(channel_flat)
            if self.stats_plot:
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

                data_type = latent[:, channel_indices[i], :, :].dtype
                axes_plot[i].annotate(
                    f"type: {data_type}",
                    xy=(0.55, 0.95),
                    xycoords="axes fraction",
                    verticalalignment="top",
                    fontsize=8,
                )

            # Force y-axis to show integer ticks
            if self.histogram_log_scale:
                axes_plot[i].yaxis.set_major_locator(ticker.LogLocator(numticks=4))
                axes_plot[i].yaxis.set_major_formatter(ticker.LogFormatter())
            else:
                axes_plot[i].yaxis.set_major_locator(ticker.LinearLocator(numticks=4))

            axes_plot[i].yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))

            if not self.common_axis:
                axes_plot[i].set_xlim([math.floor(min_val), math.ceil(max_val)])

        # Remove any unused subplots
        for j in range(num_channels_to_plot, len(axes_plot)):
            fig_plot.delaxes(axes_plot[j])

        fig_plot.suptitle(self.title, fontsize=16)
        plt.tight_layout()
        buffer = io.BytesIO()
        fig_plot.savefig(buffer, format="png")
        buffer.seek(0)
        hist_image = Image.open(buffer).convert("RGB")
        plt.close(fig_plot)

        image_dto = context.images.save(image=hist_image)
        return ImageOutput.build(image_dto)


BIT_DEPTH_OPTIONS = Literal["8", "16"]


@invocation(
    "latent_channels_to_grid",
    title="Latent Channels to Grid",
    tags=["latents", "image", "flux"],
    category="latents",
    version="1.1.0",
)
class LatentChannelsToGridInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Generates a scaled grid Images from latent channels"""

    latent: LatentsField = InputField(
        default=None,
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    scale: float = InputField(default=1.0, description="Overall scale factor for the grid.")
    normalize_channels: bool = InputField(default=False, description="Normalize all channels using a common min/max range.")
    output_bit_depth: BIT_DEPTH_OPTIONS = InputField(default="8", description="Output as 8-bit or 16-bit grayscale.")

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

        # Determine global min/max if common scaling is enabled
        global_min = 0
        global_max = 0
        if self.normalize_channels:
            global_min = np.min(channels)
            global_max = np.max(channels)

        # Determine output parameters based on bit depth
        if self.output_bit_depth == "16":
            target_max_val = 65535
            pil_dtype = np.uint16
            pil_mode = "I;16"
            # pil_new_mode = "I"  # Image.new needs "I" for 32-bit int, then we can treat as 16-bit
        else:  # Default to 8-bit
            target_max_val = 255
            pil_dtype = np.uint8
            pil_mode = "L"
            # pil_new_mode = "L"

        channel_images = []
        for channel in channels:
            min_val = global_min if self.normalize_channels else np.min(channel)
            max_val = global_max if self.normalize_channels else np.max(channel)

            normalized_channel = (channel - min_val) / (max_val - min_val)
            channel_image = Image.fromarray((normalized_channel * target_max_val).astype(pil_dtype), mode=pil_mode).resize(
                (scaled_width, scaled_height), resample=Image.NEAREST
            )

            channel_images.append(channel_image)

        grid_width = grid_size * scaled_width
        grid_height = grid_size * scaled_height

        grid_image = Image.new(pil_mode, (grid_width, grid_height))

        for i, channel_image in enumerate(channel_images):
            row = i // grid_size
            col = i % grid_size
            grid_image.paste(channel_image, (col * scaled_width, row * scaled_height))

        image_dto = context.images.save(image=grid_image)
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
    title="Latent Modify",
    tags=["latents", "modify", "channels"],
    category="latents",
    version="1.1.0",
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

        channel_indices = validate_channel_indices(self.channels, num_channels)

        modified_latent = latent.clone()  # Create a copy to avoid modifying the original

        for channel_index in channel_indices:
            modified_latent[:, channel_index, :, :] = modified_latent[:, channel_index, :, :] * self.scale + self.shift

        name = context.tensors.save(tensor=modified_latent)
        return LatentsOutput.build(latents_name=name, latents=modified_latent, seed=None)


MATCHING_METHODS = Literal["histogram", "std_dev", "mean", "std_dev+mean", "cdf", "moment", "range", "std_dev+mean+range"]


@invocation(
    "latent_match",
    title="Latent Match",
    tags=["latents", "match"],
    category="latents",
    version="1.1.0",
)
class LatentMatchInvocation(BaseInvocation):
    """
    Matches the distribution of one latent tensor to that of another,
    using either histogram matching or standard deviation matching.
    """

    input_latent: LatentsField = InputField(
        default=None,
        description="The input latent tensor to be matched.",
        input=Input.Connection,
    )
    reference_latent: LatentsField = InputField(
        default=None,
        description="The reference latent tensor to match against.",
        input=Input.Connection,
    )
    method: MATCHING_METHODS = InputField(
        default="histogram",
        description="The matching method to use.",
    )
    channels: str = InputField(
        default="all",
        description="Comma-separated list of channel indices to match. Use 'all' for all channels.",
    )

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        """
        Matches the distribution of the input latent to the reference latent
        using the specified method.
        """
        input_latent = context.tensors.load(self.input_latent.latents_name)
        reference_latent = context.tensors.load(self.reference_latent.latents_name)

        assert input_latent.shape == reference_latent.shape, "Input and reference latents must have the same shape."

        num_channels = input_latent.shape[1]

        channel_indices = validate_channel_indices(self.channels, num_channels)

        if self.method == "histogram":
            matched_latent = self._histogram_match(input_latent, reference_latent, channel_indices)
        elif self.method == "std_dev":
            matched_latent = self._match_std_dev(input_latent, reference_latent, channel_indices)
        elif self.method == "mean":
            matched_latent = self._match_mean(input_latent, reference_latent, channel_indices)
        elif self.method == "std_dev+mean":
            matched_latent = self._match_std_dev_mean(input_latent, reference_latent, channel_indices)
        elif self.method == "cdf":
            matched_latent = self._match_cdf(input_latent, reference_latent, channel_indices)
        elif self.method == "moment":
            matched_latent = self._match_moment(input_latent, reference_latent, channel_indices)
        elif self.method == "range":
            matched_latent = self._match_range(input_latent, reference_latent, channel_indices)
        elif self.method == "std_dev+mean+range":
            matched_latent = self._match_std_dev_mean_range(input_latent, reference_latent, channel_indices)
        else:
            raise ValueError(f"Invalid matching method: {self.method}")

        name = context.tensors.save(tensor=matched_latent)
        return LatentsOutput.build(latents_name=name, latents=matched_latent, seed=None)

    def _histogram_match(self, input_latent: torch.Tensor, reference_latent: torch.Tensor, channel_indices: List[int]) -> torch.Tensor:
        """
        Matches the histogram of the input latent to the reference latent.

        Args:
            input_latent: The input latent tensor.
            reference_latent: The reference latent tensor.

        Returns:
            A new latent tensor with the histogram of the reference latent.
        """
        # Create a copy of input_latent to modify
        matched_latent = input_latent.clone()
        for i in channel_indices:  # Iterate over channels
            input_channel = input_latent[:, i, :, :].cpu().float().numpy().flatten()
            reference_channel = reference_latent[:, i, :, :].cpu().float().numpy().flatten()

            # Compute histograms
            input_hist, input_bins = np.histogram(input_channel, bins=256, density=True)
            reference_hist, reference_bins = np.histogram(reference_channel, bins=256, density=True)

            # Compute cumulative distribution functions (CDFs)
            input_cdf = np.cumsum(input_hist)
            reference_cdf = np.cumsum(reference_hist)

            # Normalize CDFs
            input_cdf = input_cdf / input_cdf[-1]
            reference_cdf = reference_cdf / reference_cdf[-1]

            # Create interpolation function
            interp_values = np.interp(input_cdf, reference_cdf, reference_bins[:-1])

            # Apply the interpolation to the input channel
            matched_channel_values = np.interp(input_channel, input_bins[:-1], interp_values)
            matched_channel_tensor = (
                torch.from_numpy(matched_channel_values.reshape(input_latent.shape[0], 1, input_latent.shape[2], input_latent.shape[3]))
                .float()
                .to(dtype=input_latent.dtype, device=input_latent.device)
            )
            matched_latent[:, i, :, :] = matched_channel_tensor

        return matched_latent

    def _match_std_dev(self, input_latent: torch.Tensor, reference_latent: torch.Tensor, channel_indices: List[int]) -> torch.Tensor:
        """
        Matches the standard deviation of the input latent to the reference latent.

        Args:
            input_latent: The input latent tensor.
            reference_latent: The reference latent tensor.

        Returns:
            A new latent tensor with the standard deviation of the reference latent.
        """
        # Create a copy of input_latent to modify
        matched_latent = input_latent.clone()
        for i in channel_indices:  # Iterate over channels
            input_channel = input_latent[:, i, :, :]
            reference_channel = reference_latent[:, i, :, :]

            input_std = torch.std(input_channel)
            reference_std = torch.std(reference_channel)

            # Scale the input channel to match the reference standard deviation
            if input_std != 0:
                scaled_channel = (input_channel / input_std) * reference_std
            else:
                scaled_channel = torch.zeros_like(input_channel)
            matched_latent[:, i, :, :] = scaled_channel

        return matched_latent

    def _match_mean(self, input_latent: torch.Tensor, reference_latent: torch.Tensor, channel_indices: List[int]) -> torch.Tensor:
        """
        Matches the mean of the input latent to the reference latent.
        """
        # Create a copy of input_latent to modify
        matched_latent = input_latent.clone()
        for i in channel_indices:  # Iterate over specified channels
            input_channel = input_latent[:, i, :, :]
            reference_channel = reference_latent[:, i, :, :]

            input_mean = torch.mean(input_channel)
            reference_mean = torch.mean(reference_channel)

            # Shift the input channel to match the reference mean
            shifted_channel = input_channel - input_mean + reference_mean
            matched_latent[:, i, :, :] = shifted_channel

        return matched_latent

    def _match_std_dev_mean(self, input_latent: torch.Tensor, reference_latent: torch.Tensor, channel_indices: List[int]) -> torch.Tensor:
        """
        Matches the standard deviation and mean of the input latent to the reference latent.
        """
        # Create a copy of input_latent to modify
        matched_latent = input_latent.clone()
        for i in channel_indices:  # Iterate over specified channels
            input_channel = input_latent[:, i, :, :]
            reference_channel = reference_latent[:, i, :, :]

            input_std = torch.std(input_channel)
            reference_std = torch.std(reference_channel)
            reference_mean = torch.mean(reference_channel)

            # Scale the input channel to match the reference standard deviation
            if input_std != 0:
                scaled_channel = (input_channel / input_std) * reference_std
            else:
                scaled_channel = torch.zeros_like(input_channel)

            # Shift the scaled channel to match the reference mean
            shifted_channel = scaled_channel - torch.mean(scaled_channel) + reference_mean
            matched_latent[:, i, :, :] = shifted_channel

        return matched_latent

    def _match_cdf(self, input_latent: torch.Tensor, reference_latent: torch.Tensor, channel_indices: List[int]) -> torch.Tensor:
        """
        Matches the cumulative distribution function (CDF) of the input latent to the reference latent.
        """
        # Create a copy of input_latent to modify
        matched_latent = input_latent.clone()
        for i in channel_indices:  # Iterate over specified channels
            input_channel = input_latent[:, i, :, :].cpu().float().numpy().flatten()
            reference_channel = reference_latent[:, i, :, :].cpu().float().numpy().flatten()

            # Compute CDFs
            input_sorted = np.sort(input_channel)
            reference_sorted = np.sort(reference_channel)

            input_cdf = np.arange(len(input_sorted)) / (len(input_sorted) - 1)
            reference_cdf = np.arange(len(reference_sorted)) / (len(reference_sorted) - 1)

            # Create interpolation function
            interp_values = np.interp(input_cdf, reference_cdf, reference_sorted)

            # Apply the interpolation to the input channel
            matched_channel_values = np.interp(input_channel, input_sorted, interp_values)
            matched_channel_tensor = (
                torch.from_numpy(matched_channel_values.reshape(input_latent.shape[0], 1, input_latent.shape[2], input_latent.shape[3]))
                .float()
                .to(dtype=input_latent.dtype, device=input_latent.device)
            )
            matched_latent[:, i, :, :] = matched_channel_tensor

        return matched_latent

    def _match_moment(self, input_latent: torch.Tensor, reference_latent: torch.Tensor, channel_indices: List[int]) -> torch.Tensor:
        """
        Matches the mean, standard deviation, skewness, and kurtosis of the input latent to the reference latent.
        """
        # Create a copy of input_latent to modify
        matched_latent = input_latent.clone()
        for i in channel_indices:  # Iterate over specified channels
            input_channel = input_latent[:, i, :, :].cpu().float().numpy().flatten()
            reference_channel = reference_latent[:, i, :, :].cpu().float().numpy().flatten()

            input_mean = np.mean(input_channel)
            reference_mean = np.mean(reference_channel)
            input_std = np.std(input_channel)
            reference_std = np.std(reference_channel)
            input_skew = stats.skew(input_channel)
            reference_skew = stats.skew(reference_channel)
            input_kurt = stats.kurtosis(input_channel)
            reference_kurt = stats.kurtosis(reference_channel)

            # Normalize input channel
            normalized_input = (input_channel - input_mean) / input_std if input_std != 0 else np.zeros_like(input_channel)

            # Scale and shift to match reference moments
            if reference_std != 0:
                matched_channel_values = (normalized_input * reference_std) + reference_mean
            else:
                matched_channel_values = np.full_like(input_channel, reference_mean)

            # Attempt to match skew and kurtosis (more complex)
            # This part is simplified and might not perfectly match skew and kurtosis
            matched_channel_values = (matched_channel_values - np.mean(matched_channel_values)) / np.std(matched_channel_values)
            matched_channel_values = (matched_channel_values * (reference_skew / input_skew if input_skew != 0 else 1)) + (
                reference_kurt / input_kurt if input_kurt != 0 else 1
            )

            matched_channel_tensor = (
                torch.from_numpy(matched_channel_values.reshape(input_latent.shape[0], 1, input_latent.shape[2], input_latent.shape[3]))
                .float()
                .to(dtype=input_latent.dtype, device=input_latent.device)
            )
            matched_latent[:, i, :, :] = matched_channel_tensor

        return matched_latent

    def _match_range(self, input_latent: torch.Tensor, reference_latent: torch.Tensor, channel_indices: List[int]) -> torch.Tensor:
        """
        Matches the minimum and maximum values of the input latent to the reference latent.
        """
        # Create a copy of input_latent to modify
        matched_latent = input_latent.clone()
        for i in channel_indices:  # Iterate over specified channels
            input_channel = input_latent[:, i, :, :]
            reference_channel = reference_latent[:, i, :, :]

            input_min = torch.min(input_channel)
            input_max = torch.max(input_channel)
            reference_min = torch.min(reference_channel)
            reference_max = torch.max(reference_channel)

            if input_max == input_min:
                matched_channel = torch.full_like(input_channel, reference_min)
            elif reference_max == reference_min:
                matched_channel = torch.full_like(input_channel, reference_min)
            else:
                # Scale and shift the input channel to match the reference range
                scaled_channel = (input_channel - input_min) / (input_max - input_min)
                matched_channel = scaled_channel * (reference_max - reference_min) + reference_min

            matched_latent[:, i, :, :] = matched_channel

        return matched_latent

    def _match_std_dev_mean_range(
        self, input_latent: torch.Tensor, reference_latent: torch.Tensor, channel_indices: List[int]
    ) -> torch.Tensor:
        """
        Matches the standard deviation, mean, and range of the input latent to the reference latent.
        """
        # Create a copy of input_latent to modify
        matched_latent = input_latent.clone()
        for i in channel_indices:  # Iterate over specified channels
            input_channel = input_latent[:, i, :, :]
            reference_channel = reference_latent[:, i, :, :]

            # 1. Match std_dev and mean
            input_std = torch.std(input_channel)
            reference_std = torch.std(reference_channel)
            reference_mean = torch.mean(reference_channel)

            if input_std != 0:
                scaled_channel = (input_channel / input_std) * reference_std
            else:
                scaled_channel = torch.zeros_like(input_channel)

            shifted_channel = scaled_channel - torch.mean(scaled_channel) + reference_mean

            # 2. Match range
            input_min = torch.min(shifted_channel)
            input_max = torch.max(shifted_channel)
            reference_min = torch.min(reference_channel)
            reference_max = torch.max(reference_channel)

            if input_max == input_min:
                matched_channel = torch.full_like(shifted_channel, reference_min.item())
            elif reference_max == reference_min:
                matched_channel = torch.full_like(shifted_channel, reference_min.item())
            else:
                scaled_range_channel = (shifted_channel - input_min) / (input_max - input_min)
                matched_channel = scaled_range_channel * (reference_max - reference_min) + reference_min

            matched_latent[:, i, :, :] = matched_channel

        return matched_latent
