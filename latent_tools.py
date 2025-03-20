# 2025 skunkworxdark (https://github.com/skunkworxdark)

import io
import math
from typing import Literal

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image

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
    title="Combine Latents",
    tags=["latents", "combine"],
    category="latents",
    version="0.3.0",
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
    "latent_histogram",
    title="Latent histograms",
    tags=["latents", "image", "flux"],
    category="latents",
    version="1.0.0",
)
class LatentHistogramsInvocation(BaseInvocation, WithMetadata, WithBoard):
    latent: LatentsField = InputField(
        default=None,
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    hist_bins: int = InputField(default=100, description="Number of bins for the histogram")
    cell_size_multiplier: int = InputField(default=3, description="size multiplier for grid cells")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        """Generates histograms from latent channels and display in a grid."""
        latent = context.tensors.load(self.latent.latents_name)

        if len(latent.shape) != 4:
            raise ValueError("Latent tensor must have 4 dimensions (batch, channels, height, width).")

        channels = latent[0].cpu().float().numpy()
        num_channels = channels.shape[0]
        grid_size = int(math.ceil(math.sqrt(num_channels)))

        fig_hist, axes_hist = plt.subplots(
            grid_size, grid_size, figsize=(grid_size * self.cell_size_multiplier, grid_size * self.cell_size_multiplier)
        )
        axes_hist = axes_hist.flatten()

        for i, channel in enumerate(channels):
            if i < num_channels:
                axes_hist[i].hist(channel.flatten(), bins=self.hist_bins)
                axes_hist[i].set_title(f"{i}")
            else:
                fig_hist.delaxes(axes_hist[i])

        plt.tight_layout()
        buffer = io.BytesIO()
        fig_hist.savefig(buffer, format="png")
        buffer.seek(0)
        hist_image = Image.open(buffer).convert("RGB")
        plt.close(fig_hist)

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
