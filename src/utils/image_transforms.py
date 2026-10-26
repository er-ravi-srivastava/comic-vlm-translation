"""
Image transformation utilities for the 4 experimental visual conditions:
1. Text-Only (No image)
2. Text + Correct Image
3. Text + Mismatched Image (Adversarial control from another comic)
4. Text + Noise / Blank Image
"""

import random
from typing import List, Optional
import numpy as np
from PIL import Image


def load_image(image_path: str) -> Image.Image:
    """Safely loads an image in RGB mode."""
    return Image.open(image_path).convert("RGB")


def generate_gaussian_noise_image(reference_image: Image.Image, mean: float = 128.0, std: float = 64.0) -> Image.Image:
    """Generates an isotropic Gaussian noise image matching the dimensions of the reference image."""
    width, height = reference_image.size
    noise = np.random.normal(loc=mean, scale=std, size=(height, width, 3))
    noise = np.clip(noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noise, mode="RGB")


def generate_blank_image(reference_image: Image.Image, color: tuple = (128, 128, 128)) -> Image.Image:
    """Generates a solid neutral gray blank image matching the reference dimensions."""
    return Image.new("RGB", reference_image.size, color=color)


def sample_mismatched_image(current_unit_id: str, all_image_paths: List[str], seed: Optional[int] = 42) -> str:
    """
    Selects an adversarial mismatched image from a different scene or comic issue.
    Ensures the sampled image does not match the current unit's image.
    """
    if seed is not None:
        random.seed(seed)
    candidates = [p for p in all_image_paths if current_unit_id not in p]
    if not candidates:
        raise ValueError("Insufficient image candidate pool for mismatched sampling.")
    return random.choice(candidates)
