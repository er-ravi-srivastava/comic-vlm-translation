"""
End-to-End Multimodal Vision-Language Model Translation Runner
Evaluates across the 4 experimental visual conditions:
1. text_only
2. correct_image
3. mismatched_image
4. noise_image
"""

import argparse
import json
import os
from typing import Dict, Any, List
from PIL import Image

from ..utils.image_transforms import (
    load_image,
    generate_gaussian_noise_image,
    sample_mismatched_image
)


def run_vlm_condition(
    model_name: str,
    unit: Dict[str, Any],
    condition: str,
    target_lang: str = "es",
    all_image_paths: List[str] = None
) -> str:
    """
    Executes inference for a single dialogue unit under a specific visual condition.
    """
    source_text = unit.get("gold_transcription", "")
    prev_ctx = unit.get("previous_context", "")
    next_ctx = unit.get("next_context", "")
    surrounding_context = f"{prev_ctx} ... {next_ctx}".strip()

    # Determine visual input based on condition
    if condition == "text_only":
        image_input = None
    elif condition == "correct_image":
        image_input = load_image(unit["panel_image_path"])
    elif condition == "mismatched_image":
        mismatched_path = sample_mismatched_image(unit["unit_id"], all_image_paths)
        image_input = load_image(mismatched_path)
    elif condition == "noise_image":
        ref_img = load_image(unit["panel_image_path"])
        image_input = generate_gaussian_noise_image(ref_img)
    else:
        raise ValueError(f"Unknown visual condition: {condition}")

    # Standard Prompt formatting
    prompt_text = (
        f"Translate the following comic dialogue bubble into {target_lang.upper()}.\n"
        f"Use visual cues from the scene to resolve gender, address forms, and spatial deixis.\n\n"
        f"Source Dialogue: \"{source_text}\"\n"
        f"Context: \"{surrounding_context}\"\n\n"
        f"Translation:"
    )

    # In production/GPU environment: call model (Qwen2.5-VL / Llama-3.2-Vision / GPT-4o)
    # Placeholder returning standard structure for dry-run testing
    return f"[{condition.upper()} translation placeholder for '{source_text}']"


def main():
    parser = argparse.ArgumentParser(description="End-to-end VLM Comic Translation Runner")
    parser.add_argument("--data", type=str, default="data/annotations/sample.json", help="Path to annotations JSON")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Model identifier")
    parser.add_argument("--conditions", nargs="+", default=["text_only", "correct_image", "mismatched_image", "noise_image"])
    parser.add_argument("--target_lang", type=str, default="es", choices=["es", "fr"])
    parser.add_argument("--output_dir", type=str, default="outputs/predictions")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading dataset from: {args.data}")
    print(f"Evaluating Model: {args.model}")
    print(f"Conditions: {args.conditions}")

    # Results container
    results = {cond: [] for cond in args.conditions}
    
    # Save output placeholder
    out_file = os.path.join(args.output_dir, "results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Evaluation framework initialized. Output mapped to {out_file}")


if __name__ == "__main__":
    main()
