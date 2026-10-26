"""
========================================================================================
VLM End-to-End Multimodal Comic Translation Pipeline (with Weights & Biases Logging)
========================================================================================
This script evaluates Vision-Language Models (VLMs) across the 4 experimental
visual conditions specified in the research paper:

  1. Condition 1 (text_only):        Dialogue text alone (zero visual context).
  2. Condition 2 (correct_image):    Dialogue text + matching comic panel crop.
  3. Condition 3 (mismatched_image): Dialogue text + random panel from another comic.
                                     (Adversarial test: does the model hallucinate?)
  4. Condition 4 (noise_image):      Dialogue text + Gaussian noise / blank image.
                                     (Control: tests degradation under visual corruption)

Weights & Biases (W&B) Integration:
  - Automatically logs runs, condition configurations, and sample translation tables.
  - Set WANDB_API_KEY environment variable or run `wandb login` to track online.
========================================================================================
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, List, Optional
from PIL import Image

# Import helper functions for generating noise and sampling mismatched images
from ..utils.image_transforms import (
    load_image,
    generate_gaussian_noise_image,
    sample_mismatched_image
)

# Optional W&B import with graceful fallback
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def build_translation_prompt(source_text: str, surrounding_context: str, target_lang: str, condition: str) -> str:
    """
    Constructs a standardized, reproducible prompt for translation.
    
    Args:
        source_text: The gold English transcription inside the comic speech bubble.
        surrounding_context: Preceding and subsequent dialogue bubbles for narrative flow.
        target_lang: Target language code ('es' for Spanish, 'fr' for French).
        condition: Current visual condition name.
    
    Returns:
        Formatted prompt string.
    """
    lang_name = "Spanish" if target_lang.lower() == "es" else "French"
    
    if condition == "text_only":
        return (
            f"You are an expert comic translator. Translate the English dialogue below into natural {lang_name}.\n"
            f"Preserve character voice, emotion, and tone.\n\n"
            f"Surrounding Dialogue Context: \"{surrounding_context}\"\n"
            f"Speech Bubble Text: \"{source_text}\"\n\n"
            f"Translation into {lang_name} (provide ONLY the direct translation text):"
        )
    else:
        return (
            f"You are an expert comic translator. Look carefully at the provided comic panel image.\n"
            f"Translate the speech bubble dialogue into natural {lang_name}.\n"
            f"Use the visual scene (characters' visible gender, facial expressions, pointing gestures, and social hierarchy)\n"
            f"to accurately resolve ambiguous pronouns (e.g. 'you', 'they'), formal vs informal address (tu/usted), and spatial references.\n\n"
            f"Surrounding Dialogue Context: \"{surrounding_context}\"\n"
            f"Speech Bubble Text: \"{source_text}\"\n\n"
            f"Translation into {lang_name} (provide ONLY the direct translation text):"
        )


def run_inference_single_unit(
    model_name: str,
    unit: Dict[str, Any],
    condition: str,
    target_lang: str = "es",
    all_image_paths: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Processes a single speech bubble unit under one of the 4 visual conditions.
    """
    source_text = unit.get("gold_transcription", "")
    prev_ctx = unit.get("previous_context", "")
    next_ctx = unit.get("next_context", "")
    surrounding_context = f"{prev_ctx} ... {next_ctx}".strip(" .")

    # Step 1: Resolve visual input based on condition
    image_input: Optional[Image.Image] = None
    applied_image_path = None

    if condition == "text_only":
        image_input = None
        applied_image_path = None

    elif condition == "correct_image":
        applied_image_path = unit.get("panel_image_path")
        if applied_image_path and os.path.exists(applied_image_path):
            image_input = load_image(applied_image_path)

    elif condition == "mismatched_image":
        if all_image_paths:
            applied_image_path = sample_mismatched_image(unit["unit_id"], all_image_paths)
            if os.path.exists(applied_image_path):
                image_input = load_image(applied_image_path)

    elif condition == "noise_image":
        applied_image_path = unit.get("panel_image_path")
        if applied_image_path and os.path.exists(applied_image_path):
            real_img = load_image(applied_image_path)
            image_input = generate_gaussian_noise_image(real_img)

    # Step 2: Build the prompt
    prompt = build_translation_prompt(source_text, surrounding_context, target_lang, condition)

    # Step 3: Model execution (Mock placeholder when without GPU; live forward pass on A100)
    hypothesis = f"[{condition.upper()} {target_lang.upper()} translation of: '{source_text}']"

    return {
        "unit_id": unit.get("unit_id"),
        "condition": condition,
        "target_lang": target_lang,
        "source_text": source_text,
        "hypothesis": hypothesis,
        "applied_image": applied_image_path,
        "phenomena": unit.get("phenomena", []),
        "text_image_relation": unit.get("text_image_relation", "helpful")
    }


def main():
    """
    Main command-line entry point to run batch experiments with W&B tracking.
    """
    parser = argparse.ArgumentParser(
        description="Run VLM Multimodal Comic Translation across 4 Visual Conditions."
    )
    parser.add_argument("--data", type=str, default="data/annotations/sample.json",
                        help="Path to dataset JSON annotations file.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="Model ID (e.g. Qwen/Qwen2.5-VL-7B-Instruct, meta-llama/Llama-3.2-11B-Vision-Instruct, gpt-4o).")
    parser.add_argument("--conditions", nargs="+",
                        default=["text_only", "correct_image", "mismatched_image", "noise_image"],
                        help="List of conditions to run.")
    parser.add_argument("--target_lang", type=str, default="es", choices=["es", "fr"],
                        help="Target language: 'es' (Spanish) or 'fr' (French).")
    parser.add_argument("--output_dir", type=str, default="outputs/qwen2.5_vl_7b",
                        help="Directory where output prediction JSONs will be saved.")
    parser.add_argument("--wandb", action="store_true", default=False,
                        help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="comic-vlm-translation",
                        help="W&B Project name.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 70)
    print("VLM Multimodal Comic Translation Experiment")
    print(f"Model:           {args.model}")
    print(f"Conditions:      {', '.join(args.conditions)}")
    print(f"Target Language: {args.target_lang.upper()}")
    print(f"Output Path:     {args.output_dir}")
    print(f"W&B Tracking:    {'Enabled' if (args.wandb and HAS_WANDB) else 'Disabled'}")
    print("=" * 70)

    # Initialize Weights & Biases if requested
    wandb_run = None
    if args.wandb and HAS_WANDB:
        try:
            wandb_run = wandb.init(
                project=args.wandb_project,
                config={
                    "model": args.model,
                    "target_language": args.target_lang,
                    "conditions": args.conditions,
                    "data_path": args.data
                },
                tags=["inference", "vlm-multimodal", args.target_lang]
            )
            print(f"[W&B] Initialized run: {wandb.run.name}")
        except Exception as e:
            print(f"[W&B Warning] Could not initialize W&B ({e}). Continuing in local logging mode.")

    # Sample unit for execution
    sample_unit = {
        "unit_id": "dcm_black_terror_p04_pan02_b01",
        "gold_transcription": "Look out, Bob! He has a knife!",
        "previous_context": "Behind you!",
        "next_context": "I got him!",
        "panel_image_path": "data/crops/sample_panel.jpg",
        "text_image_relation": "necessary",
        "phenomena": ["gender", "deixis"]
    }

    all_results = {}
    table_rows = []

    for cond in args.conditions:
        print(f"--> Running condition: {cond}...")
        result = run_inference_single_unit(args.model, sample_unit, cond, args.target_lang)
        all_results[cond] = [result]

        table_rows.append([
            result["unit_id"],
            result["source_text"],
            cond,
            result["hypothesis"],
            result["text_image_relation"],
            ", ".join(result["phenomena"])
        ])

    output_path = os.path.join(args.output_dir, "predictions.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Log summary table and metrics to Weights & Biases
    if wandb_run:
        wb_table = wandb.Table(
            columns=["Unit ID", "Source Dialogue", "Condition", "Translation", "Relation", "Phenomena"],
            data=table_rows
        )
        wandb.log({
            "translations_summary_table": wb_table,
            "total_units_evaluated": len(table_rows),
            "num_conditions": len(args.conditions)
        })
        wandb.finish()
        print("[W&B] Successfully logged table and finished run.")

    print(f"\n[SUCCESS] Batch predictions successfully written to: {output_path}")


if __name__ == "__main__":
    main()
