"""
========================================================================================
Cascaded Comic Translation Pipeline: Detection / OCR -> Text Machine Translation
========================================================================================
This script implements Pipeline A and Pipeline B (Oracle) from the paper:

  Pipeline A (Cascaded Real-World):
    Comic Speech Bubble Crop ──▶ OCR Engine (Manga-OCR / PaddleOCR) ──▶ Text MT Engine ──▶ Translation

  Pipeline B (Oracle Control):
    Gold Human Transcription ──▶ Text MT Engine ──▶ Translation

Scientific Motivation (RQ3):
  By comparing Pipeline A (OCR + MT) vs Pipeline B (Gold + MT) vs End-to-End VLM,
  we pinpoint exactly where translation errors originate:
    - Character recognition failures on stylized historical comic fonts (OCR error)
    - Bubble reading order and dialogue fragmentation
    - Pure linguistic / semantic translation errors
========================================================================================
"""

import argparse
import json
import os
from typing import Dict, Any, List
from PIL import Image


def perform_ocr_on_bubble(bubble_image_path: str, ocr_engine: str = "manga-ocr") -> str:
    """
    Extracts transcribed text from a cropped speech bubble image.

    Args:
        bubble_image_path: Path to the isolated speech bubble image crop.
        ocr_engine: 'manga-ocr' or 'paddleocr'.

    Returns:
        The extracted dialogue string.
    """
    if not os.path.exists(bubble_image_path):
        return "[OCR Error: Image not found]"

    # In production with manga-ocr installed:
    #   from manga_ocr import MangaOcr
    #   mocr = MangaOcr()
    #   text = mocr(Image.open(bubble_image_path))
    #   return text

    # Friendly placeholder for dry-run testing
    return "Extracted OCR text placeholder from bubble."


def translate_text(
    source_text: str,
    target_lang: str = "es",
    mt_model: str = "meta-llama/Llama-3.1-8B-Instruct"
) -> str:
    """
    Translates extracted speech bubble dialogue using a text-only MT model or LLM.

    Args:
        source_text: English text (either from OCR or gold human transcript).
        target_lang: 'es' (Spanish) or 'fr' (French).
        mt_model: HuggingFace model ID or API endpoint.

    Returns:
        Target language translated string.
    """
    lang_name = "Spanish" if target_lang.lower() == "es" else "French"
    
    # Prompt for text-only machine translation
    prompt = (
        f"You are a professional comic translator. Translate this comic dialogue into natural {lang_name}:\n"
        f"Dialogue: \"{source_text}\"\n"
        f"Provide ONLY the translation:"
    )

    # In production: run Llama-3.1-8B or NLLB-200-3.3B
    return f"[{target_lang.upper()} translation of: '{source_text}']"


def main():
    parser = argparse.ArgumentParser(description="Cascaded Comic Translation Runner (OCR -> MT)")
    parser.add_argument("--data", type=str, default="data/annotations/sample.json",
                        help="Path to annotations JSON file.")
    parser.add_argument("--ocr", type=str, default="manga-ocr", choices=["manga-ocr", "paddleocr"],
                        help="OCR engine for speech bubble text recognition.")
    parser.add_argument("--mt", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Text MT model (e.g. meta-llama/Llama-3.1-8B-Instruct, facebook/nllb-200-3.3B).")
    parser.add_argument("--target_lang", type=str, default="es", choices=["es", "fr"],
                        help="Target language ('es' for Spanish, 'fr' for French).")
    parser.add_argument("--output_dir", type=str, default="outputs/cascaded",
                        help="Directory to save pipeline outputs.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 70)
    print("Cascaded Pipeline Execution (Pipeline A: OCR -> MT & Pipeline B: Oracle)")
    print(f"OCR Engine:      {args.ocr}")
    print(f"MT Model:        {args.mt}")
    print(f"Target Language: {args.target_lang.upper()}")
    print("=" * 70)

    # Sample unit illustrating the two conditions
    sample_unit = {
        "unit_id": "sample_001",
        "gold_transcription": "Halt! Who goes there?",
        "bubble_image_path": "data/crops/bubble_001.jpg"
    }

    # Pipeline A: Cascaded (OCR text -> MT)
    ocr_text = perform_ocr_on_bubble(sample_unit["bubble_image_path"], ocr_engine=args.ocr)
    pipeline_a_trans = translate_text(ocr_text, target_lang=args.target_lang, mt_model=args.mt)

    # Pipeline B: Oracle Cascaded (Gold text -> MT)
    pipeline_b_oracle = translate_text(sample_unit["gold_transcription"], target_lang=args.target_lang, mt_model=args.mt)

    results = {
        "pipeline_a_cascaded": [{
            "unit_id": sample_unit["unit_id"],
            "ocr_text": ocr_text,
            "translation": pipeline_a_trans
        }],
        "pipeline_b_oracle": [{
            "unit_id": sample_unit["unit_id"],
            "gold_text": sample_unit["gold_transcription"],
            "translation": pipeline_b_oracle
        }]
    }

    output_file = os.path.join(args.output_dir, "cascaded_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] Results written to: {output_file}")


if __name__ == "__main__":
    main()
