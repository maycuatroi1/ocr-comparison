#!/usr/bin/env python3
"""
DeepSeek-OCR-2 Script
Requires CUDA - Run on GPU server or Google Colab
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

# Configuration
MODEL_NAME = "deepseek-ai/DeepSeek-OCR-2"
IMAGE_DIR = Path(__file__).parent.parent / "test-images"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# OCR Prompts for DeepSeek
PROMPTS = {
    "standard": "<image>\nConvert this image to text.",
    "markdown": "<image>\n<|grounding|>Convert this document to markdown format.",
    "detailed": "<image>\nExtract all text from this image, including handwritten content. Preserve the layout.",
    "vietnamese": "<image>\nTrích xuất tất cả văn bản từ hình ảnh này, bao gồm cả chữ viết tay."
}


def check_cuda():
    """Check CUDA availability"""
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available!")
        print("DeepSeek-OCR-2 requires CUDA for optimal performance.")
        print("Consider using Google Colab with GPU runtime.")
        return False

    print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    return True


def load_model():
    """Load DeepSeek-OCR-2 model"""
    print(f"Loading model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    return model, tokenizer


def perform_ocr(model, tokenizer, image_path: Path, prompt_type: str = "detailed") -> dict:
    """Perform OCR using DeepSeek-OCR-2"""
    prompt = PROMPTS.get(prompt_type, PROMPTS["detailed"])

    start_time = time.time()
    try:
        # Load image
        image = Image.open(image_path).convert("RGB")

        # Perform inference
        result = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=str(image_path)
        )

        elapsed_time = time.time() - start_time

        return {
            "success": True,
            "text": result,
            "elapsed_time": elapsed_time,
            "model": MODEL_NAME,
            "prompt_type": prompt_type,
            "image_size": image.size
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        return {
            "success": False,
            "error": str(e),
            "elapsed_time": elapsed_time,
            "model": MODEL_NAME,
            "prompt_type": prompt_type
        }


def process_all_images(model, tokenizer, prompt_type: str = "detailed") -> dict:
    """Process all images in the test-images directory"""
    results = {}

    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    image_files = [f for f in IMAGE_DIR.iterdir()
                   if f.suffix.lower() in image_extensions]

    print(f"\nProcessing {len(image_files)} images with DeepSeek-OCR-2")
    print(f"Prompt type: {prompt_type}")
    print("-" * 50)

    for image_path in sorted(image_files):
        print(f"\nProcessing: {image_path.name}")

        result = perform_ocr(model, tokenizer, image_path, prompt_type)

        results[image_path.name] = {
            **result,
            "timestamp": datetime.now().isoformat()
        }

        if result["success"]:
            print(f"  ✓ Completed in {result['elapsed_time']:.2f}s")
            print(f"  Text length: {len(result['text'])} chars")
        else:
            print(f"  ✗ Failed: {result['error']}")

    return results


def save_results(results: dict, filename: str = "deepseek_results.json"):
    """Save results to JSON file"""
    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_path}")
    return output_path


def main():
    print("=" * 60)
    print("DeepSeek-OCR-2 Comparison Script")
    print("=" * 60)

    # Check CUDA
    if not check_cuda():
        print("\nExiting - please run on a CUDA-enabled system")
        return

    # Load model
    model, tokenizer = load_model()

    # Process with detailed prompt
    results_detailed = process_all_images(model, tokenizer, "detailed")
    save_results(results_detailed, "deepseek_detailed.json")

    # Process with Vietnamese prompt
    results_vietnamese = process_all_images(model, tokenizer, "vietnamese")
    save_results(results_vietnamese, "deepseek_vietnamese.json")

    # Process with markdown/grounding mode
    results_markdown = process_all_images(model, tokenizer, "markdown")
    save_results(results_markdown, "deepseek_markdown.json")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, result in results_detailed.items():
        print(f"\n{name}:")
        if result["success"]:
            print(f"  Time: {result['elapsed_time']:.2f}s")
            print(f"  Text preview: {result['text'][:200]}...")
        else:
            print(f"  Error: {result['error']}")


if __name__ == "__main__":
    main()
