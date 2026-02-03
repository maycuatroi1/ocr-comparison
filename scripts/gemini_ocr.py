#!/usr/bin/env python3
"""
Gemini Flash OCR Script
Runs on Mac M4 (no CUDA required)
"""

import os
import json
import time
import base64
from pathlib import Path
from datetime import datetime

import google.generativeai as genai
from PIL import Image

# Configuration
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyDzgmUbGw_uwxjyjWh9y7x1EBmjAwaOiUo")
MODEL_NAME = "gemini-2.0-flash"
IMAGE_DIR = Path(__file__).parent.parent / "test-images"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# OCR Prompts
PROMPTS = {
    "standard": "Extract all text from this image. Return only the extracted text, preserving the layout as much as possible.",
    "detailed": """Perform OCR on this image and extract all text content.
Instructions:
1. Extract ALL visible text, including handwritten text
2. Preserve the original layout and structure
3. For tables, maintain the tabular format
4. Note any text that is unclear with [unclear]
5. Include any numbers, symbols, or special characters

Return the extracted text:""",
    "vietnamese": """Trích xuất tất cả văn bản từ hình ảnh này.
Yêu cầu:
1. Trích xuất TẤT CẢ văn bản có thể nhìn thấy
2. Giữ nguyên bố cục và cấu trúc gốc
3. Bao gồm cả văn bản viết tay
4. Đánh dấu những phần không rõ ràng bằng [không rõ]

Trả về văn bản đã trích xuất:"""
}


def setup_gemini():
    """Initialize Gemini API"""
    genai.configure(api_key=GOOGLE_API_KEY)
    return genai.GenerativeModel(MODEL_NAME)


def load_image(image_path: Path) -> Image.Image:
    """Load image from path"""
    return Image.open(image_path)


def perform_ocr(model, image: Image.Image, prompt_type: str = "detailed") -> dict:
    """Perform OCR using Gemini Flash"""
    prompt = PROMPTS.get(prompt_type, PROMPTS["detailed"])

    start_time = time.time()
    try:
        response = model.generate_content([prompt, image])
        elapsed_time = time.time() - start_time

        return {
            "success": True,
            "text": response.text,
            "elapsed_time": elapsed_time,
            "model": MODEL_NAME,
            "prompt_type": prompt_type
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


def process_all_images(model, prompt_type: str = "detailed") -> dict:
    """Process all images in the test-images directory"""
    results = {}

    image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
    image_files = [f for f in IMAGE_DIR.iterdir()
                   if f.suffix.lower() in image_extensions]

    print(f"\nProcessing {len(image_files)} images with Gemini Flash ({MODEL_NAME})")
    print(f"Prompt type: {prompt_type}")
    print("-" * 50)

    for image_path in sorted(image_files):
        print(f"\nProcessing: {image_path.name}")

        image = load_image(image_path)
        result = perform_ocr(model, image, prompt_type)

        results[image_path.name] = {
            **result,
            "image_size": image.size,
            "timestamp": datetime.now().isoformat()
        }

        if result["success"]:
            print(f"  ✓ Completed in {result['elapsed_time']:.2f}s")
            print(f"  Text length: {len(result['text'])} chars")
        else:
            print(f"  ✗ Failed: {result['error']}")

    return results


def save_results(results: dict, filename: str = "gemini_results.json"):
    """Save results to JSON file"""
    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_path}")
    return output_path


def main():
    print("=" * 60)
    print("Gemini Flash OCR Comparison Script")
    print("=" * 60)

    # Setup
    model = setup_gemini()

    # Process with detailed prompt (good for mixed content)
    results_detailed = process_all_images(model, "detailed")
    save_results(results_detailed, "gemini_detailed.json")

    # Process with Vietnamese prompt (better for Vietnamese text)
    results_vietnamese = process_all_images(model, "vietnamese")
    save_results(results_vietnamese, "gemini_vietnamese.json")

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
