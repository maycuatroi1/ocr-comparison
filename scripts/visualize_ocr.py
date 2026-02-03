#!/usr/bin/env python3
"""
OCR Visualization Script
Visualizes OCR results directly on images with bounding boxes
"""

import os
import json
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import google.generativeai as genai

# Configuration
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyDzgmUbGw_uwxjyjWh9y7x1EBmjAwaOiUo")
MODEL_NAME = "gemini-2.0-flash"
IMAGE_DIR = Path(__file__).parent.parent / "test-images"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "visualized"

# Prompt for OCR with bounding boxes
BBOX_PROMPT = """Analyze this image and extract all text with their bounding box coordinates.

For each text element found, return in this exact JSON format:
{
  "texts": [
    {
      "text": "the extracted text",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.95
    }
  ]
}

Where bbox coordinates are:
- x1, y1: top-left corner (as percentage of image width/height, 0-100)
- x2, y2: bottom-right corner (as percentage of image width/height, 0-100)

Important:
- Extract ALL visible text including handwritten text
- Group text logically (words or short phrases)
- Return ONLY valid JSON, no other text
"""

BBOX_PROMPT_VI = """Phân tích hình ảnh này và trích xuất tất cả văn bản cùng với tọa độ bounding box.

Với mỗi phần tử văn bản tìm thấy, trả về theo định dạng JSON chính xác sau:
{
  "texts": [
    {
      "text": "văn bản trích xuất",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.95
    }
  ]
}

Trong đó tọa độ bbox là:
- x1, y1: góc trên bên trái (theo phần trăm chiều rộng/cao của ảnh, 0-100)
- x2, y2: góc dưới bên phải (theo phần trăm chiều rộng/cao của ảnh, 0-100)

Quan trọng:
- Trích xuất TẤT CẢ văn bản có thể nhìn thấy bao gồm cả chữ viết tay
- Nhóm văn bản một cách logic (từ hoặc cụm từ ngắn)
- CHỈ trả về JSON hợp lệ, không có văn bản khác
"""


def setup_gemini():
    """Initialize Gemini API"""
    genai.configure(api_key=GOOGLE_API_KEY)
    return genai.GenerativeModel(MODEL_NAME)


def extract_json_from_response(text: str) -> dict:
    """Extract JSON from model response"""
    # Try to find JSON in the response
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Try parsing the whole response
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"texts": [], "error": "Failed to parse JSON"}


def get_ocr_with_bbox(model, image: Image.Image, use_vietnamese: bool = True) -> dict:
    """Get OCR results with bounding boxes from Gemini"""
    prompt = BBOX_PROMPT_VI if use_vietnamese else BBOX_PROMPT

    try:
        response = model.generate_content([prompt, image])
        result = extract_json_from_response(response.text)
        return result
    except Exception as e:
        return {"texts": [], "error": str(e)}


def get_font(size: int = 14):
    """Get a font for drawing text"""
    # Try to load a good font, fallback to default
    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]

    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except:
                continue

    # Return default font
    try:
        return ImageFont.truetype("Arial", size)
    except:
        return ImageFont.load_default()


def visualize_ocr(image_path: Path, ocr_result: dict, output_path: Path):
    """Draw OCR results on image"""
    # Load image
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size

    # Get font
    font_size = max(12, min(width, height) // 50)
    font = get_font(font_size)
    small_font = get_font(font_size - 2)

    texts = ocr_result.get("texts", [])

    # Colors for different text elements
    colors = [
        ("#FF6B6B", "#FF0000"),  # Red
        ("#4ECDC4", "#00AA99"),  # Teal
        ("#45B7D1", "#0088AA"),  # Blue
        ("#96CEB4", "#66AA88"),  # Green
        ("#FFEAA7", "#DDCC00"),  # Yellow
        ("#DDA0DD", "#AA66AA"),  # Purple
        ("#F39C12", "#DD8800"),  # Orange
    ]

    for i, item in enumerate(texts):
        text = item.get("text", "")
        bbox = item.get("bbox", [])

        if not text or len(bbox) != 4:
            continue

        # Convert percentage to pixel coordinates
        x1 = int(bbox[0] * width / 100)
        y1 = int(bbox[1] * height / 100)
        x2 = int(bbox[2] * width / 100)
        y2 = int(bbox[3] * height / 100)

        # Get color for this text
        fill_color, outline_color = colors[i % len(colors)]

        # Draw semi-transparent rectangle
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        # Parse hex color
        r = int(fill_color[1:3], 16)
        g = int(fill_color[3:5], 16)
        b = int(fill_color[5:7], 16)

        overlay_draw.rectangle([x1, y1, x2, y2], fill=(r, g, b, 60))
        img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(img)

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=outline_color, width=2)

        # Draw text label above the box
        text_short = text[:30] + "..." if len(text) > 30 else text

        # Calculate text background
        text_bbox = draw.textbbox((x1, y1 - font_size - 4), text_short, font=small_font)

        # Draw text background
        draw.rectangle(
            [text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2],
            fill=outline_color
        )

        # Draw text
        draw.text((x1, y1 - font_size - 4), text_short, fill="white", font=small_font)

    # Add legend at bottom
    legend_y = height - 30
    draw.rectangle([0, legend_y - 5, width, height], fill=(0, 0, 0, 180))
    draw.text(
        (10, legend_y),
        f"OCR Results: {len(texts)} text regions detected | Model: {MODEL_NAME}",
        fill="white",
        font=small_font
    )

    # Save result
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, quality=95)
    print(f"  ✓ Saved: {output_path}")

    return img


def create_side_by_side(original_path: Path, visualized_path: Path, output_path: Path):
    """Create side-by-side comparison"""
    original = Image.open(original_path)
    visualized = Image.open(visualized_path)

    # Resize to same height
    height = max(original.height, visualized.height)

    orig_ratio = original.width / original.height
    vis_ratio = visualized.width / visualized.height

    orig_resized = original.resize((int(height * orig_ratio), height))
    vis_resized = visualized.resize((int(height * vis_ratio), height))

    # Create combined image
    total_width = orig_resized.width + vis_resized.width + 20
    combined = Image.new('RGB', (total_width, height + 40), (30, 30, 30))

    # Paste images
    combined.paste(orig_resized, (0, 40))
    combined.paste(vis_resized, (orig_resized.width + 20, 40))

    # Add labels
    draw = ImageDraw.Draw(combined)
    font = get_font(20)

    draw.text((10, 10), "Original", fill="white", font=font)
    draw.text((orig_resized.width + 30, 10), "OCR Visualization", fill="white", font=font)

    combined.save(output_path, quality=95)
    print(f"  ✓ Side-by-side: {output_path}")


def process_all_images():
    """Process all images and create visualizations"""
    model = setup_gemini()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    image_files = [f for f in IMAGE_DIR.iterdir()
                   if f.suffix.lower() in image_extensions]

    print(f"\n{'='*60}")
    print("OCR Visualization with Bounding Boxes")
    print(f"{'='*60}")
    print(f"Processing {len(image_files)} images...")

    all_results = {}

    for image_path in sorted(image_files):
        print(f"\n📷 {image_path.name}")

        # Load image
        image = Image.open(image_path)

        # Get OCR with bounding boxes
        print("  Extracting text with bounding boxes...")
        ocr_result = get_ocr_with_bbox(model, image, use_vietnamese=True)

        if "error" in ocr_result and not ocr_result.get("texts"):
            print(f"  ⚠ Error: {ocr_result.get('error')}")
            continue

        texts = ocr_result.get("texts", [])
        print(f"  Found {len(texts)} text regions")

        # Save raw results
        all_results[image_path.name] = ocr_result

        # Create visualization
        output_path = OUTPUT_DIR / f"{image_path.stem}_ocr.png"
        visualize_ocr(image_path, ocr_result, output_path)

        # Create side-by-side
        side_by_side_path = OUTPUT_DIR / f"{image_path.stem}_comparison.png"
        create_side_by_side(image_path, output_path, side_by_side_path)

        # Print detected texts
        print("  Detected texts:")
        for item in texts[:5]:  # Show first 5
            text = item.get("text", "")[:50]
            print(f"    - {text}")
        if len(texts) > 5:
            print(f"    ... and {len(texts) - 5} more")

    # Save all results
    results_path = OUTPUT_DIR / "ocr_bbox_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Results saved to: {results_path}")

    print(f"\n{'='*60}")
    print(f"Visualization complete! Check: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    process_all_images()
