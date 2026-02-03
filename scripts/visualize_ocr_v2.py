#!/usr/bin/env python3
"""
OCR Visualization V2 - Improved version
- Better text label positioning (avoid overlap)
- Numbered boxes with legend
- Cleaner visualization
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
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "visualized_v2"

# Improved prompt for better bbox accuracy
BBOX_PROMPT = """Analyze this image carefully and extract all visible text with precise bounding box coordinates.

Return JSON format:
{
  "texts": [
    {"text": "extracted text", "bbox": [x1, y1, x2, y2]}
  ]
}

Rules:
- bbox coordinates are percentages (0-100) of image width/height
- x1,y1 = top-left corner, x2,y2 = bottom-right corner
- Be PRECISE with coordinates - boxes should tightly fit the text
- For handwritten text, include the full word/phrase
- Group logically connected text together
- Include ALL visible text, even if faint

Return ONLY valid JSON."""


def setup_gemini():
    genai.configure(api_key=GOOGLE_API_KEY)
    return genai.GenerativeModel(MODEL_NAME)


def extract_json(text: str) -> dict:
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    return {"texts": []}


def get_ocr_with_bbox(model, image: Image.Image) -> dict:
    try:
        response = model.generate_content([BBOX_PROMPT, image])
        return extract_json(response.text)
    except Exception as e:
        return {"texts": [], "error": str(e)}


def get_font(size: int = 14):
    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except:
                continue
    return ImageFont.load_default()


def create_visualization(image_path: Path, ocr_result: dict, output_dir: Path):
    """Create multiple visualization outputs"""
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    texts = ocr_result.get("texts", [])

    # Define colors
    colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
        "#FFEAA7", "#DDA0DD", "#F39C12", "#3498DB",
        "#E74C3C", "#2ECC71", "#9B59B6", "#1ABC9C"
    ]

    # === VERSION 1: Boxes with numbers only (clean) ===
    img_numbered = img.copy()
    draw = ImageDraw.Draw(img_numbered)
    font = get_font(16)
    small_font = get_font(12)

    for i, item in enumerate(texts):
        bbox = item.get("bbox", [])
        if len(bbox) != 4:
            continue

        x1 = int(bbox[0] * width / 100)
        y1 = int(bbox[1] * height / 100)
        x2 = int(bbox[2] * width / 100)
        y2 = int(bbox[3] * height / 100)

        color = colors[i % len(colors)]

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Draw number circle
        circle_r = 10
        cx, cy = x1 - 5, y1 - 5
        draw.ellipse([cx - circle_r, cy - circle_r, cx + circle_r, cy + circle_r],
                     fill=color, outline="white", width=1)

        # Draw number
        num_text = str(i + 1)
        draw.text((cx - 5 if i < 9 else cx - 8, cy - 7), num_text, fill="white", font=small_font)

    # Save numbered version
    numbered_path = output_dir / f"{image_path.stem}_numbered.png"
    img_numbered.save(numbered_path, quality=95)

    # === VERSION 2: Create legend image ===
    legend_width = 400
    items_per_col = 20
    num_cols = (len(texts) + items_per_col - 1) // items_per_col
    legend_height = min(len(texts), items_per_col) * 25 + 60

    legend = Image.new('RGB', (legend_width * num_cols, legend_height), (40, 40, 40))
    legend_draw = ImageDraw.Draw(legend)

    title_font = get_font(18)
    legend_draw.text((10, 10), f"OCR Results: {len(texts)} regions", fill="white", font=title_font)

    for i, item in enumerate(texts):
        col = i // items_per_col
        row = i % items_per_col
        x = 10 + col * legend_width
        y = 45 + row * 25

        color = colors[i % len(colors)]
        text = item.get("text", "")[:35]
        if len(item.get("text", "")) > 35:
            text += "..."

        # Draw color box
        legend_draw.rectangle([x, y, x + 15, y + 15], fill=color)
        # Draw number and text
        legend_draw.text((x + 20, y), f"{i+1}. {text}", fill="white", font=small_font)

    legend_path = output_dir / f"{image_path.stem}_legend.png"
    legend.save(legend_path, quality=95)

    # === VERSION 3: Combined view (image + legend below) ===
    combined_height = height + legend_height + 20
    combined = Image.new('RGB', (max(width, legend_width * num_cols), combined_height), (40, 40, 40))
    combined.paste(img_numbered, (0, 0))
    combined.paste(legend, (0, height + 10))

    combined_path = output_dir / f"{image_path.stem}_combined.png"
    combined.save(combined_path, quality=95)

    # === VERSION 4: Inline labels (with smart positioning) ===
    img_inline = img.copy()
    draw = ImageDraw.Draw(img_inline)

    # Track used positions to avoid overlap
    used_positions = []

    for i, item in enumerate(texts):
        bbox = item.get("bbox", [])
        text = item.get("text", "")
        if len(bbox) != 4 or not text:
            continue

        x1 = int(bbox[0] * width / 100)
        y1 = int(bbox[1] * height / 100)
        x2 = int(bbox[2] * width / 100)
        y2 = int(bbox[3] * height / 100)

        color = colors[i % len(colors)]

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Smart label positioning
        label = text[:25] + "..." if len(text) > 25 else text
        text_bbox = draw.textbbox((0, 0), label, font=small_font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        # Try different positions: above, below, left, right
        positions = [
            (x1, y1 - text_h - 4),  # above
            (x1, y2 + 2),           # below
            (x2 + 2, y1),           # right
            (x1 - text_w - 2, y1),  # left
        ]

        # Find non-overlapping position
        label_x, label_y = positions[0]
        for px, py in positions:
            overlap = False
            for ux, uy, uw, uh in used_positions:
                if not (px + text_w < ux or px > ux + uw or
                        py + text_h < uy or py > uy + uh):
                    overlap = True
                    break
            if not overlap and 0 <= px < width - text_w and 0 <= py < height - text_h:
                label_x, label_y = px, py
                break

        used_positions.append((label_x, label_y, text_w, text_h))

        # Draw label background and text
        draw.rectangle([label_x - 1, label_y - 1, label_x + text_w + 1, label_y + text_h + 1],
                      fill=color)
        draw.text((label_x, label_y), label, fill="white", font=small_font)

    inline_path = output_dir / f"{image_path.stem}_inline.png"
    img_inline.save(inline_path, quality=95)

    print(f"  ✓ {image_path.stem}_numbered.png")
    print(f"  ✓ {image_path.stem}_legend.png")
    print(f"  ✓ {image_path.stem}_combined.png")
    print(f"  ✓ {image_path.stem}_inline.png")

    return {
        "numbered": numbered_path,
        "legend": legend_path,
        "combined": combined_path,
        "inline": inline_path
    }


def main():
    model = setup_gemini()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    image_files = [f for f in IMAGE_DIR.iterdir() if f.suffix.lower() in image_extensions]

    print(f"\n{'='*60}")
    print("OCR Visualization V2 - Improved")
    print(f"{'='*60}")

    all_results = {}

    for image_path in sorted(image_files):
        print(f"\n📷 {image_path.name}")

        image = Image.open(image_path)
        ocr_result = get_ocr_with_bbox(model, image)
        texts = ocr_result.get("texts", [])
        print(f"  Found {len(texts)} text regions")

        all_results[image_path.name] = ocr_result
        create_visualization(image_path, ocr_result, OUTPUT_DIR)

    # Save results
    results_path = OUTPUT_DIR / "ocr_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Done! Check: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
