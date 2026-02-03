#!/usr/bin/env python3
"""
Side-by-side OCR Visualization
Left: Original image | Right: OCR text results
"""

import os
import textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import google.generativeai as genai

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyDzgmUbGw_uwxjyjWh9y7x1EBmjAwaOiUo")
MODEL_NAME = "gemini-2.0-flash"
IMAGE_DIR = Path(__file__).parent.parent / "test-images"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "sidebyside"

OCR_PROMPT = """Trích xuất TẤT CẢ văn bản từ hình ảnh này.
- Bao gồm cả chữ viết tay
- Giữ nguyên bố cục theo dòng
- Mỗi vùng text trên một dòng mới
- Đánh dấu [?] nếu không rõ

Trả về văn bản đã trích xuất:"""


def setup_gemini():
    genai.configure(api_key=GOOGLE_API_KEY)
    return genai.GenerativeModel(MODEL_NAME)


def get_font(size: int = 14):
    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except:
                continue
    return ImageFont.load_default()


def perform_ocr(model, image: Image.Image) -> tuple[str, float]:
    import time
    start = time.time()
    try:
        response = model.generate_content([OCR_PROMPT, image])
        elapsed = time.time() - start
        return response.text, elapsed
    except Exception as e:
        elapsed = time.time() - start
        return f"Error: {str(e)}", elapsed


def create_sidebyside(image_path: Path, ocr_text: str, processing_time: float, output_path: Path):
    """Create side-by-side: original image | OCR text"""

    # Load and resize original image
    original = Image.open(image_path).convert("RGB")

    # Target height
    target_height = 800
    ratio = target_height / original.height
    new_width = int(original.width * ratio)
    original_resized = original.resize((new_width, target_height), Image.Resampling.LANCZOS)

    # Create text panel
    text_panel_width = 600
    text_panel = Image.new('RGB', (text_panel_width, target_height), (30, 30, 30))
    draw = ImageDraw.Draw(text_panel)

    # Fonts
    title_font = get_font(20)
    text_font = get_font(14)

    # Draw title
    draw.text((15, 15), "OCR Results (Gemini Flash)", fill="#4ECDC4", font=title_font)
    draw.line([(15, 45), (text_panel_width - 15, 45)], fill="#4ECDC4", width=2)

    # Wrap and draw OCR text
    y_offset = 60
    line_height = 20
    max_chars = 55  # characters per line

    lines = ocr_text.split('\n')
    line_num = 1

    for line in lines:
        if not line.strip():
            y_offset += line_height // 2
            continue

        # Wrap long lines
        wrapped = textwrap.wrap(line, width=max_chars) or ['']

        for i, wrapped_line in enumerate(wrapped):
            if y_offset > target_height - 40:
                draw.text((15, y_offset), "... [truncated]", fill="#888888", font=text_font)
                break

            # Draw line number for first wrapped segment
            if i == 0:
                draw.text((15, y_offset), f"{line_num:2d}.", fill="#888888", font=text_font)
                line_num += 1

            # Draw text
            draw.text((50, y_offset), wrapped_line, fill="white", font=text_font)
            y_offset += line_height

        if y_offset > target_height - 40:
            break

    # Add stats at bottom
    stats_y = target_height - 30
    draw.line([(15, stats_y - 10), (text_panel_width - 15, stats_y - 10)], fill="#444444", width=1)
    num_lines = len([l for l in lines if l.strip()])
    num_chars = len(ocr_text)
    draw.text((15, stats_y), f"⏱ {processing_time:.2f}s | Lines: {num_lines} | Chars: {num_chars} | {MODEL_NAME}",
              fill="#4ECDC4", font=get_font(12))

    # Combine images
    divider_width = 3
    total_width = new_width + divider_width + text_panel_width
    combined = Image.new('RGB', (total_width, target_height), (60, 60, 60))

    # Paste original
    combined.paste(original_resized, (0, 0))

    # Draw divider
    divider_draw = ImageDraw.Draw(combined)
    divider_draw.rectangle([new_width, 0, new_width + divider_width, target_height], fill="#4ECDC4")

    # Paste text panel
    combined.paste(text_panel, (new_width + divider_width, 0))

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(output_path, quality=95)

    return combined


def main():
    model = setup_gemini()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    image_files = [f for f in IMAGE_DIR.iterdir() if f.suffix.lower() in image_extensions]

    print(f"\n{'='*60}")
    print("Side-by-Side OCR Visualization")
    print(f"{'='*60}")

    for image_path in sorted(image_files):
        print(f"\n📷 {image_path.name}")

        # Get OCR
        image = Image.open(image_path)
        print("  Extracting text...")
        ocr_text, processing_time = perform_ocr(model, image)
        print(f"  ⏱ Processing time: {processing_time:.2f}s")

        # Preview
        preview = ocr_text[:100].replace('\n', ' ')
        print(f"  Preview: {preview}...")

        # Create visualization
        output_path = OUTPUT_DIR / f"{image_path.stem}_sidebyside.png"
        create_sidebyside(image_path, ocr_text, processing_time, output_path)
        print(f"  ✓ Saved: {output_path.name}")

        # Also save raw text
        txt_path = OUTPUT_DIR / f"{image_path.stem}_ocr.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(ocr_text)
        print(f"  ✓ Saved: {txt_path.name}")

    print(f"\n{'='*60}")
    print(f"Done! Check: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
