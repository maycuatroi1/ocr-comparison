# Gemini Flash OCR Results Summary

**Test Date:** February 3, 2026
**Model:** gemini-2.0-flash
**Platform:** MacBook Pro M4

## Test Images

| Image | Size | Description |
|-------|------|-------------|
| Document.png | 1239x929 | Whiteboard with Vietnamese handwriting, notes, diagrams |
| crazy-hand-writing.png | 1239x929 | Same whiteboard (different name) |
| wild.png | 697x929 | Stone monument with Vietnamese text |

## Performance Results

| Image | Detailed Prompt | Vietnamese Prompt |
|-------|-----------------|-------------------|
| Document.png | 3.73s (444 chars) | 3.94s (371 chars) |
| crazy-hand-writing.png | 3.82s (423 chars) | 4.31s (430 chars) |
| wild.png | 2.82s (80 chars) | 2.75s (79 chars) |

**Average Response Time:** ~3.4s

## OCR Quality Analysis

### 1. wild.png (Stone Monument)

**Ground Truth (approximate):**
```
CÂY LỘC VỪNG
DO ĐẠI TƯỚNG TÔ LÂM
UVTW ĐẢNG
THỨ TRƯỞNG BỘ CÔNG AN
TRỒNG NGÀY 15-2-2015
```

**Gemini Output (detailed):**
```
CÂY LỘC VÙNG
DO ĐỤC TÔ LÂM
UVTW DANG
THỨ TRƯỞNG BỘ CÔNG AN
TRÔNG NGAY 15-2-2015
```

**Errors:**
- "VỪNG" → "VÙNG" (diacritics error)
- "ĐẠI TƯỚNG" → "ĐỤC" (major error - missed words)
- "ĐẢNG" → "DANG" (missing diacritics)
- "TRỒNG" → "TRÔNG" (diacritics error)
- "NGÀY" → "NGAY" (diacritics error)

### 2. Document.png / crazy-hand-writing.png (Whiteboard)

The handwritten whiteboard contains messy Vietnamese notes about a meeting/project discussion. Key elements detected:
- "thiếu 3mm" - correctly identified
- Questions marked with "Q:"
- Names: "Cường", "Bình", "Hưng"
- Meeting notes: "Book MR & Hưng 15"
- Mixed Vietnamese/English content

**Challenges:**
- Handwriting quality varies
- Multiple orientations
- Overlapping text
- Unclear abbreviations

## Observations

1. **Speed:** Gemini Flash is fast (~3s per image) and works well on Mac M4 without GPU
2. **Vietnamese diacritics:** Struggles with some Vietnamese diacritics, especially on non-standard fonts/stone carvings
3. **Handwriting:** Reasonable performance on messy handwriting but makes errors
4. **Layout:** Attempts to preserve some layout structure

## Recommendations

1. For critical Vietnamese OCR, post-processing with spell-check recommended
2. Vietnamese prompt marginally improves diacritic handling
3. For comparison, DeepSeek-OCR-2 should be tested on same images via Colab
