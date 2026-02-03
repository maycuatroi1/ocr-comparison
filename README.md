# OCR Comparison: DeepSeek-OCR-2 vs Gemini Flash

This repository compares OCR accuracy between **DeepSeek-OCR-2** (Visual Causal Flow) and **Google Gemini Flash**.

## DeepSeek-OCR-2 Paper Summary

**Paper:** [arXiv:2601.20552](https://arxiv.org/abs/2601.20552)
**Title:** DeepSeek-OCR 2: Visual Causal Flow
**Authors:** Haoran Wei, Yaofeng Sun, Yukun Li
**Date:** January 2026

### Key Innovation: DeepEncoder V2

The paper introduces a novel approach to OCR that challenges conventional vision-language model design:

1. **Problem with existing systems:** Current VLMs process visual tokens in rigid raster-scan order (left-to-right, top-to-bottom), which doesn't match how humans actually perceive images.

2. **Solution - Visual Causal Flow:** A new encoder architecture (DeepEncoder V2) that dynamically reorganizes visual tokens based on image content semantics, mimicking human visual perception through "causally-informed sequential processing."

3. **Core Question:** Can 2D image understanding be achieved through cascaded 1D causal reasoning structures?

### Model Specifications

- **Visual tokens:** (0-6)×144 + 256 tokens via dynamic resolution
- **Resolution:** Multiple 768×768 patches + one 1024×1024 patch
- **Two modes:**
  - Document mode with layout detection (`<|grounding|>`)
  - Standard OCR without layout analysis
- **License:** Apache 2.0

## Project Structure

```
ocr-comparison/
├── test-images/           # Test images for comparison
├── scripts/
│   ├── gemini_ocr.py     # Gemini Flash OCR (runs on Mac M4)
│   ├── deepseek_ocr.py   # DeepSeek-OCR-2 (requires CUDA)
│   └── compare.py        # Compare results
├── notebooks/
│   └── deepseek_colab.ipynb  # Google Colab notebook
├── results/              # OCR results stored here
└── requirements.txt
```

## Quick Start

### Local (Mac M4) - Gemini Flash Only

```bash
pip install -r requirements.txt
export GOOGLE_API_KEY="your-api-key"
python scripts/gemini_ocr.py
```

### Google Colab - DeepSeek-OCR-2

1. Upload `notebooks/deepseek_colab.ipynb` to Google Colab
2. Enable GPU runtime (T4 or better)
3. Run all cells

### Full Comparison

```bash
# After running both OCR systems
python scripts/compare.py
```

## Requirements

- **Gemini Flash:** Python 3.10+, google-generativeai
- **DeepSeek-OCR-2:** CUDA 11.8+, PyTorch 2.6.0, vLLM 0.8.5, flash-attention 2.7.3

## Results

See `results/` folder for detailed comparison outputs.
