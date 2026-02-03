#!/usr/bin/env python3
"""
Compare OCR results between DeepSeek-OCR-2 and Gemini Flash
"""

import json
from pathlib import Path
from difflib import SequenceMatcher
import pandas as pd

RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_results(filename: str) -> dict:
    """Load results from JSON file"""
    filepath = RESULTS_DIR / filename
    if not filepath.exists():
        print(f"Warning: {filename} not found")
        return {}

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate text similarity using SequenceMatcher"""
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    if not text:
        return ""
    # Remove extra whitespace
    return " ".join(text.split())


def compare_results():
    """Compare OCR results between models"""

    # Load all results
    gemini_detailed = load_results("gemini_detailed.json")
    gemini_vietnamese = load_results("gemini_vietnamese.json")
    deepseek_detailed = load_results("deepseek_detailed.json")
    deepseek_vietnamese = load_results("deepseek_vietnamese.json")

    # Collect all image names
    all_images = set()
    for results in [gemini_detailed, gemini_vietnamese, deepseek_detailed, deepseek_vietnamese]:
        all_images.update(results.keys())

    print("=" * 80)
    print("OCR COMPARISON: DeepSeek-OCR-2 vs Gemini Flash")
    print("=" * 80)

    comparison_data = []

    for image_name in sorted(all_images):
        print(f"\n{'='*80}")
        print(f"IMAGE: {image_name}")
        print("=" * 80)

        # Get results for this image
        gd = gemini_detailed.get(image_name, {})
        gv = gemini_vietnamese.get(image_name, {})
        dd = deepseek_detailed.get(image_name, {})
        dv = deepseek_vietnamese.get(image_name, {})

        # Print timing comparison
        print("\n⏱️  TIMING:")
        if gd.get("success"):
            print(f"  Gemini (detailed):    {gd.get('elapsed_time', 0):.2f}s")
        if gv.get("success"):
            print(f"  Gemini (vietnamese):  {gv.get('elapsed_time', 0):.2f}s")
        if dd.get("success"):
            print(f"  DeepSeek (detailed):  {dd.get('elapsed_time', 0):.2f}s")
        if dv.get("success"):
            print(f"  DeepSeek (vietnamese):{dv.get('elapsed_time', 0):.2f}s")

        # Print text length comparison
        print("\n📏 TEXT LENGTH:")
        if gd.get("success"):
            print(f"  Gemini (detailed):    {len(gd.get('text', ''))} chars")
        if gv.get("success"):
            print(f"  Gemini (vietnamese):  {len(gv.get('text', ''))} chars")
        if dd.get("success"):
            print(f"  DeepSeek (detailed):  {len(dd.get('text', ''))} chars")
        if dv.get("success"):
            print(f"  DeepSeek (vietnamese):{len(dv.get('text', ''))} chars")

        # Calculate similarity between models
        if gd.get("success") and dd.get("success"):
            similarity = calculate_similarity(gd.get("text", ""), dd.get("text", ""))
            print(f"\n🔄 SIMILARITY (Gemini vs DeepSeek detailed): {similarity:.2%}")

        # Print extracted text
        print("\n" + "-" * 40)
        print("GEMINI FLASH OUTPUT (detailed):")
        print("-" * 40)
        if gd.get("success"):
            print(gd.get("text", "")[:1000])
            if len(gd.get("text", "")) > 1000:
                print("... [truncated]")
        else:
            print(f"Error: {gd.get('error', 'No result')}")

        print("\n" + "-" * 40)
        print("DEEPSEEK-OCR-2 OUTPUT (detailed):")
        print("-" * 40)
        if dd.get("success"):
            print(dd.get("text", "")[:1000])
            if len(dd.get("text", "")) > 1000:
                print("... [truncated]")
        else:
            print(f"Error: {dd.get('error', 'No result')}")

        # Collect data for summary table
        comparison_data.append({
            "image": image_name,
            "gemini_time": gd.get("elapsed_time", None) if gd.get("success") else None,
            "deepseek_time": dd.get("elapsed_time", None) if dd.get("success") else None,
            "gemini_chars": len(gd.get("text", "")) if gd.get("success") else 0,
            "deepseek_chars": len(dd.get("text", "")) if dd.get("success") else 0,
            "similarity": calculate_similarity(gd.get("text", ""), dd.get("text", "")) if (gd.get("success") and dd.get("success")) else None
        })

    # Create summary table
    if comparison_data:
        print("\n" + "=" * 80)
        print("SUMMARY TABLE")
        print("=" * 80)

        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))

        # Save summary
        summary_path = RESULTS_DIR / "comparison_summary.csv"
        df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to: {summary_path}")

        # Calculate averages
        print("\n" + "-" * 40)
        print("AVERAGES:")
        if df["gemini_time"].notna().any():
            print(f"  Gemini avg time:   {df['gemini_time'].mean():.2f}s")
        if df["deepseek_time"].notna().any():
            print(f"  DeepSeek avg time: {df['deepseek_time'].mean():.2f}s")
        if df["similarity"].notna().any():
            print(f"  Avg similarity:    {df['similarity'].mean():.2%}")


def main():
    compare_results()


if __name__ == "__main__":
    main()
