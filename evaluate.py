"""
Evaluate the pretrained classifier on a labeled image-folder dataset.

Expected structure:
dataset/
  test/
    angry/ image1.jpg ...
    disgust/ ...
    fear/ ...
    happy/ ...
    sad/ ...
    surprise/ ...
    neutral/ ...

Run:
  python evaluate.py --data_dir dataset/test --output_dir evaluation

The script reports metrics on the supplied test set. It does not invent or hard-code accuracy.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import pipeline

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Root directory with one subfolder per emotion class")
    parser.add_argument("--output_dir", default="evaluation", help="Where to save metrics and predictions")
    parser.add_argument("--model", default=os.getenv("EMOTION_MODEL_ID", "trpakov/vit-face-expression"))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"Dataset directory not found: {data_dir}")
    class_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())
    if not class_dirs:
        raise SystemExit("No class folders found. See README.md for the required directory layout.")

    samples = []
    for class_dir in class_dirs:
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in VALID_EXTS:
                samples.append((path, class_dir.name.lower()))
    if not samples:
        raise SystemExit("No supported image files found under the class folders.")

    print(f"Loading model: {args.model}")
    classifier = pipeline("image-classification", model=args.model, top_k=1)
    y_true, y_pred, rows = [], [], []
    for i, (path, label) in enumerate(samples, 1):
        try:
            image = Image.open(path).convert("RGB")
            result = classifier(image)
            if result and isinstance(result[0], list):
                result = result[0]
            pred = str(result[0]["label"]).strip().lower().replace(" ", "_")
            y_true.append(label)
            y_pred.append(pred)
            rows.append({"file": str(path), "true_label": label, "predicted_label": pred,
                         "model_score": float(result[0]["score"])})
        except Exception as exc:
            print(f"Skipping {path}: {exc}")
        if i % 100 == 0:
            print(f"Processed {i}/{len(samples)} images")

    if not y_true:
        raise SystemExit("No images were successfully evaluated.")
    labels = sorted(set(y_true) | set(y_pred))
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=labels, columns=labels).to_csv(out / "confusion_matrix.csv")
    with open(out / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    summary = {"model": args.model, "evaluated_images": len(y_true), "accuracy": accuracy,
               "macro_f1": report["macro avg"]["f1-score"],
               "weighted_f1": report["weighted avg"]["f1-score"],
               "labels": labels}
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nEvaluation summary")
    print(json.dumps(summary, indent=2))
    print("\nFull classification report")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    print(f"\nSaved results to: {out.resolve()}")

if __name__ == "__main__":
    main()
