# Facial Emotion Recognition using Deep Learning

A GitHub-ready Streamlit application that estimates facial-expression categories from uploaded images using a pretrained Vision Transformer (ViT) image-classification model.

> **Responsible-use note:** Facial expressions are not reliable proof of a person's internal emotional state. This is an educational computer-vision demo, not a medical, hiring, policing, surveillance, or psychological assessment tool. Predictions can be wrong and may vary across lighting, pose, image quality, and demographic groups.

## Features

- Streamlit interface for JPG, JPEG, PNG, and WEBP uploads.
- OpenCV face detection; falls back to the full image if no face is detected.
- Pretrained Hugging Face image-classification model.
- Top predicted label, model score, and score chart.
- Evaluation script that calculates accuracy, precision, recall, F1-score, confusion matrix, and per-class results on a labeled test dataset.
- Reproducible setup and GitHub structure.

## Model and emotion labels

Default model: [`trpakov/vit-face-expression`](https://huggingface.co/trpakov/vit-face-expression). The app downloads model files from Hugging Face on first use, so the first launch requires an internet connection. The model is pretrained: this repository does **not** train a new network during app use.

The model's supported labels are expected to correspond to common expression categories such as angry, disgust, fear, happy, sad, surprise, and neutral. Check the model card for the current label mapping and limitations.

## Project structure

```text
facial-emotion-recognition/
├── app.py
├── evaluate.py
├── requirements.txt
├── README.md
├── .gitignore
├── LICENSE
└── dataset/
    └── README.md
```

## Setup (Windows)

Use Python 3.10 or 3.11 for broad compatibility.

1. Download or clone this repository and open its folder in Terminal / PowerShell.
2. Create a virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal and activate again.

3. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Start the app:

   ```powershell
   streamlit run app.py
   ```

5. The browser will open the local app. Upload an image. On first run, the pretrained model is downloaded and cached.

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Dataset

The app can run inference without a local training dataset. For formal evaluation, obtain a properly licensed labeled facial-expression dataset. One commonly used research dataset is **FER-2013**; its access and terms may depend on the hosting platform. Do not commit a large dataset, private face images, or data you do not have permission to redistribute.

For the evaluation script, arrange a labeled test split as image folders, with one folder per class:

```text
dataset/test/
├── angry/
├── disgust/
├── fear/
├── happy/
├── neutral/
├── sad/
└── surprise/
```

Put the corresponding images inside each folder. The folder names are treated as the ground-truth labels. Ensure the labels match the model's label names; use the exact mapping provided by the dataset and model card. A dataset with a different class set requires an explicit label mapping before interpreting the results.

See [`dataset/README.md`](dataset/README.md) for more details.

## Evaluation metrics

Run evaluation against a held-out test set:

```bash
python evaluate.py --data_dir dataset/test --output_dir evaluation
```

The script creates:
- `evaluation/summary.json` — image count, accuracy, macro-F1, weighted-F1.
- `evaluation/classification_report.json` — precision, recall, F1-score, and support by class.
- `evaluation/confusion_matrix.csv` — true-vs-predicted class counts.
- `evaluation/predictions.csv` — individual predictions and model scores.

**No accuracy is claimed in advance.** Metrics depend on the exact test dataset, its label mapping, preprocessing, and model version. Report the actual output from your own held-out evaluation rather than inventing a score. Do not use training images as the test set.

## How it works

1. The user uploads an image.
2. OpenCV attempts to locate faces using a Haar cascade.
3. Each detected face crop is passed to a pretrained Vision Transformer classifier.
4. The UI displays the highest-scoring class and score distribution.
5. `evaluate.py` compares model predictions with ground-truth folder labels and computes standard classification metrics.

## Common issues

- **First run is slow:** model weights are downloaded once and cached.
- **Model download fails:** check internet access, Hugging Face availability, and firewall/proxy settings.
- **PyTorch installation problem:** install a PyTorch build suitable for your Python and operating system from the official PyTorch installation selector, then rerun `pip install -r requirements.txt`.
- **No face detected:** try a well-lit, front-facing image. The app falls back to the full image, so the result may be less reliable.
- **Evaluation labels mismatch:** verify the exact class names and label mapping in both the dataset and model card.

## Suggested GitHub description

**A Streamlit facial-expression classification demo powered by a pretrained Vision Transformer, with OpenCV face detection and a reproducible evaluation script.**

## Limitations

- A model score is not a calibrated probability that a person truly feels a particular emotion.
- Face detection and classification can fail on occluded, profile, low-resolution, or poorly lit images.
- Dataset bias and label ambiguity affect performance.
- Do not use this project to make consequential decisions about people.
