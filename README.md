# DermaAI: AI-Powered Facial Skin Analysis and Personalized Skincare Recommendation System

## Overview

DermaAI is a deep learning-based facial skin analysis system that utilizes Convolutional Neural Networks (CNNs) and computer vision techniques to analyze facial images and provide personalized skincare recommendations. The system identifies skin types, skin tones, and common skin concerns, helping users make informed skincare decisions through AI-powered analysis.

---

## Features

- Skin Type Classification
  - Oily
  - Dry
  - Combination
  - Sensitive
  - Normal

- Skin Tone Detection
  - Light
  - Medium
  - Tan
  - Dark
  - Deep

- Skin Concern Analysis
  - Acne
  - Pigmentation
  - Dark Spots
  - Dark Circles
  - Open Pores
  - Blackheads
  - Whiteheads
  - Wrinkles
  - Fine Lines
  - Redness
  - Dullness

- CNN-Based Deep Learning Model
- Computer Vision-Based Facial Analysis
- Automated Skin Assessment
- Personalized Skincare Recommendation Support

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- OpenCV
- Computer Vision
- Deep Learning
- Convolutional Neural Networks (CNN)

---

## Project Structure

```text
DermaAI/
│
├── screenshots/
│   ├── home_screen.png
│   ├── form_page.png
│   ├── analysis_output.png
│   └── recommendation_results.png
│
├── README.md
├── train.py
├── skin_analysis_cnn.py
├── download_indian_skin_disease_dataset.py
├── download_skincare_products_and_ingredients_dataset.py
└── requirements.txt
```

---

## Datasets

This project utilizes publicly available datasets from Kaggle and Hugging Face for:

- Skin Disease Classification
- Acne Detection
- Skin Type Classification
- Skincare Products Dataset
- Skincare Ingredients Dataset

Dataset download scripts are included in this repository.

---

## Model Architecture

The CNN model consists of:

- Convolutional Layers
- Batch Normalization
- Max Pooling Layers
- Global Average Pooling
- Dense Layers
- Dropout Layers
- Multi-Task Output Heads

The model predicts:

- Skin Type
- Skin Tone
- Skin Concerns

---

## Workflow

1. User uploads a facial image.
2. Image is preprocessed and resized.
3. CNN extracts facial skin features.
4. Model predicts skin type, skin tone, and skin concerns.
5. Results are analyzed.
6. Personalized skincare recommendations are generated.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/DermaAI.git
cd DermaAI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Train the model:

```bash
python train.py
```

Run skin analysis:

```bash
python skin_analysis_cnn.py
```

---

## Project Screenshots

### Home Screen

![Home Screen](screenshots/home_screen.png)

### User Input Form

![Form Page](screenshots/form_page.png)

### Skin Analysis Output

![Analysis Output](screenshots/analysis_output.png)

### Recommendation Results

![Recommendation Results](screenshots/recommendation_results.png)

---

## Future Enhancements

- Real-Time Webcam Skin Analysis
- Web-Based Interactive Dashboard
- Advanced Product Recommendation Engine
- Mobile Application Support
- Dermatology AI Assistant
- Cloud Deployment

---

## Author

**Nidhi**

Bachelor of Engineering (Computer Science and Engineering)

Specialization in IoT, Cybersecurity, and Blockchain

---

## License

This project is developed for educational, research, and learning purposes.
