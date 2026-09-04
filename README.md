# DermaAI: AI-Powered Facial Skin Analysis and Personalized Skincare Recommendation System

## Overview
DermaAI is a deep learning-based facial skin analysis system that uses Convolutional Neural Networks (CNNs) and computer vision techniques to analyze facial images and provide personalized skincare recommendations. The system identifies skin types, skin tones, and common skin concerns to help users make informed skincare decisions.

## Features
- Skin Type Classification (Oily, Dry, Combination, Sensitive, Normal)
- Skin Tone Detection
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
- Automated Skin Analysis Pipeline
- Personalized Skincare Recommendation Support

## Technologies Used
- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- OpenCV
- Computer Vision
- Deep Learning
- CNN (Convolutional Neural Networks)

## Project Structure

```text
DermaAI/
│
├── README.md
├── train.py
├── skin_analysis_cnn.py
├── download_indian_skin_disease_dataset.py
├── download_skincare_products_and_ingredients_dataset.py
└── requirements.txt
```

## Datasets
The project utilizes publicly available datasets from Kaggle and Hugging Face for:

- Skin Disease Classification
- Acne Detection
- Skin Type Classification
- Skincare Products and Ingredients Analysis

Dataset download scripts are included in the repository.

## Model Architecture
The system uses a custom CNN architecture consisting of:

- Convolution Layers
- Batch Normalization
- Max Pooling Layers
- Global Average Pooling
- Fully Connected Dense Layers
- Multi-Task Output Heads

The model predicts:
- Skin Type
- Skin Tone
- Skin Concerns

## How It Works
1. Facial image is provided as input.
2. Image is preprocessed and resized.
3. CNN extracts facial skin features.
4. Model predicts skin type, skin tone, and skin concerns.
5. Results are used to support skincare recommendations.

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

## Usage

Train the model:

```bash
python train.py
```

Run skin analysis using the trained CNN model:

```bash
python skin_analysis_cnn.py
```

## Future Enhancements
- Real-time webcam skin analysis
- Web-based user interface
- Advanced skincare product recommendation engine
- Mobile application integration
- Dermatology-focused AI assistant

## Author

**Nidhi**

Bachelor of Engineering (Computer Science and Engineering)
Specialization in IoT, Cybersecurity, and Blockchain

## License

This project is intended for educational and research purposes.
