# 🌍 Multilingual Fake News Detector

LSTM-based fake news detection supporting multiple African languages (Yoruba, French, Fon).

![Deep Learning](https://img.shields.io/badge/Deep_Learning-LSTM-red?logo=keras)
![NLP](https://img.shields.io/badge/NLP-Multilingual-blue)

---

## 📋 Overview

A BiLSTM neural network for detecting fake news across multiple languages:
- **English**
- **French** 
- **Yoruba**
- **Fon**

---

## 🚀 Quick Start

```bash
git clone https://github.com/dlifeofjay/News-Detector.git
cd News-Detector
pip install -r requirements.txt
python news_pred.py
```

---

## 📁 Project Structure

```
News-Detector/
├── News Detector.ipynb              # Training notebook
├── news_pred.py                     # Inference script
├── fake_news_bilstm_model.keras    # Trained BiLSTM model
├── tokenizer.pickle                 # Text tokenizer
├── yoruba_dataset.csv               # Yoruba news data
├── french_dataset.csv               # French news data
├── fon_dataset.csv                  # Fon language data
└── multilingual_news_dataset.csv    # Combined dataset
```

---

## 🧠 Architecture

- **Model**: Bidirectional LSTM
- **Embedding**: Learned embeddings
- **Languages**: 4 (EN, FR, YO, FON)

---

## 👨‍💻 Author

**Jubril Ifekoya** - Data Scientist & ML Engineer
