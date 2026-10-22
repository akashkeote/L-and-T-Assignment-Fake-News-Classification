# PROJECT 15 — Machine Learning–Based Fake News Detector Using Text Classification

> **Sensitivity:** LNT Construction Internal Use  
> **Live Production URL:** [https://fakenewsclassification.vercel.app](https://fakenewsclassification.vercel.app)  
> **PDF Project Report:** [Download Report](https://fakenewsclassification.vercel.app/download-report)

---

## 📌 Executive Summary
This project implements an end-to-end Machine Learning pipeline that filters misleading or fake news articles before publication. The system extracts textual features from article headlines and body content using TF-IDF n-grams (1-2 words), analyzes writing styles (capitalization ratios, word counts), and evaluates predictions using a tuned Linear Support Vector Classifier (LinearSVC).

The application is deployed live on **Vercel Serverless** with a responsive **White Theme UI** and real-time Explainable AI (XAI) feature attribution.

---

## 🚀 Live Demo & Endpoints
* **Web Application:** `https://fakenewsclassification.vercel.app`
* **Real-Time Prediction API:** `POST https://fakenewsclassification.vercel.app/api/predict`
* **Model Statistics API:** `GET https://fakenewsclassification.vercel.app/api/stats`
* **PDF Project Report:** `GET https://fakenewsclassification.vercel.app/download-report`

### Sample Prediction Request
```bash
curl -X POST https://fakenewsclassification.vercel.app/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Kerry to go to Paris in gesture of sympathy",
    "content": "U.S. Secretary of State John F. Kerry said Monday that he will stop in Paris later this week."
  }'
```

### Sample Response
```json
{
  "label": "REAL",
  "confidence": 80.08,
  "real_probability": 80.1,
  "fake_probability": 19.9,
  "decision_score": -1.391,
  "risk_level": "Low",
  "influential_real_terms": ["march", "sunday", "said", "monday", "kerry", "kerry said"],
  "influential_fake_terms": [],
  "metadata": {
    "char_length": 232,
    "word_count": 40,
    "caps_ratio": 5.6,
    "cleaned_tokens_count": 24
  }
}
```

---

## 📊 Model Benchmark & Results
Trained on **6,335 verified articles** (3,171 Real, 3,164 Fake) with an 80/20 stratified split:

| Model | Accuracy | Macro F1 | Precision | Recall | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **LinearSVC (Tuned)** | **93.69%** | **0.9367** | **93.81%** | **93.54%** | **0.983** | **Production Deployed** |
| **Logistic Regression** | 92.82% | 0.9279 | 93.12% | 0.9248 | 0.980 | Trained & Benchmarked |
| **Passive Aggressive** | 92.50% | 0.9249 | 91.68% | 0.9347 | 0.975 | Trained & Benchmarked |
| **Multinomial Naive Bayes** | 89.66% | 0.8962 | 90.80% | 0.8841 | 0.957 | Baseline Benchmark |

---

## 📁 Repository Structure
```
fake_news_classification/
├── api/
│   ├── index.py                    # Flask serverless application & White UI
│   ├── best_model.joblib           # Trained LinearSVC weights (64 KB)
│   ├── tfidf_vectorizer.joblib     # Fitted TF-IDF Vectorizer (306 KB)
│   ├── Fake_News_Classification_Project_Report.pdf
│   └── plots/                      # High-resolution EDA and evaluation charts
├── plots/                          # Visualizations (Confusion Matrix, ROC, etc.)
├── best_model.joblib               # Root model checkpoint
├── tfidf_vectorizer.joblib         # Root vectorizer checkpoint
├── fake_news_detector.ipynb        # Complete interactive Jupyter Notebook
├── train_and_evaluate.py           # Training and hyperparameter tuning script
├── gradio_app.py                   # Local Gradio GUI script
├── requirements.txt                # Vercel deployment dependencies
├── vercel.json                     # Vercel serverless routing configuration
├── Fake_News_Classification_Project_Report.pdf  # Comprehensive 5-page PDF report
└── README.md                       # Project documentation
```

---

## 🛠️ Local Setup & Execution
1. **Clone or extract project files**
2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run local web application:**
   ```bash
   python api/index.py
   ```
   Open `http://localhost:5000` in your web browser.
