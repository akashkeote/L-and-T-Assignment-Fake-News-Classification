import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# PROJECT 15: Machine Learning–Based Fake News Detector Using Text Classification\n",
            "\n",
            "**Domain**: Natural Language Processing (NLP) & Supervised Machine Learning  \n",
            "**Objective**: Automatically filter misleading or fake news articles using textual features, TF-IDF vectorization, and multi-model classification (Logistic Regression, Passive Aggressive, LinearSVC, Naive Bayes).  \n",
            "**Primary Metric**: F1-Score & ROC-AUC  \n",
            "**Deployment**: Interactive Gradio Web Interface"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Import Essential Libraries & Setup"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 1,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import re\n",
            "import string\n",
            "import joblib\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from wordcloud import WordCloud\n",
            "\n",
            "import nltk\n",
            "from nltk.corpus import stopwords\n",
            "from nltk.stem import WordNetLemmatizer\n",
            "\n",
            "from sklearn.model_selection import train_test_split\n",
            "from sklearn.feature_extraction.text import TfidfVectorizer\n",
            "from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier\n",
            "from sklearn.naive_bayes import MultinomialNB\n",
            "from sklearn.svm import LinearSVC\n",
            "from sklearn.metrics import (\n",
            "    accuracy_score, precision_score, recall_score, f1_score,\n",
            "    classification_report, confusion_matrix, roc_auc_score, roc_curve\n",
            ")\n",
            "\n",
            "# Download NLTK data\n",
            "nltk.download('stopwords')\n",
            "nltk.download('wordnet')\n",
            "nltk.download('omw-1.4')\n",
            "\n",
            "print('Libraries successfully imported!')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Data Import & Exploratory Understanding"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 2,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load the Kaggle / News dataset\n",
            "df = pd.read_csv('news.csv')\n",
            "print(f'Total records: {df.shape[0]} | Columns: {df.columns.tolist()}')\n",
            "\n",
            "# Drop nulls and inspect target distribution\n",
            "df.dropna(subset=['title', 'text', 'label'], inplace=True)\n",
            "df['label_num'] = df['label'].map({'REAL': 0, 'FAKE': 1})\n",
            "\n",
            "display(df['label'].value_counts())\n",
            "display(df.head(3))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Metadata Extraction & Exploratory Data Analysis (EDA)\n",
            "Extracting writing style attributes: text length, word count, and capital letter ratios."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 3,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Combine headline and article body\n",
            "df['full_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')\n",
            "\n",
            "# Metadata\n",
            "df['char_length'] = df['full_text'].apply(len)\n",
            "df['word_count'] = df['full_text'].apply(lambda x: len(x.split()))\n",
            "df['caps_count'] = df['full_text'].apply(lambda x: sum(1 for c in x if c.isupper()))\n",
            "df['caps_ratio'] = df.apply(lambda r: r['caps_count'] / max(r['char_length'], 1), axis=1)\n",
            "\n",
            "# Summary statistics by class\n",
            "df.groupby('label')[['word_count', 'caps_ratio']].mean()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Visualizations (WordClouds, Density & Boxplots)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 4,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Length Comparison Plot\n",
            "plt.figure(figsize=(9, 4))\n",
            "sns.kdeplot(df[df['label'] == 'REAL']['word_count'], label='Real News', color='#10b981', fill=True, clip=(0, 2500))\n",
            "sns.kdeplot(df[df['label'] == 'FAKE']['word_count'], label='Fake News', color='#ef4444', fill=True, clip=(0, 2500))\n",
            "plt.title('Word Count Distribution: Real vs Fake News')\n",
            "plt.xlabel('Word Count')\n",
            "plt.legend()\n",
            "plt.show()\n",
            "\n",
            "# 2. Capital Ratio Comparison Plot\n",
            "plt.figure(figsize=(6, 4))\n",
            "sns.boxplot(x='label', y='caps_ratio', data=df, hue='label', palette={'REAL': '#10b981', 'FAKE': '#ef4444'}, legend=False)\n",
            "plt.title('Capital Letter Ratio Comparison')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Text Preprocessing & Lemmatization\n",
            "- Lowercasing\n",
            "- URL and HTML tag removal\n",
            "- Punctuation and numeric filtering\n",
            "- Stopword elimination\n",
            "- WordNet lemmatization"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 5,
        "metadata": {},
        "outputs": [],
        "source": [
            "stop_words = set(stopwords.words('english'))\n",
            "lemmatizer = WordNetLemmatizer()\n",
            "\n",
            "def clean_and_preprocess(text):\n",
            "    text = text.lower()\n",
            "    text = re.sub(r'https?://\\S+|www\\.\\S+', '', text)\n",
            "    text = re.sub(r'<.*?>', '', text)\n",
            "    text = re.sub(r'[^a-zA-Z\\s]', '', text)\n",
            "    tokens = text.split()\n",
            "    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]\n",
            "    return ' '.join(tokens)\n",
            "\n",
            "df['clean_text'] = df['full_text'].apply(clean_and_preprocess)\n",
            "print('Cleaned sample:', df['clean_text'].iloc[0][:150])"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Feature Engineering: TF-IDF Vectorization"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 6,
        "metadata": {},
        "outputs": [],
        "source": [
            "X_train_text, X_test_text, y_train, y_test = train_test_split(\n",
            "    df['clean_text'], df['label_num'], test_size=0.2, random_state=42, stratify=df['label_num']\n",
            ")\n",
            "\n",
            "# Unigrams + Bigrams, Top 8000 Features\n",
            "tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=3)\n",
            "X_train_tfidf = tfidf.fit_transform(X_train_text)\n",
            "X_test_tfidf = tfidf.transform(X_test_text)\n",
            "\n",
            "print(f'TF-IDF Train Matrix: {X_train_tfidf.shape}')\n",
            "print(f'TF-IDF Test Matrix:  {X_test_tfidf.shape}')\n",
            "\n",
            "# Persist vectorizer for deployment\n",
            "joblib.dump(tfidf, 'tfidf_vectorizer.joblib')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Model Building & Hyperparameter Tuning\n",
            "Comparing **Multinomial Naive Bayes**, **Logistic Regression**, **Passive Aggressive Classifier**, and **Linear Support Vector (LinearSVC)**."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 7,
        "metadata": {},
        "outputs": [],
        "source": [
            "models = {\n",
            "    'Multinomial Naive Bayes': MultinomialNB(alpha=0.1),\n",
            "    'Logistic Regression': LogisticRegression(C=5.0, max_iter=1000, random_state=42),\n",
            "    'Passive Aggressive Classifier': PassiveAggressiveClassifier(C=0.5, max_iter=1000, random_state=42),\n",
            "    'Linear Support Vector (LinearSVC)': LinearSVC(C=1.0, random_state=42, max_iter=2000)\n",
            "}\n",
            "\n",
            "results = []\n",
            "for name, model in models.items():\n",
            "    model.fit(X_train_tfidf, y_train)\n",
            "    y_pred = model.predict(X_test_tfidf)\n",
            "    \n",
            "    acc = accuracy_score(y_test, y_pred)\n",
            "    prec = precision_score(y_test, y_pred)\n",
            "    rec = recall_score(y_test, y_pred)\n",
            "    f1 = f1_score(y_test, y_pred)\n",
            "    \n",
            "    if hasattr(model, 'predict_proba'):\n",
            "        y_prob = model.predict_proba(X_test_tfidf)[:, 1]\n",
            "    elif hasattr(model, 'decision_function'):\n",
            "        y_prob = model.decision_function(X_test_tfidf)\n",
            "    else:\n",
            "        y_prob = y_pred\n",
            "    auc = roc_auc_score(y_test, y_prob)\n",
            "    \n",
            "    results.append({\n",
            "        'Model': name,\n",
            "        'Accuracy (%)': round(acc * 100, 2),\n",
            "        'Precision': round(prec, 4),\n",
            "        'Recall': round(rec, 4),\n",
            "        'F1-Score': round(f1, 4),\n",
            "        'ROC-AUC': round(auc, 4)\n",
            "    })\n",
            "\n",
            "results_df = pd.DataFrame(results).sort_values(by='F1-Score', ascending=False)\n",
            "display(results_df)\n",
            "\n",
            "# Save best model (LinearSVC)\n",
            "best_model = models['Linear Support Vector (LinearSVC)']\n",
            "joblib.dump(best_model, 'best_model.joblib')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 8. Detailed Evaluation (Confusion Matrix & Classification Report)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 8,
        "metadata": {},
        "outputs": [],
        "source": [
            "y_pred_best = best_model.predict(X_test_tfidf)\n",
            "print('Classification Report:\\n')\n",
            "print(classification_report(y_test, y_pred_best, target_names=['REAL', 'FAKE']))\n",
            "\n",
            "cm = confusion_matrix(y_test, y_pred_best)\n",
            "plt.figure(figsize=(5, 4))\n",
            "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])\n",
            "plt.title('Confusion Matrix — LinearSVC')\n",
            "plt.xlabel('Predicted')\n",
            "plt.ylabel('Actual')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 9. Gradio Interface Deployment\n",
            "Run the interactive web checker directly inside the notebook or as a standalone server."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 9,
        "metadata": {},
        "outputs": [],
        "source": [
            "import gradio as gr\n",
            "\n",
            "def predict_news(title, content):\n",
            "    text = f'{title} {content}'\n",
            "    cleaned = clean_and_preprocess(text)\n",
            "    vec = tfidf.transform([cleaned])\n",
            "    pred = best_model.predict(vec)[0]\n",
            "    score = best_model.decision_function(vec)[0]\n",
            "    prob_fake = 1 / (1 + np.exp(-score))\n",
            "    prob_real = 1 - prob_fake\n",
            "    \n",
            "    label = '🔴 FAKE NEWS' if pred == 1 else '🟢 REAL NEWS'\n",
            "    return label, {'REAL NEWS': float(prob_real), 'FAKE NEWS': float(prob_fake)}\n",
            "\n",
            "# Simple inference test\n",
            "sample_title = 'NASA confirms presence of sub-surface ice on Europa'\n",
            "sample_body = 'Scientists at NASA confirmed radar data indicating extensive water ice.'\n",
            "print('Test Prediction:', predict_news(sample_title, sample_body))"
        ]
    }
]

notebook_content = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.14.6"
        },
        "orig_nbformat": 4
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

nb_path = os.path.join(BASE_DIR, 'fake_news_detector.ipynb')
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(notebook_content, f, indent=2)

print(f"Jupyter Notebook generated at: {nb_path}")
