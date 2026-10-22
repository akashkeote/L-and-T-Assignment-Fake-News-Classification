import os
import sys
import re
import string
import joblib

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from scipy.sparse import hstack

# Output directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

print("=" * 65)
print("  PROJECT 15: FAKE NEWS CLASSIFIER USING MACHINE LEARNING")
print("=" * 65)

# ── 1. LOAD DATASET ───────────────────────────────────────────
data_path = os.path.join(BASE_DIR, 'news.csv')
df = pd.read_csv(data_path)
print(f"[1/7] Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Drop any nulls
df.dropna(subset=['text', 'title', 'label'], inplace=True)
df['label_num'] = df['label'].map({'REAL': 0, 'FAKE': 1})
print(f"      Label distribution: REAL={sum(df['label_num']==0)}, FAKE={sum(df['label_num']==1)}")

# ── 2. FEATURE EXTRACTION (METADATA) ───────────────────────────
print("[2/7] Extracting Metadata Features (Length, Caps Ratio)...")
# Combine Title and Text
df['full_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

# Metadata before preprocessing (to capture casing habits)
df['char_length'] = df['full_text'].apply(len)
df['word_count'] = df['full_text'].apply(lambda x: len(x.split()))
df['caps_count'] = df['full_text'].apply(lambda x: sum(1 for c in x if c.isupper()))
df['caps_ratio'] = df.apply(lambda r: r['caps_count'] / max(r['char_length'], 1), axis=1)

# ── 3. TEXT PREPROCESSING ──────────────────────────────────────
print("[3/7] Preprocessing Text (Lowercasing, Cleaning, Stopwords, Lemmatization)...")
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_and_preprocess(text):
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove numbers and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize and remove stopwords + lemmatize
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

# Process in batches for fast execution
df['clean_text'] = df['full_text'].apply(clean_and_preprocess)
print("      Preprocessing complete! Sample cleaned tokens created.")

# ── 4. EXPLORATORY DATA ANALYSIS (EDA) & VISUALIZATIONS ────────
print("[4/7] Generating EDA Visualizations & Plots...")

# 4.1 Length Comparison Plot
plt.figure(figsize=(9, 4.5))
sns.kdeplot(df[df['label'] == 'REAL']['word_count'], label='Real News', color='#10b981', fill=True, clip=(0, 2500))
sns.kdeplot(df[df['label'] == 'FAKE']['word_count'], label='Fake News', color='#ef4444', fill=True, clip=(0, 2500))
plt.title('Word Count Distribution: Real vs Fake News', fontsize=12, fontweight='bold')
plt.xlabel('Word Count (Clipped at 2500)')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'length_distribution.png'), dpi=180)
plt.close()

# 4.2 Capital Ratio Comparison Plot
plt.figure(figsize=(7, 4))
sns.boxplot(x='label', y='caps_ratio', data=df, palette={'REAL': '#10b981', 'FAKE': '#ef4444'})
plt.title('Capital Letter Ratio Comparison', fontsize=12, fontweight='bold')
plt.xlabel('News Category')
plt.ylabel('Capital Letter Ratio')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'caps_ratio_comparison.png'), dpi=180)
plt.close()

# 4.3 WordClouds
fake_text = ' '.join(df[df['label'] == 'FAKE']['clean_text'].sample(min(800, len(df)), random_state=42))
real_text = ' '.join(df[df['label'] == 'REAL']['clean_text'].sample(min(800, len(df)), random_state=42))

wc_fake = WordCloud(width=800, height=450, background_color='#0f172a', colormap='Reds').generate(fake_text)
plt.figure(figsize=(10, 5))
plt.imshow(wc_fake, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud — Fake News Keywords', fontsize=14, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'wordcloud_fake.png'), dpi=180)
plt.close()

wc_real = WordCloud(width=800, height=450, background_color='#0f172a', colormap='Greens').generate(real_text)
plt.figure(figsize=(10, 5))
plt.imshow(wc_real, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud — Real News Keywords', fontsize=14, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'wordcloud_real.png'), dpi=180)
plt.close()

# 4.4 Top 15 Frequent Words Bar Chart
all_words = ' '.join(df['clean_text']).split()
word_freq = pd.Series(all_words).value_counts().head(15)

plt.figure(figsize=(9, 4.5))
sns.barplot(x=word_freq.values, y=word_freq.index, palette='viridis')
plt.title('Top 15 Most Frequent Words Across Corpus', fontsize=12, fontweight='bold')
plt.xlabel('Frequency')
plt.ylabel('Word')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'frequent_words.png'), dpi=180)
plt.close()

print("      Plots saved in ./plots/ directory.")

# ── 5. FEATURE ENGINEERING (TF-IDF + METADATA) ────────────────
print("[5/7] TF-IDF Vectorization & Feature Engineering...")
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df['clean_text'], df['label_num'], test_size=0.2, random_state=42, stratify=df['label_num']
)

# TF-IDF Vectorizer with n-grams (1, 2) and max 8000 features
tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=3)
X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)

# Save Vectorizer
joblib.dump(tfidf, os.path.join(BASE_DIR, 'tfidf_vectorizer.joblib'))
print("      Saved: tfidf_vectorizer.joblib")

# ── 6. MODEL TRAINING & COMPARISON ─────────────────────────────
print("[6/7] Training Models (Logistic Regression, Passive Aggressive, Naive Bayes)...")

models = {
    'Multinomial Naive Bayes': MultinomialNB(alpha=0.1),
    'Logistic Regression': LogisticRegression(C=5.0, max_iter=1000, random_state=42),
    'Passive Aggressive Classifier': PassiveAggressiveClassifier(C=0.5, max_iter=1000, random_state=42),
    'Linear Support Vector (LinearSVC)': LinearSVC(C=1.0, random_state=42, max_iter=2000)
}

results = {}
for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Calculate ROC-AUC if decision_function or predict_proba available
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test_tfidf)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_prob = model.decision_function(X_test_tfidf)
    else:
        y_prob = y_pred
    auc = roc_auc_score(y_test, y_prob)
    
    results[name] = {
        'model': model,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
        'y_pred': y_pred,
        'y_prob': y_prob
    }
    print(f"      {name:<33} | Acc: {acc*100:.2f}% | F1: {f1:.4f} | AUC: {auc:.4f}")

# Select Best Model based on F1-Score
best_model_name = max(results, key=lambda k: results[k]['f1'])
best_model = results[best_model_name]['model']
print(f"\n  >> Best Performing Model: {best_model_name} (F1 = {results[best_model_name]['f1']:.4f})")

# Save Best Model
joblib.dump(best_model, os.path.join(BASE_DIR, 'best_model.joblib'))
print("      Saved: best_model.joblib")

# ── 7. DETAILED EVALUATION & CHARTS ───────────────────────────
print("[7/7] Generating Confusion Matrix & ROC Curves...")

# 7.1 Confusion Matrix for Best Model
cm = confusion_matrix(y_test, results[best_model_name]['y_pred'])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Real (0)', 'Fake (1)'],
            yticklabels=['Real (0)', 'Fake (1)'])
plt.title(f'Confusion Matrix — {best_model_name}', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('Actual True Label')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'), dpi=180)
plt.close()

# 7.2 ROC Curves for All Models
plt.figure(figsize=(8, 5.5))
for name, data in results.items():
    fpr, tpr, _ = roc_curve(y_test, data['y_prob'])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {data['auc']:.3f})", lw=2)

plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Guessing (AUC = 0.50)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity / Recall)')
plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=12, fontweight='bold')
plt.legend(loc='lower right', fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'roc_curve.png'), dpi=180)
plt.close()

# Save performance comparison table as CSV
perf_df = pd.DataFrame([
    {
        'Model': name,
        'Accuracy (%)': round(d['accuracy'] * 100, 2),
        'Precision': round(d['precision'], 4),
        'Recall': round(d['recall'], 4),
        'F1-Score': round(d['f1'], 4),
        'ROC-AUC': round(d['auc'], 4)
    }
    for name, d in results.items()
]).sort_values(by='F1-Score', ascending=False)

perf_df.to_csv(os.path.join(BASE_DIR, 'model_comparison.csv'), index=False)
print("\n" + "=" * 65)
print("  MODEL COMPARISON SUMMARY")
print("=" * 65)
print(perf_df.to_string(index=False))
print("\n✅ All training, artifacts, and plots generated successfully!")
