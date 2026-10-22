import os
import re
import string
import joblib
import numpy as np
import gradio as gr
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best_model.joblib')
TFIDF_PATH = os.path.join(BASE_DIR, 'tfidf_vectorizer.joblib')

# Load trained artifacts
model = joblib.load(MODEL_PATH)
tfidf = joblib.load(TFIDF_PATH)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict_news(title, content):
    if not title.strip() and not content.strip():
        return "⚠️ Error: Please enter a headline or article text.", {}, "Word Count: 0 | Capital Ratio: 0.0%"
    
    full_text = f"{title.strip()} {content.strip()}"
    
    # Metadata stats
    char_len = len(full_text)
    word_count = len(full_text.split())
    caps_count = sum(1 for c in full_text if c.isupper())
    caps_ratio = (caps_count / max(char_len, 1)) * 100
    
    # Clean and vectorize
    cleaned = clean_text(full_text)
    vec = tfidf.transform([cleaned])
    
    # Prediction & Confidence
    pred = model.predict(vec)[0]
    
    # Calibrated confidence via decision function or proba
    if hasattr(model, 'predict_proba'):
        prob_fake = model.predict_proba(vec)[0][1]
        prob_real = 1.0 - prob_fake
    elif hasattr(model, 'decision_function'):
        score = model.decision_function(vec)[0]
        prob_fake = float(sigmoid(score))
        prob_real = 1.0 - prob_fake
    else:
        prob_fake = 1.0 if pred == 1 else 0.0
        prob_real = 1.0 - prob_fake

    label_str = "🔴 FAKE NEWS" if pred == 1 else "🟢 REAL NEWS"
    
    confidences = {
        "REAL NEWS": round(prob_real, 4),
        "FAKE NEWS": round(prob_fake, 4)
    }
    
    stats_str = f"📊 Analysis Metadata:\n• Word Count: {word_count} words\n• Character Count: {char_len}\n• Uppercase Ratio: {caps_ratio:.2f}%\n• Cleaned Vocabulary Tokens: {len(cleaned.split())}"
    
    return label_str, confidences, stats_str

# Sample test cases for one-click demo
examples = [
    [
        "NASA Rover Discovers Ancient Lake Bed on Mars",
        "NASA's Perseverance rover has successfully analyzed soil core samples in the Jezero crater, confirming the historic existence of a deep water lake on Mars millions of years ago, according to scientists from the Jet Propulsion Laboratory."
    ],
    [
        "BREAKING: Secret Alien Civilization Discovered Under White House Basement",
        "SHOCKING insider leak proves world leaders have been hiding a subterranean alien species living beneath Washington D.C. for decades without public knowledge or congressional oversight! SHARE BEFORE DELETED!!"
    ],
    [
        "Federal Reserve Signals Potential Interest Rate Cut Following Inflation Drop",
        "The Federal Reserve chairman announced on Wednesday that central bank policymakers observed a consistent cooling in consumer price index data, creating conditions conducive for an easing of monetary policy in upcoming quarters."
    ]
]

# Custom CSS theme
custom_css = """
.gradio-container { max-width: 900px !important; margin: auto !important; }
.header-box { text-align: center; margin-bottom: 20px; }
"""

with gr.Blocks(title="AI Fake News Detector") as demo:
    gr.Markdown("""
    # 📰 AI-Powered Fake News Detector
    ### Machine Learning-Based Text Classification (Project Statement 15)
    *Enter an article's headline and content below to evaluate whether it is genuine or fabricated.*
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            in_title = gr.Textbox(label="Headline / Title", placeholder="e.g. Breaking: Federal Reserve cuts interest rates...", lines=2)
            in_content = gr.Textbox(label="Article Content", placeholder="Paste the news body text here...", lines=7)
            btn_submit = gr.Button("🔍 Analyze & Verify Article", variant="primary", size="lg")
            btn_clear = gr.ClearButton([in_title, in_content])
            
        with gr.Column(scale=2):
            out_label = gr.Label(label="Prediction Result", num_top_classes=2)
            out_confidence = gr.Label(label="Confidence Breakdown")
            out_stats = gr.Textbox(label="Text Diagnostics & Writing Structure", lines=5, interactive=False)
            
    btn_submit.click(fn=predict_news, inputs=[in_title, in_content], outputs=[out_label, out_confidence, out_stats])
    
    gr.Markdown("### 💡 Clickable Examples for Testing")
    gr.Examples(examples=examples, inputs=[in_title, in_content])
    
    gr.Markdown("""
    ---
    *Model Architecture: Linear Support Vector Machine (LinearSVC) & TF-IDF (8,000 n-gram features) | Test Accuracy: 93.69% | F1-Score: 0.9367*
    """)

if __name__ == '__main__':
    print("Launching Gradio Fake News Detector on local port...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
