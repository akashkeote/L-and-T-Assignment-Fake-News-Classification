import os
import sys
import re
import io
import joblib
import numpy as np
import pypdf
from flask import Flask, request, jsonify, render_template_string, send_from_directory, send_file

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

app = Flask(__name__)

# Directory references
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))

def find_file(filename):
    for p in [os.path.join(BASE_DIR, filename), os.path.join(PARENT_DIR, filename)]:
        if os.path.exists(p):
            return p
    return os.path.join(BASE_DIR, filename)

MODEL_PATH = find_file('best_model.joblib')
TFIDF_PATH = find_file('tfidf_vectorizer.joblib')
REPORT_PATH = find_file('Fake_News_Classification_Project_Report.pdf')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
if not os.path.exists(PLOTS_DIR):
    PLOTS_DIR = os.path.join(PARENT_DIR, 'plots')

# Load model and vectorizer
model = None
tfidf = None
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(TFIDF_PATH):
        model = joblib.load(MODEL_PATH)
        tfidf = joblib.load(TFIDF_PATH)
except Exception as e:
    print(f"Error loading artifacts: {e}", file=sys.stderr)

# Exact NLTK English stopwords set (zero runtime network dependency)
STOPWORDS = set([
    'a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren',
    "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by',
    'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don',
    "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn',
    "hasn't", 'have', 'haven', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', 'hers', 'herself',
    'him', 'himself', 'his', 'how', 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', 'isn', "isn't", 'it',
    "it'd", "it'll", "it's", 'its', 'itself', 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most',
    'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once',
    'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't",
    'she', "she'd", "she'll", "she's", 'should', "should've", 'shouldn', "shouldn't", 'so', 'some', 'such', 't',
    'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they',
    "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've',
    'very', 'was', 'wasn', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', 'weren', "weren't", 'what',
    'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't",
    'y', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'
])

def simple_lemmatize(word):
    if word.endswith('ies') and len(word) > 4: return word[:-3] + 'y'
    if word.endswith('es') and len(word) > 4: return word[:-2]
    if word.endswith('s') and not word.endswith('ss') and len(word) > 3: return word[:-1]
    return word

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = [simple_lemmatize(w) for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return ' '.join(tokens)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -25, 25)))

# ══════════════════════════════════════════════════
#  API ROUTES
# ══════════════════════════════════════════════════

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(silent=True) or {}
    title = data.get('title', '').strip()
    content = data.get('content', '').strip()
    
    if not title and not content:
        return jsonify({'error': 'Please enter a headline or article body text.'}), 400
    
    full_text = f"{title} {content}".strip()
    
    # Metadata features
    char_len = len(full_text)
    word_count = len(full_text.split())
    caps_count = sum(1 for c in full_text if c.isupper())
    caps_ratio = round((caps_count / max(char_len, 1)) * 100, 2)
    
    # Preprocessing
    cleaned = clean_text(full_text)
    clean_tokens = cleaned.split()
    
    if model is None or tfidf is None:
        return jsonify({'error': 'Model artifacts not loaded.'}), 500
    
    vec = tfidf.transform([cleaned])
    pred = int(model.predict(vec)[0])
    
    if hasattr(model, 'decision_function'):
        score = float(model.decision_function(vec)[0])
        prob_fake = float(sigmoid(score))
    elif hasattr(model, 'predict_proba'):
        prob_fake = float(model.predict_proba(vec)[0][1])
        score = float(prob_fake - 0.5) * 4
    else:
        prob_fake = 1.0 if pred == 1 else 0.0
        score = 1.0 if pred == 1 else -1.0
        
    prob_real = float(1.0 - prob_fake)
    
    # Top influential words from this article
    top_fake_words = []
    top_real_words = []
    try:
        feature_names = tfidf.get_feature_names_out()
        coefs = model.coef_[0]
        coo = vec.tocoo()
        word_impacts = []
        for col_idx, tfidf_val in zip(coo.col, coo.data):
            impact = tfidf_val * coefs[col_idx]
            word_impacts.append((feature_names[col_idx], impact))
        
        # Sort by impact
        word_impacts.sort(key=lambda x: x[1])
        top_real_words = [w for w, imp in word_impacts[:6] if imp < -0.05]
        top_fake_words = [w for w, imp in reversed(word_impacts) if imp > 0.05][:6]
    except Exception:
        pass

    confidence = round(prob_fake * 100 if pred == 1 else prob_real * 100, 2)
    
    return jsonify({
        'label': 'FAKE' if pred == 1 else 'REAL',
        'confidence': confidence,
        'fake_probability': round(prob_fake * 100, 1),
        'real_probability': round(prob_real * 100, 1),
        'decision_score': round(score, 3),
        'metadata': {
            'char_length': char_len,
            'word_count': word_count,
            'caps_count': caps_count,
            'caps_ratio': caps_ratio,
            'cleaned_tokens_count': len(clean_tokens)
        },
        'influential_fake_terms': top_fake_words,
        'influential_real_terms': top_real_words,
        'risk_level': 'High' if pred == 1 else 'Low'
    })

# ══════════════════════════════════════════════════
#  PDF DOCUMENT FACT-CHECKER (RAG PIPELINE)
# ══════════════════════════════════════════════════

@app.route('/api/analyze-pdf', methods=['GET', 'POST'])
def api_analyze_pdf():
    file_bytes = None
    filename = 'document.pdf'

    # Check query preset sample
    sample_type = request.args.get('sample')
    if sample_type == 'real':
        sample_path = find_file('sample_real_news.pdf')
        if os.path.exists(sample_path):
            with open(sample_path, 'rb') as f:
                file_bytes = f.read()
            filename = 'sample_diplomatic_briefing.pdf'
    elif sample_type == 'fake':
        sample_path = find_file('sample_fake_news.pdf')
        if os.path.exists(sample_path):
            with open(sample_path, 'rb') as f:
                file_bytes = f.read()
            filename = 'sample_alien_conspiracy.pdf'

    # Check file upload in POST
    if file_bytes is None:
        if 'file' in request.files:
            uploaded = request.files['file']
            filename = uploaded.filename or 'uploaded_document.pdf'
            file_bytes = uploaded.read()
        else:
            json_data = request.get_json(silent=True) or {}
            if 'text' in json_data:
                # Text payload fallback
                return analyze_text_claims(json_data.get('text', ''), json_data.get('filename', 'article.txt'))

    if not file_bytes:
        return jsonify({'error': 'No PDF file uploaded or selected.'}), 400

    try:
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        num_pages = len(reader.pages)
        claims = []
        total_words = 0

        feature_names = tfidf.get_feature_names_out()
        coefs = model.coef_[0]

        for p_idx, page in enumerate(reader.pages, 1):
            raw_text = page.extract_text() or ''
            lines = [l.strip() for l in raw_text.split('\n') if len(l.strip()) > 0]
            
            # Semantic RAG Chunking
            curr_chunk = []
            for line in lines:
                curr_chunk.append(line)
                combined = ' '.join(curr_chunk).strip()
                if len(combined.split()) >= 16 or line.endswith(('.', '!', '?')):
                    if len(combined.split()) >= 8 and any(c.islower() for c in combined):
                        cleaned = clean_text(combined)
                        if cleaned:
                            total_words += len(combined.split())
                            vec = tfidf.transform([cleaned])
                            score = float(model.decision_function(vec)[0])
                            prob_fake = float(sigmoid(score))
                            pred = 1 if score > 0 else 0
                            conf = float(prob_fake if pred == 1 else (1.0 - prob_fake))

                            # Influential terms
                            coo = vec.tocoo()
                            impacts = []
                            for col, val in zip(coo.col, coo.data):
                                impacts.append((feature_names[col], val * coefs[col]))
                            impacts.sort(key=lambda x: abs(x[1]), reverse=True)
                            top_terms = [w for w, _ in impacts[:3]]

                            claims.append({
                                'id': len(claims) + 1,
                                'page': p_idx,
                                'snippet': combined if len(combined) <= 220 else combined[:217] + '...',
                                'verdict': 'FAKE' if pred == 1 else 'REAL',
                                'confidence': round(conf * 100, 1),
                                'score': round(score, 3),
                                'risk': 'High' if pred == 1 else 'Low',
                                'terms': top_terms
                            })
                    curr_chunk = []

        fake_count = sum(1 for c in claims if c['verdict'] == 'FAKE')
        real_count = sum(1 for c in claims if c['verdict'] == 'REAL')
        total_claims = len(claims)

        if total_claims == 0:
            doc_verdict = 'REAL'
            overall_conf = 85.0
            risk_level = 'Low'
            summary_txt = 'Document contains minimal text assertions or tabular data.'
        else:
            fake_ratio = fake_count / total_claims
            doc_verdict = 'FAKE' if fake_count > real_count else 'REAL'
            overall_conf = round((fake_ratio if doc_verdict == 'FAKE' else (1.0 - fake_ratio)) * 100, 1)
            risk_level = 'High' if doc_verdict == 'FAKE' else 'Low'
            summary_txt = (
                f"Document audited with RAG semantic chunking across {num_pages} page(s). "
                f"Out of {total_claims} verified claims, {real_count} exhibit credible journalistic framing "
                f"and {fake_count} were flagged as misleading or sensational."
            )

        return jsonify({
            'filename': filename,
            'pages_count': num_pages,
            'total_words': total_words,
            'total_claims': total_claims,
            'real_claims_count': real_count,
            'fake_claims_count': fake_count,
            'overall_verdict': doc_verdict,
            'confidence': overall_conf,
            'risk_level': risk_level,
            'summary': summary_txt,
            'claims': claims
        })

    except Exception as e:
        return jsonify({'error': f"Failed to parse PDF: {str(e)}"}), 500

def analyze_text_claims(text, filename):
    # Fallback text chunker
    lines = [p.strip() for p in text.split('\n') if len(p.strip()) > 8]
    claims = []
    feature_names = tfidf.get_feature_names_out()
    coefs = model.coef_[0]
    for idx, line in enumerate(lines, 1):
        cleaned = clean_text(line)
        if cleaned:
            vec = tfidf.transform([cleaned])
            score = float(model.decision_function(vec)[0])
            prob_fake = float(sigmoid(score))
            pred = 1 if score > 0 else 0
            conf = float(prob_fake if pred == 1 else (1.0 - prob_fake))
            claims.append({
                'id': idx,
                'page': 1,
                'snippet': line[:200],
                'verdict': 'FAKE' if pred == 1 else 'REAL',
                'confidence': round(conf * 100, 1),
                'score': round(score, 3),
                'risk': 'High' if pred == 1 else 'Low',
                'terms': []
            })
    fake_cnt = sum(1 for c in claims if c['verdict'] == 'FAKE')
    real_cnt = sum(1 for c in claims if c['verdict'] == 'REAL')
    verdict = 'FAKE' if fake_cnt > real_cnt else 'REAL'
    return jsonify({
        'filename': filename,
        'pages_count': 1,
        'total_words': len(text.split()),
        'total_claims': len(claims),
        'real_claims_count': real_cnt,
        'fake_claims_count': fake_cnt,
        'overall_verdict': verdict,
        'confidence': 85.0,
        'risk_level': 'High' if verdict == 'FAKE' else 'Low',
        'summary': f"Processed {len(claims)} claim chunks from text document.",
        'claims': claims
    })

@app.route('/api/stats')
def api_stats():
    return jsonify({
        'dataset_size': 6335,
        'real_count': 3171,
        'fake_count': 3164,
        'best_model': 'Linear Support Vector Classifier (LinearSVC)',
        'accuracy': 93.69,
        'f1_score': 0.9367,
        'precision': 93.81,
        'recall': 93.54,
        'vocabulary_size': 8000
    })

@app.route('/plots/<filename>')
def serve_plot(filename):
    if os.path.exists(os.path.join(PLOTS_DIR, filename)):
        return send_from_directory(PLOTS_DIR, filename)
    return "Plot not found", 404

@app.route('/download-report')
def download_report():
    if os.path.exists(REPORT_PATH):
        return send_file(REPORT_PATH, as_attachment=False, mimetype='application/pdf')
    return "Report not found", 404

# ══════════════════════════════════════════════════
#  MODERN WHITE UI TEMPLATE (WITH RAG TAB)
# ══════════════════════════════════════════════════

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FakeGuard AI — Real-Time Fake News & PDF Document Detector</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #f8fafc;
    --card: #ffffff;
    --border: #e2e8f0;
    --border-light: #f1f5f9;
    --text: #0f172a;
    --text-muted: #64748b;
    --text-sub: #475569;
    --indigo: #4f46e5;
    --indigo-hover: #4338ca;
    --indigo-light: #eef2ff;
    --indigo-border: #c7d2fe;
    --emerald: #059669;
    --emerald-light: #ecfdf5;
    --emerald-border: #a7f3d0;
    --rose: #e11d48;
    --rose-light: #fff1f2;
    --rose-border: #fecdd3;
    --amber: #d97706;
    --amber-light: #fffbeb;
    --amber-border: #fde68a;
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    --shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px -1px rgba(0, 0, 0, 0.1);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -4px rgba(0, 0, 0, 0.05);
    --radius-sm: 8px;
    --radius: 12px;
    --radius-lg: 16px;
    --radius-xl: 20px;
  }
  body {
    font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
    background-color: var(--bg);
    color: var(--text);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    -webkit-font-smoothing: antialiased;
  }
  
  /* NAVBAR */
  nav {
    position: sticky;
    top: 0;
    z-index: 100;
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border);
    padding: 0 28px;
    height: 68px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: var(--shadow-sm);
  }
  .brand-group {
    display: flex;
    align-items: center;
    gap: 14px;
  }
  .brand-icon {
    width: 42px;
    height: 42px;
    border-radius: var(--radius);
    background: linear-gradient(135deg, #4f46e5 0%, #2563eb 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
  }
  .brand-title {
    font-size: 17px;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .brand-badge {
    background: var(--indigo-light);
    color: var(--indigo);
    font-size: 11px;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 6px;
    border: 1px solid var(--indigo-border);
  }
  .brand-sub {
    font-size: 12px;
    color: var(--text-muted);
    font-weight: 500;
  }
  .nav-right {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .status-pill {
    display: flex;
    align-items: center;
    gap: 8px;
    background: var(--emerald-light);
    border: 1px solid var(--emerald-border);
    color: var(--emerald);
    padding: 6px 12px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 600;
  }
  .pulse-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--emerald);
    position: relative;
  }
  .pulse-dot::after {
    content: '';
    position: absolute;
    top: -2px; left: -2px; right: -2px; bottom: -2px;
    border-radius: 50%;
    background: var(--emerald);
    opacity: 0.6;
    animation: pulse 2s infinite ease-out;
  }
  @keyframes pulse {
    0% { transform: scale(1); opacity: 0.8; }
    100% { transform: scale(2.4); opacity: 0; }
  }
  .btn-outline {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--card);
    border: 1px solid var(--border);
    color: var(--text-sub);
    padding: 7px 14px;
    border-radius: var(--radius-sm);
    font-size: 13px;
    font-weight: 600;
    text-decoration: none;
    transition: all 0.15s ease;
    cursor: pointer;
  }
  .btn-outline:hover {
    background: var(--bg);
    border-color: var(--indigo);
    color: var(--indigo);
  }

  /* LAYOUT */
  main {
    flex: 1;
    max-width: 1280px;
    width: 100%;
    margin: 0 auto;
    padding: 28px 24px;
    display: flex;
    flex-direction: column;
    gap: 24px;
  }

  /* MODE SELECTOR TABS */
  .mode-tabs {
    display: flex;
    gap: 8px;
    background: var(--card);
    padding: 6px;
    border-radius: var(--radius);
    border: 1px solid var(--border);
    box-shadow: var(--shadow-sm);
    max-width: 620px;
  }
  .tab-btn {
    flex: 1;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 10px 16px;
    border: none;
    border-radius: var(--radius-sm);
    font-family: inherit;
    font-size: 13px;
    font-weight: 700;
    cursor: pointer;
    background: transparent;
    color: var(--text-muted);
    transition: all 0.15s ease;
  }
  .tab-btn.active {
    background: var(--indigo);
    color: white;
    box-shadow: 0 2px 6px rgba(79, 70, 229, 0.3);
  }

  /* STATS BANNER */
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
  }
  .stat-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px 20px;
    box-shadow: var(--shadow-sm);
    display: flex;
    flex-direction: column;
    gap: 6px;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
  }
  .stat-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
  }
  .stat-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .stat-label {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-muted);
  }
  .stat-badge {
    font-size: 11px;
    font-weight: 600;
    padding: 2px 7px;
    border-radius: 9999px;
  }
  .stat-value {
    font-size: 26px;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--text);
  }
  .stat-sub {
    font-size: 12px;
    color: var(--text-muted);
  }

  /* WORKSPACE SPLIT */
  .workspace-grid {
    display: grid;
    grid-template-columns: 1.15fr 0.85fr;
    gap: 24px;
    align-items: start;
  }
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow);
    overflow: hidden;
  }
  .card-header {
    padding: 20px 24px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--card);
  }
  .card-title-group {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .card-title {
    font-size: 16px;
    font-weight: 700;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .card-subtitle {
    font-size: 12px;
    color: var(--text-muted);
  }
  .card-body {
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }

  /* INPUTS */
  .form-group {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .form-label {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-sub);
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .input-text {
    width: 100%;
    padding: 12px 14px;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    font-family: inherit;
    font-size: 14px;
    color: var(--text);
    background: var(--card);
    outline: none;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
  }
  .input-text:focus, .textarea-text:focus {
    border-color: var(--indigo);
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.12);
  }
  .textarea-text {
    width: 100%;
    height: 180px;
    padding: 14px;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    font-family: inherit;
    font-size: 14px;
    line-height: 1.6;
    color: var(--text);
    background: var(--card);
    outline: none;
    resize: vertical;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
  }

  /* LIVE STATS BAR */
  .text-metrics-bar {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 10px 14px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    font-size: 12px;
    color: var(--text-sub);
  }
  .metric-item {
    display: flex;
    align-items: center;
    gap: 5px;
    font-weight: 500;
  }
  .metric-val {
    font-weight: 700;
    color: var(--text);
    font-family: 'JetBrains Mono', monospace;
  }
  .caps-badge {
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 700;
  }
  .caps-normal { background: var(--emerald-light); color: var(--emerald); }
  .caps-high   { background: var(--amber-light); color: var(--amber); }
  .caps-alert  { background: var(--rose-light); color: var(--rose); }

  /* PRESET BUTTONS */
  .presets-section {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .presets-label {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-muted);
  }
  .presets-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
  }
  .preset-btn {
    text-align: left;
    padding: 10px 12px;
    border: 1px solid var(--border);
    background: var(--card);
    border-radius: var(--radius-sm);
    cursor: pointer;
    font-family: inherit;
    transition: all 0.15s ease;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .preset-btn:hover {
    border-color: var(--indigo);
    background: var(--indigo-light);
  }
  .preset-title {
    font-size: 12px;
    font-weight: 700;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .preset-desc {
    font-size: 11px;
    color: var(--text-muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  /* ACTIONS */
  .action-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding-top: 4px;
  }
  .btn-primary {
    flex: 1;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    background: var(--indigo);
    color: white;
    border: none;
    padding: 13px 20px;
    border-radius: var(--radius-sm);
    font-size: 14px;
    font-weight: 700;
    cursor: pointer;
    box-shadow: 0 2px 8px rgba(79, 70, 229, 0.35);
    transition: all 0.15s ease;
  }
  .btn-primary:hover {
    background: var(--indigo-hover);
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.45);
  }
  .btn-primary:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
  .btn-secondary {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 13px 18px;
    border: 1px solid var(--border);
    background: var(--card);
    color: var(--text-sub);
    border-radius: var(--radius-sm);
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
  }
  .btn-secondary:hover {
    background: var(--bg);
  }

  /* RIGHT PANEL: VERDICT */
  .verdict-placeholder {
    padding: 60px 20px;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    color: var(--text-muted);
  }
  .verdict-placeholder-icon {
    width: 60px;
    height: 60px;
    border-radius: 50%;
    background: var(--border-light);
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-muted);
  }
  .verdict-box {
    display: none;
    flex-direction: column;
    gap: 20px;
    animation: fadeIn 0.3s ease forwards;
  }
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
  }

  /* VERDICT BANNER */
  .verdict-banner {
    padding: 20px;
    border-radius: var(--radius);
    display: flex;
    align-items: flex-start;
    gap: 16px;
  }
  .banner-fake {
    background: var(--rose-light);
    border: 1px solid var(--rose-border);
  }
  .banner-real {
    background: var(--emerald-light);
    border: 1px solid var(--emerald-border);
  }
  .verdict-icon-bubble {
    width: 48px;
    height: 48px;
    border-radius: var(--radius);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }
  .icon-fake { background: #fee2e2; color: var(--rose); }
  .icon-real { background: #d1fae5; color: var(--emerald); }
  .verdict-details {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .verdict-header-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .verdict-title {
    font-size: 19px;
    font-weight: 800;
    letter-spacing: -0.02em;
  }
  .title-fake { color: #9f1239; }
  .title-real { color: #065f46; }
  .badge-risk-fake {
    background: #fecdd3;
    color: #9f1239;
    font-size: 11px;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 9999px;
  }
  .badge-risk-real {
    background: #a7f3d0;
    color: #065f46;
    font-size: 11px;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 9999px;
  }
  .verdict-desc {
    font-size: 13px;
    line-height: 1.5;
    color: var(--text-sub);
  }

  /* CONFIDENCE BARS */
  .confidence-section {
    display: flex;
    flex-direction: column;
    gap: 10px;
    background: var(--bg);
    border: 1px solid var(--border);
    padding: 16px;
    border-radius: var(--radius);
  }
  .conf-bar-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 13px;
    font-weight: 600;
  }
  .progress-track {
    height: 10px;
    background: #e2e8f0;
    border-radius: 9999px;
    overflow: hidden;
    display: flex;
  }
  .progress-fill-fake {
    background: linear-gradient(90deg, #f43f5e, #e11d48);
    transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
  }
  .progress-fill-real {
    background: linear-gradient(90deg, #10b981, #059669);
    transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
  }
  .dual-conf-labels {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    font-weight: 600;
    color: var(--text-sub);
  }

  /* INFLUENTIAL KEYWORDS */
  .keywords-card {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .section-mini-title {
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-muted);
  }
  .pill-cloud {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }
  .word-pill {
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
  }
  .pill-fake { background: #ffe4e6; color: #be123c; border: 1px solid #fecdd3; }
  .pill-real { background: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }

  /* METRICS MINI-GRID */
  .metrics-mini-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
  }
  .metric-mini-box {
    background: var(--bg);
    border: 1px solid var(--border);
    padding: 10px 12px;
    border-radius: var(--radius-sm);
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .metric-mini-lbl { font-size: 11px; color: var(--text-muted); font-weight: 600; }
  .metric-mini-val { font-size: 15px; font-weight: 700; color: var(--text); font-family: 'JetBrains Mono', monospace; }

  /* ══════════════════════════════════════════════ */
  /* PDF & RAG DROPZONE STYLES */
  /* ══════════════════════════════════════════════ */
  .dropzone {
    border: 2px dashed #cbd5e1;
    border-radius: var(--radius);
    padding: 36px 20px;
    text-align: center;
    background: var(--bg);
    cursor: pointer;
    transition: all 0.15s ease;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
  }
  .dropzone:hover, .dropzone.dragover {
    border-color: var(--indigo);
    background: var(--indigo-light);
  }
  .dropzone-icon {
    width: 52px;
    height: 52px;
    border-radius: 50%;
    background: #e0e7ff;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--indigo);
  }
  .pdf-claims-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }
  .pdf-claims-table th {
    background: var(--bg);
    padding: 10px 12px;
    font-weight: 700;
    color: var(--text-sub);
    border-bottom: 1px solid var(--border);
    text-align: left;
  }
  .pdf-claims-table td {
    padding: 10px 12px;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
  }
  .badge-tag-fake {
    background: #fee2e2;
    color: #991b1b;
    font-weight: 700;
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 4px;
    display: inline-block;
  }
  .badge-tag-real {
    background: #d1fae5;
    color: #065f46;
    font-weight: 700;
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 4px;
    display: inline-block;
  }

  /* MODEL BENCHMARKS SECTION */
  .benchmark-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow);
    overflow: hidden;
  }
  .table-responsive {
    overflow-x: auto;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    text-align: left;
    font-size: 13px;
  }
  th {
    background: var(--bg);
    padding: 14px 18px;
    font-weight: 700;
    color: var(--text-sub);
    border-bottom: 1px solid var(--border);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  td {
    padding: 14px 18px;
    border-bottom: 1px solid var(--border);
    color: var(--text);
  }
  tr:last-child td { border-bottom: none; }
  tr.highlight-row {
    background: #f5f3ff;
    font-weight: 600;
  }
  .model-badge-winner {
    background: #4f46e5;
    color: white;
    font-size: 10px;
    font-weight: 700;
    padding: 2px 6px;
    border-radius: 4px;
    margin-left: 6px;
  }

  /* VISUALIZATION GALLERY */
  .gallery-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    padding: 20px 24px;
  }
  .plot-item {
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    background: var(--card);
    box-shadow: var(--shadow-sm);
    cursor: pointer;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
  }
  .plot-item:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
  }
  .plot-img {
    width: 100%;
    height: 160px;
    object-fit: cover;
    background: #f1f5f9;
    display: block;
  }
  .plot-caption {
    padding: 10px 12px;
    font-size: 12px;
    font-weight: 600;
    color: var(--text-sub);
    border-top: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  /* MODAL */
  .modal-backdrop {
    display: none;
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(15, 23, 42, 0.7);
    backdrop-filter: blur(4px);
    z-index: 999;
    align-items: center;
    justify-content: center;
    padding: 20px;
  }
  .modal-content {
    background: var(--card);
    border-radius: var(--radius-lg);
    max-width: 850px;
    width: 100%;
    max-height: 90vh;
    overflow-y: auto;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
    display: flex;
    flex-direction: column;
  }
  .modal-header {
    padding: 16px 20px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .modal-close {
    background: none;
    border: none;
    font-size: 20px;
    cursor: pointer;
    color: var(--text-muted);
  }
  .modal-body {
    padding: 20px;
    display: flex;
    justify-content: center;
  }
  .modal-img {
    max-width: 100%;
    max-height: 70vh;
    border-radius: var(--radius);
    object-fit: contain;
  }

  /* FOOTER */
  footer {
    background: var(--card);
    border-top: 1px solid var(--border);
    padding: 24px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 40px;
    font-size: 12px;
    color: var(--text-muted);
  }

  /* RESPONSIVE */
  @media (max-width: 900px) {
    .workspace-grid { grid-template-columns: 1fr; }
    .stats-grid { grid-template-columns: 1fr 1fr; }
    .gallery-grid { grid-template-columns: 1fr; }
    .presets-grid { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>

  <!-- NAVBAR -->
  <nav>
    <div class="brand-group">
      <div class="brand-icon">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
          <path d="m9 12 2 2 4-4"/>
        </svg>
      </div>
      <div>
        <div class="brand-title">
          FakeGuard AI
          <span class="brand-badge">LinearSVC 93.7%</span>
        </div>
        <div class="brand-sub">Real-Time News & Document Fact-Checker · Sensitivity: LNT Construction Internal Use</div>
      </div>
    </div>
    <div class="nav-right">
      <div class="status-pill">
        <span class="pulse-dot"></span>
        Real-Time Active
      </div>
      <a href="/download-report" target="_blank" class="btn-outline">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
          <polyline points="14 2 14 8 20 8"/>
          <line x1="12" y1="18" x2="12" y2="12"/>
          <line x1="9" y1="15" x2="15" y2="15"/>
        </svg>
        Download PDF Report
      </a>
    </div>
  </nav>

  <main>

    <!-- MODE SWITCHER -->
    <div class="mode-tabs">
      <button class="tab-btn active" id="tabTextBtn" onclick="switchMode('text')">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1-2.5-2.5Z"/>
          <path d="M6 6h10"/>
          <path d="M6 10h10"/>
        </svg>
        Real-Time Article Checker
      </button>
      <button class="tab-btn" id="tabPdfBtn" onclick="switchMode('pdf')">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
          <polyline points="14 2 14 8 20 8"/>
          <line x1="16" y1="13" x2="8" y2="13"/>
          <line x1="16" y1="17" x2="8" y2="17"/>
        </svg>
        PDF Document Analyzer (RAG Pipeline)
      </button>
    </div>

    <!-- STATS ROW -->
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-header">
          <span class="stat-label">Model Accuracy</span>
          <span class="stat-badge" style="background:#e0e7ff; color:#4338ca;">Top Model</span>
        </div>
        <div class="stat-value">93.69%</div>
        <div class="stat-sub">LinearSVC on 1,267 holdout test set</div>
      </div>
      <div class="stat-card">
        <div class="stat-header">
          <span class="stat-label">Macro F1-Score</span>
          <span class="stat-badge" style="background:#ecfdf5; color:#047857;">Balanced</span>
        </div>
        <div class="stat-value">0.9367</div>
        <div class="stat-sub">Precision: 93.8% · Recall: 93.5%</div>
      </div>
      <div class="stat-card">
        <div class="stat-header">
          <span class="stat-label">Corpus Size</span>
          <span class="stat-badge" style="background:#f1f5f9; color:#475569;">Verified</span>
        </div>
        <div class="stat-value">6,335</div>
        <div class="stat-sub">3,171 Real & 3,164 Fake Articles</div>
      </div>
      <div class="stat-card">
        <div class="stat-header">
          <span class="stat-label">Real-Time Latency</span>
          <span class="stat-badge" style="background:#fffbeb; color:#b45309;">Serverless</span>
        </div>
        <div class="stat-value">&lt; 15 ms</div>
        <div class="stat-sub">Sub-second inference pipeline</div>
      </div>
    </div>

    <!-- MODE 1: ARTICLE TEXT CHECKER -->
    <div id="modeTextSection" class="workspace-grid">

      <!-- LEFT: ARTICLE INPUT -->
      <div class="card">
        <div class="card-header">
          <div class="card-title-group">
            <div class="card-title">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color:var(--indigo);">
                <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1-2.5-2.5Z"/>
                <path d="M6 6h10"/>
                <path d="M6 10h10"/>
                <path d="M6 14h6"/>
              </svg>
              News Article Inspector
            </div>
            <div class="card-subtitle">Paste headline and text for automated classification</div>
          </div>
        </div>

        <div class="card-body">
          <div class="form-group">
            <label class="form-label">
              <span>Headline / Title <span style="font-weight:400; color:var(--text-muted);">(Optional)</span></span>
            </label>
            <input type="text" id="newsTitle" class="input-text" placeholder="e.g. Kerry to go to Paris in gesture of sympathy" oninput="updateLiveMetrics()">
          </div>

          <div class="form-group">
            <label class="form-label">
              <span>Article Body Content</span>
              <span id="charCountNotice" style="font-size:11px; color:var(--text-muted);">0 characters</span>
            </label>
            <textarea id="newsContent" class="textarea-text" placeholder="Paste the news story, press release, or editorial text here..." oninput="updateLiveMetrics()"></textarea>
          </div>

          <!-- LIVE METRICS BAR -->
          <div class="text-metrics-bar">
            <div class="metric-item">
              <span>Words:</span>
              <span class="metric-val" id="metricWords">0</span>
            </div>
            <div class="metric-item">
              <span>Chars:</span>
              <span class="metric-val" id="metricChars">0</span>
            </div>
            <div class="metric-item">
              <span>Caps Ratio:</span>
              <span class="caps-badge caps-normal" id="metricCaps">0.0%</span>
            </div>
          </div>

          <!-- 1-CLICK PRESETS -->
          <div class="presets-section">
            <div class="presets-label">⚡ One-Click Test Scenarios</div>
            <div class="presets-grid">
              <button class="preset-btn" onclick="loadSample(1)">
                <div class="preset-title">
                  <span style="color:#059669;">●</span> Real: Kerry Paris Diplomacy
                </div>
                <div class="preset-desc">U.S. Secretary of State official diplomacy visit</div>
              </button>
              <button class="preset-btn" onclick="loadSample(2)">
                <div class="preset-title">
                  <span style="color:#e11d48;">●</span> Fake: Hillary Fear & FBI War
                </div>
                <div class="preset-desc">Sensational blog speculation and conspiratorial tone</div>
              </button>
              <button class="preset-btn" onclick="loadSample(3)">
                <div class="preset-title">
                  <span style="color:#059669;">●</span> Real: Federal Reserve Interest Rates
                </div>
                <div class="preset-desc">Financial policy reporting on inflation indices</div>
              </button>
              <button class="preset-btn" onclick="loadSample(4)">
                <div class="preset-title">
                  <span style="color:#e11d48;">●</span> Fake: Secret Alien Bunker
                </div>
                <div class="preset-desc">Clickbait alien conspiracy with extreme capitalization</div>
              </button>
            </div>
          </div>

          <!-- ACTION BUTTONS -->
          <div class="action-row">
            <button class="btn-primary" id="btnAnalyze" onclick="analyzeArticle()">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="11" cy="11" r="8"/>
                <line x1="21" y1="21" x2="16.65" y2="16.65"/>
              </svg>
              <span>Analyze Article with ML</span>
            </button>
            <button class="btn-secondary" onclick="clearForm()">
              Clear
            </button>
          </div>
        </div>
      </div>

      <!-- RIGHT: VERDICT & METRICS -->
      <div class="card">
        <div class="card-header">
          <div class="card-title-group">
            <div class="card-title">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color:var(--emerald);">
                <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
                <polyline points="22 4 12 14.01 9 11.01"/>
              </svg>
              Classification Verdict
            </div>
            <div class="card-subtitle">Probability, linguistic signals & model confidence</div>
          </div>
        </div>

        <div class="card-body">

          <!-- INITIAL PLACEHOLDER -->
          <div class="verdict-placeholder" id="verdictPlaceholder">
            <div class="verdict-placeholder-icon">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"/>
                <line x1="12" y1="16" x2="12" y2="12"/>
                <line x1="12" y1="8" x2="12.01" y2="8"/>
              </svg>
            </div>
            <div style="font-size:15px; font-weight:700; color:var(--text);">No Article Analyzed Yet</div>
            <div style="font-size:13px; max-width:320px; line-height:1.5;">
              Type or paste an article on the left, or pick one of the preset scenarios to evaluate with the trained LinearSVC model.
            </div>
          </div>

          <!-- VERDICT RESULT BOX -->
          <div class="verdict-box" id="verdictBox">

            <!-- BANNER -->
            <div class="verdict-banner" id="verdictBanner">
              <div class="verdict-icon-bubble" id="verdictBubble"></div>
              <div class="verdict-details">
                <div class="verdict-header-row">
                  <div class="verdict-title" id="verdictTitle">REAL NEWS</div>
                  <span class="badge-risk-real" id="verdictRiskBadge">CREDIBLE</span>
                </div>
                <div class="verdict-desc" id="verdictDesc">
                  Text patterns and attribution tokens align with verified journalistic standards.
                </div>
              </div>
            </div>

            <!-- PROBABILITY BARS -->
            <div class="confidence-section">
              <div class="conf-bar-header">
                <span>Model Confidence</span>
                <span id="verdictConfidenceText" style="font-weight:800; font-family:'JetBrains Mono', monospace;">94.2%</span>
              </div>
              <div class="progress-track">
                <div class="progress-fill-fake" id="barFake" style="width: 15%;"></div>
                <div class="progress-fill-real" id="barReal" style="width: 85%;"></div>
              </div>
              <div class="dual-conf-labels">
                <span style="color:#e11d48;">Fake Prob: <span id="valProbFake">15.0%</span></span>
                <span style="color:#059669;">Real Prob: <span id="valProbReal">85.0%</span></span>
              </div>
            </div>

            <!-- INFLUENTIAL KEYWORDS -->
            <div class="keywords-card">
              <div class="section-mini-title">Key Contributing Tokens (TF-IDF Influencers)</div>
              <div class="pill-cloud" id="pillCloud"></div>
            </div>

            <!-- METRICS MINI-GRID -->
            <div class="metrics-mini-grid">
              <div class="metric-mini-box">
                <span class="metric-mini-lbl">Decision Margin</span>
                <span class="metric-mini-val" id="metricDecisionScore">-1.42</span>
              </div>
              <div class="metric-mini-box">
                <span class="metric-mini-lbl">Caps Letter Ratio</span>
                <span class="metric-mini-val" id="metricCapsResult">2.1%</span>
              </div>
              <div class="metric-mini-box">
                <span class="metric-mini-lbl">Cleaned Tokens</span>
                <span class="metric-mini-val" id="metricTokenCount">142</span>
              </div>
            </div>

          </div>

        </div>
      </div>

    </div>

    <!-- MODE 2: PDF DOCUMENT FACT-CHECKER (RAG PIPELINE) -->
    <div id="modePdfSection" class="workspace-grid" style="display:none;">

      <!-- LEFT: PDF UPLOAD & PARSER -->
      <div class="card">
        <div class="card-header">
          <div class="card-title-group">
            <div class="card-title">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color:var(--indigo);">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                <polyline points="14 2 14 8 20 8"/>
              </svg>
              PDF Document Ingestion & RAG
            </div>
            <div class="card-subtitle">Extracts semantic claim chunks and audits factuality page-by-page</div>
          </div>
        </div>

        <div class="card-body">
          <input type="file" id="pdfFileInput" accept="application/pdf" style="display:none;" onchange="handleFileUpload(this.files)">
          
          <div class="dropzone" id="pdfDropzone" onclick="document.getElementById('pdfFileInput').click()">
            <div class="dropzone-icon">
              <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                <polyline points="17 8 12 3 7 8"/>
                <line x1="12" y1="3" x2="12" y2="15"/>
              </svg>
            </div>
            <div>
              <strong style="font-size:14px; color:var(--text);">Drop news report or document PDF here</strong>
              <div style="font-size:12px; color:var(--text-muted); margin-top:3px;">or click to browse local files</div>
            </div>
            <div style="font-size:11px; color:var(--indigo); font-weight:600; background:var(--indigo-light); padding:3px 8px; border-radius:6px;">
              Supports multi-page PDF documents
            </div>
          </div>

          <!-- 1-CLICK SAMPLE PDF AUDITS -->
          <div class="presets-section">
            <div class="presets-label">⚡ Test Pre-Packaged Documents</div>
            <div class="presets-grid">
              <button class="preset-btn" onclick="testPdfSample('real')">
                <div class="preset-title">
                  <span style="color:#059669;">●</span> Official Diplomatic Briefing (PDF)
                </div>
                <div class="preset-desc">Credible State Department international diplomacy report</div>
              </button>
              <button class="preset-btn" onclick="testPdfSample('fake')">
                <div class="preset-title">
                  <span style="color:#e11d48;">●</span> Secret Alien Underground Bunker (PDF)
                </div>
                <div class="preset-desc">Sensational clickbait flyer alleging alien base</div>
              </button>
            </div>
          </div>

          <div id="pdfLoadingNotice" style="display:none; padding:14px; background:var(--indigo-light); border-radius:var(--radius-sm); font-size:13px; color:var(--indigo); font-weight:600; text-align:center;">
            ⏳ Parsing document pages & running RAG semantic claim audit...
          </div>
        </div>
      </div>

      <!-- RIGHT: PDF AUDIT SUMMARY & CLAIMS -->
      <div class="card">
        <div class="card-header">
          <div class="card-title-group">
            <div class="card-title">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color:var(--emerald);">
                <path d="M9 11l3 3L22 4"/>
                <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/>
              </svg>
              Document Authenticity Audit
            </div>
            <div class="card-subtitle">Aggregated document credibility & extracted claims breakdown</div>
          </div>
        </div>

        <div class="card-body">
          <div class="verdict-placeholder" id="pdfPlaceholder">
            <div class="verdict-placeholder-icon">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                <polyline points="14 2 14 8 20 8"/>
              </svg>
            </div>
            <div style="font-size:15px; font-weight:700; color:var(--text);">No Document Uploaded</div>
            <div style="font-size:13px; max-width:320px; line-height:1.5;">
              Upload a PDF news article on the left or click a pre-packaged test document to run the RAG claim verification pipeline.
            </div>
          </div>

          <div class="verdict-box" id="pdfResultBox" style="display:none;">

            <!-- OVERALL DOC BANNER -->
            <div class="verdict-banner" id="pdfVerdictBanner">
              <div class="verdict-icon-bubble" id="pdfVerdictBubble"></div>
              <div class="verdict-details">
                <div class="verdict-header-row">
                  <div class="verdict-title" id="pdfVerdictTitle">REAL NEWS DOCUMENT</div>
                  <span class="badge-risk-real" id="pdfRiskBadge">CREDIBLE</span>
                </div>
                <div class="verdict-desc" id="pdfDocSummary">
                  Document text passed automated RAG verification.
                </div>
              </div>
            </div>

            <!-- DOC METRICS -->
            <div class="metrics-mini-grid">
              <div class="metric-mini-box">
                <span class="metric-mini-lbl">Pages Scanned</span>
                <span class="metric-mini-val" id="pdfPagesVal">1</span>
              </div>
              <div class="metric-mini-box">
                <span class="metric-mini-lbl">Claims Audited</span>
                <span class="metric-mini-val" id="pdfClaimsVal">0</span>
              </div>
              <div class="metric-mini-box">
                <span class="metric-mini-lbl">Document Health</span>
                <span class="metric-mini-val" id="pdfHealthVal">100%</span>
              </div>
            </div>

            <!-- CLAIMS TABLE -->
            <div style="display:flex; flex-direction:column; gap:8px;">
              <div class="section-mini-title">Extracted Claims & Factual Integrity</div>
              <div style="max-height:280px; overflow-y:auto; border:1px solid var(--border); border-radius:var(--radius-sm);">
                <table class="pdf-claims-table">
                  <thead>
                    <tr>
                      <th style="width:40px;">P.</th>
                      <th>Extracted Claim Chunk</th>
                      <th style="width:70px;">Verdict</th>
                      <th style="width:55px;">Conf.</th>
                    </tr>
                  </thead>
                  <tbody id="pdfClaimsTableBody"></tbody>
                </table>
              </div>
            </div>

          </div>
        </div>
      </div>

    </div>

    <!-- MODEL BENCHMARKS TABLE -->
    <div class="benchmark-card" id="benchmarks">
      <div class="card-header">
        <div class="card-title-group">
          <div class="card-title">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color:var(--indigo);">
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
            </svg>
            Model Benchmark & Comparative Evaluation
          </div>
          <div class="card-subtitle">Trained on 6,335 articles using 80-20 stratified train/test split</div>
        </div>
      </div>
      <div class="table-responsive">
        <table>
          <thead>
            <tr>
              <th>Algorithm</th>
              <th>Accuracy</th>
              <th>F1-Score</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            <tr class="highlight-row">
              <td>
                <strong>Linear Support Vector Classifier (LinearSVC)</strong>
                <span class="model-badge-winner">BEST</span>
              </td>
              <td><strong>93.69%</strong></td>
              <td><strong>0.9367</strong></td>
              <td><strong>93.81%</strong></td>
              <td><strong>93.54%</strong></td>
              <td><span style="color:#4f46e5; font-weight:700;">Deployed in Production</span></td>
            </tr>
            <tr>
              <td>Logistic Regression (C=1.0, L2)</td>
              <td>92.82%</td>
              <td>0.9279</td>
              <td>93.12%</td>
              <td>92.48%</td>
              <td><span style="color:var(--text-muted);">Trained & Saved</span></td>
            </tr>
            <tr>
              <td>Passive Aggressive Classifier</td>
              <td>92.50%</td>
              <td>0.9249</td>
              <td>91.68%</td>
              <td>93.47%</td>
              <td><span style="color:var(--text-muted);">Trained & Evaluated</span></td>
            </tr>
            <tr>
              <td>Multinomial Naive Bayes (alpha=0.1)</td>
              <td>89.66%</td>
              <td>0.8962</td>
              <td>90.80%</td>
              <td>88.41%</td>
              <td><span style="color:var(--text-muted);">Baseline Benchmark</span></td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- EDA & VISUALIZATIONS GALLERY -->
    <div class="card">
      <div class="card-header">
        <div class="card-title-group">
          <div class="card-title">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color:var(--indigo);">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
              <circle cx="8.5" cy="8.5" r="1.5"/>
              <polyline points="21 15 16 10 5 21"/>
            </svg>
            Exploratory Data Analysis (EDA) & Model Visualizations
          </div>
          <div class="card-subtitle">Click any visualization to expand in high-definition</div>
        </div>
      </div>
      <div class="gallery-grid">
        <div class="plot-item" onclick="openModal('/plots/confusion_matrix.png', 'Confusion Matrix — LinearSVC (93.7% Accuracy)')">
          <img src="/plots/confusion_matrix.png" class="plot-img" alt="Confusion Matrix">
          <div class="plot-caption">
            <span>Confusion Matrix (LinearSVC)</span>
            <span style="color:var(--indigo);">View ↗</span>
          </div>
        </div>
        <div class="plot-item" onclick="openModal('/plots/roc_curve.png', 'ROC-AUC Curve — AUC = 0.983')">
          <img src="/plots/roc_curve.png" class="plot-img" alt="ROC Curve">
          <div class="plot-caption">
            <span>ROC-AUC Curve (0.983 AUC)</span>
            <span style="color:var(--indigo);">View ↗</span>
          </div>
        </div>
        <div class="plot-item" onclick="openModal('/plots/length_distribution.png', 'Article Word Count Distribution: Real vs Fake News')">
          <img src="/plots/length_distribution.png" class="plot-img" alt="Length Distribution">
          <div class="plot-caption">
            <span>Length KDE Distribution</span>
            <span style="color:var(--indigo);">View ↗</span>
          </div>
        </div>
        <div class="plot-item" onclick="openModal('/plots/caps_ratio_comparison.png', 'Capital Letter Ratio Comparison: Boxplot')">
          <img src="/plots/caps_ratio_comparison.png" class="plot-img" alt="Capital Ratio Comparison">
          <div class="plot-caption">
            <span>Caps Ratio Boxplot</span>
            <span style="color:var(--indigo);">View ↗</span>
          </div>
        </div>
        <div class="plot-item" onclick="openModal('/plots/frequent_words.png', 'Top 15 Frequent Words: Real vs Fake News')">
          <img src="/plots/frequent_words.png" class="plot-img" alt="Frequent Words">
          <div class="plot-caption">
            <span>Top Frequent Vocabulary</span>
            <span style="color:var(--indigo);">View ↗</span>
          </div>
        </div>
        <div class="plot-item" onclick="openModal('/plots/wordcloud_fake.png', 'WordCloud — Fake News Corpus')">
          <img src="/plots/wordcloud_fake.png" class="plot-img" alt="Fake News WordCloud">
          <div class="plot-caption">
            <span>Fake News WordCloud</span>
            <span style="color:var(--indigo);">View ↗</span>
          </div>
        </div>
      </div>
    </div>

  </main>

  <!-- MODAL FOR HIGH-RES PLOTS -->
  <div class="modal-backdrop" id="plotModal" onclick="closeModal(event)">
    <div class="modal-content" onclick="event.stopPropagation()">
      <div class="modal-header">
        <strong id="modalTitle" style="font-size:15px; color:var(--text);">Visualization Preview</strong>
        <button class="modal-close" onclick="closeModal()">&times;</button>
      </div>
      <div class="modal-body">
        <img id="modalImg" src="" class="modal-img" alt="Visualization">
      </div>
    </div>
  </div>

  <!-- FOOTER -->
  <footer>
    <div>
      <strong>Project 15:</strong> Machine Learning–Based Fake News Detector Using Text Classification
    </div>
    <div>
      Powered by Scikit-Learn 1.6.1 · LinearSVC · PyPDF · Vercel Serverless
    </div>
  </footer>

  <!-- JAVASCRIPT LOGIC -->
  <script>
    function switchMode(mode) {
      const textBtn = document.getElementById('tabTextBtn');
      const pdfBtn = document.getElementById('tabPdfBtn');
      const textSec = document.getElementById('modeTextSection');
      const pdfSec = document.getElementById('modePdfSection');

      if (mode === 'text') {
        textBtn.className = 'tab-btn active';
        pdfBtn.className = 'tab-btn';
        textSec.style.display = 'grid';
        pdfSec.style.display = 'none';
      } else {
        textBtn.className = 'tab-btn';
        pdfBtn.className = 'tab-btn active';
        textSec.style.display = 'none';
        pdfSec.style.display = 'grid';
      }
    }

    const SAMPLES = {
      1: {
        title: "Kerry to go to Paris in gesture of sympathy",
        content: "U.S. Secretary of State John F. Kerry said Monday that he will stop in Paris later this week, amid criticism that no top American officials attended Sunday's unity march against terrorism. Kerry said he expects to arrive in Paris Thursday evening, as he heads home after a week abroad. He said he will fly to France at the conclusion of a series of meetings with European leaders to offer condolences and reiterate American solidarity with the people of France."
      },
      2: {
        title: "You Can Smell Hillary's Fear and Panic Over Leaks",
        content: "Daniel Greenfield, a Shillman Journalism Fellow at the Freedom Center, is a New York writer focusing on radical Islam. In the final stretch of the election, Hillary Rodham Clinton has gone to war with the FBI. The word 'unprecedented' has been thrown around so often this election that it ought to be retired. But it's still unprecedented for the establishment media to smear federal investigators uncovering corruption. SHARE THIS BEFORE THE MAINSTREAM MEDIA TAKES IT DOWN!"
      },
      3: {
        title: "Federal Reserve Signals Potential Interest Rate Cut Following Inflation Drop",
        content: "The Federal Reserve chairman announced on Wednesday that central bank policymakers observed a consistent cooling in consumer price index data, creating conditions conducive for an easing of monetary policy in upcoming quarters. Economic analysts noted that sustained deceleration in core services inflation provides room for the Federal Open Market Committee to recalibrate borrowing costs without reigniting demand-side pressures."
      },
      4: {
        title: "BREAKING: Secret Alien Civilization Discovered Under White House Basement",
        content: "SHOCKING insider leak proves world leaders have been hiding a subterranean alien species living beneath Washington D.C. for decades without public knowledge or congressional oversight! Former intelligence operatives confirm secret underground tunnels connecting key federal buildings to alien spacecraft launch pads! SHARE IMMEDIATELY BEFORE CENSORSHIP BEGINS!!"
      }
    };

    function loadSample(id) {
      const sample = SAMPLES[id];
      if (!sample) return;
      document.getElementById('newsTitle').value = sample.title;
      document.getElementById('newsContent').value = sample.content;
      updateLiveMetrics();
      analyzeArticle();
    }

    function clearForm() {
      document.getElementById('newsTitle').value = '';
      document.getElementById('newsContent').value = '';
      updateLiveMetrics();
      document.getElementById('verdictBox').style.display = 'none';
      document.getElementById('verdictPlaceholder').style.display = 'flex';
    }

    function updateLiveMetrics() {
      const title = document.getElementById('newsTitle').value;
      const content = document.getElementById('newsContent').value;
      const full = (title + ' ' + content).trim();
      
      const chars = full.length;
      const words = full.length ? full.split(/\s+/).length : 0;
      let caps = 0;
      for (let c of full) { if (c >= 'A' && c <= 'Z') caps++; }
      const ratio = chars > 0 ? ((caps / chars) * 100).toFixed(1) : "0.0";

      document.getElementById('metricChars').innerText = chars;
      document.getElementById('metricWords').innerText = words;
      document.getElementById('charCountNotice').innerText = chars + " characters";

      const badge = document.getElementById('metricCaps');
      badge.innerText = ratio + "%";
      badge.className = "caps-badge " + (ratio > 10 ? "caps-alert" : (ratio > 4 ? "caps-high" : "caps-normal"));
    }

    async function analyzeArticle() {
      const title = document.getElementById('newsTitle').value.trim();
      const content = document.getElementById('newsContent').value.trim();
      
      if (!title && !content) {
        alert("Please enter a headline or article text to analyze.");
        return;
      }

      const btn = document.getElementById('btnAnalyze');
      const origText = btn.innerHTML;
      btn.disabled = true;
      btn.innerHTML = `<span style="display:inline-block; animation:spin 1s linear infinite;">⏳</span> Analyzing in Real-Time...`;

      try {
        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title, content })
        });
        const data = await res.json();

        if (res.ok) {
          renderVerdict(data);
        } else {
          alert(data.error || "Failed to analyze article.");
        }
      } catch (err) {
        alert("Network or inference error: " + err.message);
      } finally {
        btn.disabled = false;
        btn.innerHTML = origText;
      }
    }

    function renderVerdict(data) {
      document.getElementById('verdictPlaceholder').style.display = 'none';
      const box = document.getElementById('verdictBox');
      box.style.display = 'flex';

      const isFake = data.label === 'FAKE';
      const banner = document.getElementById('verdictBanner');
      const bubble = document.getElementById('verdictBubble');
      const title = document.getElementById('verdictTitle');
      const badge = document.getElementById('verdictRiskBadge');
      const desc = document.getElementById('verdictDesc');

      if (isFake) {
        banner.className = 'verdict-banner banner-fake';
        bubble.className = 'verdict-icon-bubble icon-fake';
        bubble.innerHTML = `<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`;
        title.className = 'verdict-title title-fake';
        title.innerText = 'FAKE / MISLEADING NEWS';
        badge.className = 'badge-risk-fake';
        badge.innerText = 'HIGH RISK';
        desc.innerText = 'Elevated sensationalism, unverified attribution patterns, or conspiratorial lexical features detected in real-time.';
      } else {
        banner.className = 'verdict-banner banner-real';
        bubble.className = 'verdict-icon-bubble icon-real';
        bubble.innerHTML = `<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>`;
        title.className = 'verdict-title title-real';
        title.innerText = 'REAL / CREDIBLE NEWS';
        badge.className = 'badge-risk-real';
        badge.innerText = 'CREDIBLE';
        desc.innerText = 'Lexical structure, neutral syntax, and attribution tokens correspond closely to verified authentic news reports.';
      }

      // Confidence bars
      document.getElementById('verdictConfidenceText').innerText = data.confidence.toFixed(1) + "%";
      document.getElementById('valProbFake').innerText = data.fake_probability.toFixed(1) + "%";
      document.getElementById('valProbReal').innerText = data.real_probability.toFixed(1) + "%";
      document.getElementById('barFake').style.width = data.fake_probability + "%";
      document.getElementById('barReal').style.width = data.real_probability + "%";

      // Metrics
      document.getElementById('metricDecisionScore').innerText = (data.decision_score > 0 ? "+" : "") + data.decision_score;
      document.getElementById('metricCapsResult').innerText = data.metadata.caps_ratio + "%";
      document.getElementById('metricTokenCount').innerText = data.metadata.cleaned_tokens_count;

      // Influential Keywords
      const cloud = document.getElementById('pillCloud');
      cloud.innerHTML = '';
      const realTokens = data.influential_real_terms || [];
      const fakeTokens = data.influential_fake_terms || [];

      if (realTokens.length === 0 && fakeTokens.length === 0) {
        cloud.innerHTML = '<span style="font-size:12px; color:var(--text-muted);">No high-magnitude TF-IDF outlier tokens detected.</span>';
      } else {
        realTokens.forEach(t => {
          const pill = document.createElement('span');
          pill.className = 'word-pill pill-real';
          pill.innerText = '✓ ' + t;
          cloud.appendChild(pill);
        });
        fakeTokens.forEach(t => {
          const pill = document.createElement('span');
          pill.className = 'word-pill pill-fake';
          pill.innerText = '⚠ ' + t;
          cloud.appendChild(pill);
        });
      }
    }

    // ══════════════════════════════════════════════
    // PDF AUDIT FUNCTIONS
    // ══════════════════════════════════════════════

    async function handleFileUpload(files) {
      if (!files || files.length === 0) return;
      const file = files[0];
      const formData = new FormData();
      formData.append('file', file);
      runPdfAudit('/api/analyze-pdf', formData);
    }

    async function testPdfSample(type) {
      runPdfAudit('/api/analyze-pdf?sample=' + type, null);
    }

    async function runPdfAudit(url, body) {
      const notice = document.getElementById('pdfLoadingNotice');
      notice.style.display = 'block';

      try {
        const opts = body ? { method: 'POST', body } : { method: 'GET' };
        const res = await fetch(url, opts);
        const data = await res.json();

        if (res.ok) {
          renderPdfAudit(data);
        } else {
          alert(data.error || "Failed to audit PDF.");
        }
      } catch (err) {
        alert("Error during PDF audit: " + err.message);
      } finally {
        notice.style.display = 'none';
      }
    }

    function renderPdfAudit(data) {
      document.getElementById('pdfPlaceholder').style.display = 'none';
      const box = document.getElementById('pdfResultBox');
      box.style.display = 'flex';

      const isFake = data.overall_verdict === 'FAKE';
      const banner = document.getElementById('pdfVerdictBanner');
      const bubble = document.getElementById('pdfVerdictBubble');
      const title = document.getElementById('pdfVerdictTitle');
      const badge = document.getElementById('pdfRiskBadge');
      const desc = document.getElementById('pdfDocSummary');

      if (isFake) {
        banner.className = 'verdict-banner banner-fake';
        bubble.className = 'verdict-icon-bubble icon-fake';
        bubble.innerHTML = `<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/></svg>`;
        title.className = 'verdict-title title-fake';
        title.innerText = 'SUSPICIOUS / FAKE DOCUMENT';
        badge.className = 'badge-risk-fake';
        badge.innerText = 'HIGH RISK';
      } else {
        banner.className = 'verdict-banner banner-real';
        bubble.className = 'verdict-icon-bubble icon-real';
        bubble.innerHTML = `<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>`;
        title.className = 'verdict-title title-real';
        title.innerText = 'AUTHENTIC / CREDIBLE DOCUMENT';
        badge.className = 'badge-risk-real';
        badge.innerText = 'CREDIBLE';
      }

      desc.innerText = data.summary;
      document.getElementById('pdfPagesVal').innerText = data.pages_count;
      document.getElementById('pdfClaimsVal').innerText = data.total_claims;
      const healthPct = data.total_claims > 0 ? ((data.real_claims_count / data.total_claims) * 100).toFixed(0) : "100";
      document.getElementById('pdfHealthVal').innerText = healthPct + "%";

      const tbody = document.getElementById('pdfClaimsTableBody');
      tbody.innerHTML = '';
      (data.claims || []).forEach(c => {
        const tr = document.createElement('tr');
        const badgeHtml = c.verdict === 'REAL' 
          ? `<span class="badge-tag-real">✓ REAL</span>` 
          : `<span class="badge-tag-fake">⚠ FAKE</span>`;
        tr.innerHTML = `
          <td style="color:var(--text-muted); font-weight:600;">${c.page}</td>
          <td style="color:var(--text-sub); line-height:1.4;">${c.snippet}</td>
          <td>${badgeHtml}</td>
          <td style="font-family:'JetBrains Mono', monospace; font-weight:600;">${c.confidence}%</td>
        `;
        tbody.appendChild(tr);
      });
    }

    function openModal(src, title) {
      document.getElementById('modalImg').src = src;
      document.getElementById('modalTitle').innerText = title;
      document.getElementById('plotModal').style.display = 'flex';
    }

    function closeModal() {
      document.getElementById('plotModal').style.display = 'none';
    }
  </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
