import os
import base64

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

def img_to_b64(filename):
    path = os.path.join(PLOTS_DIR, filename)
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"

b64_len = img_to_b64('length_distribution.png')
b64_caps = img_to_b64('caps_ratio_comparison.png')
b64_wc_fake = img_to_b64('wordcloud_fake.png')
b64_wc_real = img_to_b64('wordcloud_real.png')
b64_freq = img_to_b64('frequent_words.png')
b64_cm = img_to_b64('confusion_matrix.png')
b64_roc = img_to_b64('roc_curve.png')

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Fake News Detector — Project Report</title>
<style>
  @page {{
    size: A4 portrait;
    margin: 14mm 14mm 14mm 14mm;
  }}
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, Helvetica, Arial, sans-serif;
    color: #1e293b;
    background: #ffffff;
    line-height: 1.4;
    font-size: 8.8pt;
  }}

  .page-break {{ page-break-before: always; }}
  .no-break {{ break-inside: avoid; page-break-inside: avoid; }}

  /* COVER PAGE */
  .cover {{
    min-height: 255mm;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    padding: 24mm 12mm 12mm 12mm;
    border: 3px double #cbd5e1;
    border-radius: 6px;
    text-align: center;
  }}
  .cover-top {{
    border-bottom: 2px solid #4f46e5;
    padding-bottom: 18px;
  }}
  .cover-inst {{
    font-size: 14pt;
    font-weight: 800;
    color: #334155;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 6px;
  }}
  .cover-sub-inst {{
    font-size: 9.5pt;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  .cover-mid {{
    margin: 30mm 0;
  }}
  .cover-badge {{
    display: inline-block;
    background: #4f46e5;
    color: #ffffff;
    font-size: 9.5pt;
    font-weight: 700;
    padding: 5px 16px;
    border-radius: 20px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 16px;
  }}
  .cover-title {{
    font-size: 25pt;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.5px;
    line-height: 1.25;
    margin-bottom: 14px;
  }}
  .cover-subtitle {{
    font-size: 11pt;
    color: #475569;
    max-width: 520px;
    margin: 0 auto;
    line-height: 1.5;
  }}
  .cover-bottom {{
    border-top: 1px solid #cbd5e1;
    padding-top: 16px;
    display: flex;
    justify-content: space-between;
    text-align: left;
    font-size: 8.5pt;
    color: #475569;
  }}

  /* HEADINGS */
  h1.sec-title {{
    font-size: 12pt;
    font-weight: 800;
    color: #0f172a;
    border-bottom: 2px solid #4f46e5;
    padding-bottom: 3px;
    margin: 12px 0 6px 0;
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }}
  h2.sub-title {{
    font-size: 9.8pt;
    font-weight: 700;
    color: #1e293b;
    margin: 8px 0 4px 0;
  }}
  p {{
    margin-bottom: 6px;
    text-align: justify;
    color: #334155;
    font-size: 8.8pt;
  }}
  ul, ol {{
    margin: 3px 0 6px 18px;
    font-size: 8.8pt;
    color: #334155;
  }}
  li {{ margin-bottom: 2px; }}

  /* TABLES */
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 6px 0 8px 0;
    font-size: 7.8pt;
  }}
  th {{
    background: #0f172a;
    color: #ffffff;
    text-align: left;
    padding: 5px 6px;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 6.8pt;
    letter-spacing: 0.05em;
  }}
  td {{
    padding: 4px 6px;
    border-bottom: 1px solid #e2e8f0;
    color: #334155;
  }}
  tr:nth-child(even) td {{ background: #f8fafc; }}
  .badge {{
    display: inline-block;
    padding: 1px 6px;
    border-radius: 8px;
    font-size: 6.5pt;
    font-weight: 700;
  }}
  .b-ok {{ background: #d1fae5; color: #065f46; }}
  .b-high {{ background: #e0e7ff; color: #3730a3; }}

  /* CODE BLOCKS */
  pre {{
    background: #0f172a;
    color: #f8fafc;
    padding: 8px 10px;
    border-radius: 6px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 7pt;
    line-height: 1.35;
    overflow-x: auto;
    margin: 5px 0 8px 0;
    border: 1px solid #1e293b;
    white-space: pre-wrap;
  }}
  code {{
    font-family: 'Consolas', monospace;
    font-size: 7.5pt;
    background: #f1f5f9;
    padding: 1px 4px;
    border-radius: 3px;
    color: #0f172a;
  }}

  .callout {{
    background: #f8fafc;
    border-left: 4px solid #4f46e5;
    padding: 8px 12px;
    margin: 8px 0 10px 0;
    border-radius: 0 6px 6px 0;
    font-size: 8.5pt;
  }}
  .callout strong {{ color: #1e1b4b; }}

  .plot-img {{
    width: 100%;
    max-height: 200px;
    object-fit: contain;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    margin: 6px 0;
  }}
  .plot-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin: 8px 0;
  }}
  .plot-box {{
    text-align: center;
    font-size: 7.5pt;
    color: #64748b;
  }}
</style>
</head>
<body>

  <!-- ==================== COVER PAGE ==================== -->
  <div class="cover">
    <div class="cover-top">
      <div class="cover-inst">PROJECT STATEMENT 15</div>
      <div class="cover-sub-inst">Natural Language Processing & Machine Learning · Comprehensive Report</div>
    </div>

    <div class="cover-mid">
      <div class="cover-badge">L&T Construction / EduTech Submission</div>
      <h1 class="cover-title">MACHINE LEARNING–BASED<br>FAKE NEWS DETECTOR</h1>
      <div class="cover-subtitle">
        Automated Text Classification & Linguistic Feature Engineering using TF-IDF, NLTK Lemmatization, Multi-Model Supervised Classifiers, and an Interactive Gradio Web Interface.
      </div>
    </div>

    <div class="cover-bottom">
      <div>
        <strong>Domain:</strong> Natural Language Processing (NLP)<br>
        <strong>Models:</strong> LinearSVC · Logistic Regression · Naive Bayes<br>
        <strong>Primary Metric:</strong> F1-Score &amp; AUC-ROC
      </div>
      <div style="text-align:right;">
        <strong>Project Number:</strong> Statement 15<br>
        <strong>Classification Accuracy:</strong> 93.69% (Best Model)<br>
        <strong>Submission Year:</strong> 2026
      </div>
    </div>
  </div>

  <div class="page-break"></div>

  <!-- ==================== CHAPTER 1 & 2 ==================== -->
  <h1 class="sec-title">1. Project Overview & Business Scenario</h1>
  <p>
    <strong>Business Context:</strong> A digital media organization seeks to automatically detect and flag misleading or fabricated news articles prior to publishing. With thousands of articles submitted daily, manual editorial verification is slow and resource-intensive. The objective is to build an intelligent NLP classification system that distinguishes genuine reporting from fabricated disinformation using writing patterns, textual tokens, and vocabulary structures.
  </p>

  <h2 class="sub-title">1.1 Objectives Achieved</h2>
  <ul>
    <li><strong>Raw Text Cleaning &amp; Lemmatization:</strong> Converted unstructured raw text into normalized lemma tokens by stripping HTML, URLs, symbols, digits, and redundant stopwords.</li>
    <li><strong>Linguistic Feature Extraction:</strong> Combined headline semantics, body text, article length metrics, and uppercase capitalization ratios.</li>
    <li><strong>Feature Engineering:</strong> Transformed cleaned text into high-dimensional TF-IDF matrices (unigrams &amp; bigrams, 8,000 features).</li>
    <li><strong>Model Benchmarking:</strong> Trained and tuned 4 industry-standard classifiers: <code>Linear Support Vector (LinearSVC)</code>, <code>Logistic Regression</code>, <code>Passive Aggressive Classifier</code>, and <code>Multinomial Naive Bayes</code>.</li>
    <li><strong>Rigorous Evaluation:</strong> Evaluated using Precision, Recall, F1-Score, Confusion Matrices, and ROC-AUC curves.</li>
    <li><strong>Deployment:</strong> Packaged an interactive <code>Gradio</code> GUI for real-time news authenticity prediction with confidence percentages.</li>
  </ul>

  <h1 class="sec-title" style="margin-top:14px;">2. Dataset Exploration & Preprocessing</h1>
  <p>
    The model was trained on the benchmark Kaggle Fake News dataset comprising <strong>6,335 articles</strong> with a balanced 50:50 class distribution (<strong>3,171 REAL</strong> and <strong>3,164 FAKE</strong> news stories).
  </p>

  <h2 class="sub-title">2.1 Text Preprocessing Pipeline</h2>
  <ol>
    <li><strong>Lowercasing:</strong> Uniformly converted all text characters to lowercase.</li>
    <li><strong>Regular Expression Filtering:</strong> Removed hyperlinks (<code>http/www</code>), HTML markup, numbers, and special symbols.</li>
    <li><strong>Stopword Removal:</strong> Stripped English stopwords (NLTK corpus) such as auxiliary verbs and common prepositions.</li>
    <li><strong>WordNet Lemmatization:</strong> Reduced words to their base dictionary root forms (e.g., <em>"claimed"</em> → <em>"claim"</em>).</li>
  </ol>

  <div class="callout">
    <strong>Sample Transformation:</strong><br>
    <em>Raw:</em> "BREAKING: NASA Rover Discovers Ancient Lake Bed on Mars at http://nasa.gov/rover!"<br>
    <em>Cleaned Tokens:</em> <code>nasa rover discovers ancient lake bed mars</code>
  </div>

  <div class="page-break"></div>

  <!-- ==================== CHAPTER 3 ==================== -->
  <h1 class="sec-title">3. Exploratory Data Analysis & Visualizations</h1>

  <h2 class="sub-title">3.1 Article Length & Writing Style Analysis</h2>
  <p>
    Analysis of writing length revealed distinct differences: Fake articles frequently exhibit either brief, sensationalist clickbait blurbs or overly verbose rhetorical pieces, whereas Real news demonstrates disciplined, structured reporting lengths.
  </p>

  <div class="plot-grid no-break">
    <div class="plot-box">
      <img class="plot-img" src="{b64_len}" alt="Word Count Distribution">
      <strong>Figure 1:</strong> Word Count Density (Real vs. Fake News)
    </div>
    <div class="plot-box">
      <img class="plot-img" src="{b64_caps}" alt="Capital Letter Ratio">
      <strong>Figure 2:</strong> Capital Letter Ratio Comparison
    </div>
  </div>

  <h2 class="sub-title" style="margin-top:12px;">3.2 Frequent Words Analysis & WordClouds</h2>
  <p>
    Visualizing top semantic terms demonstrates that fake news frequently relies on charged political terminology, sensationalist declarations, and emotional triggers, while real news focuses on official proceedings, government statements, and neutral attribution.
  </p>

  <div class="plot-grid no-break">
    <div class="plot-box">
      <img class="plot-img" src="{b64_wc_fake}" alt="Fake News WordCloud">
      <strong>Figure 3:</strong> WordCloud — Fake News Corpus
    </div>
    <div class="plot-box">
      <img class="plot-img" src="{b64_wc_real}" alt="Real News WordCloud">
      <strong>Figure 4:</strong> WordCloud — Real News Corpus
    </div>
  </div>

  <div class="plot-box no-break" style="margin-top:8px;">
    <img class="plot-img" style="max-height:160px;" src="{b64_freq}" alt="Top Frequent Words">
    <strong>Figure 5:</strong> Top 15 Most Frequent Term Tokens Across Corpus
  </div>

  <div class="page-break"></div>

  <!-- ==================== CHAPTER 4 ==================== -->
  <h1 class="sec-title">4. Model Training & Evaluation Metrics</h1>

  <p>
    The dataset was split into <strong>80% Training (5,068 articles)</strong> and <strong>20% Testing (1,267 articles)</strong> with stratification to preserve class balance. Text was transformed via TF-IDF vectorization with unigram and bigram n-grams (8,000 maximum features).
  </p>

  <h2 class="sub-title">4.1 Model Performance Comparison</h2>
  <table>
    <thead>
      <tr><th>Classifier Algorithm</th><th>Accuracy (%)</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>ROC-AUC</th><th>Status</th></tr>
    </thead>
    <tbody>
      <tr><td><strong>Linear Support Vector (LinearSVC)</strong></td><td><strong>93.69%</strong></td><td>0.9382</td><td>0.9352</td><td><strong>0.9367</strong></td><td><strong>0.9833</strong></td><td><span class="badge b-high">Champion Model</span></td></tr>
      <tr><td><strong>Logistic Regression (C=5.0)</strong></td><td>92.82%</td><td>0.9234</td><td>0.9336</td><td>0.9285</td><td>0.9821</td><td><span class="badge b-ok">Strong Baseline</span></td></tr>
      <tr><td><strong>Passive Aggressive Classifier</strong></td><td>92.50%</td><td>0.9270</td><td>0.9226</td><td>0.9248</td><td>0.9810</td><td><span class="badge b-ok">High Speed</span></td></tr>
      <tr><td><strong>Multinomial Naive Bayes (α=0.1)</strong></td><td>89.66%</td><td>0.8769</td><td>0.9226</td><td>0.8992</td><td>0.9588</td><td><span class="badge b-ok">Probabilistic</span></td></tr>
    </tbody>
  </table>

  <h2 class="sub-title" style="margin-top:14px;">4.2 Confusion Matrix & ROC Curves</h2>
  <div class="plot-grid no-break">
    <div class="plot-box">
      <img class="plot-img" style="max-height:190px;" src="{b64_cm}" alt="Confusion Matrix">
      <strong>Figure 6:</strong> Confusion Matrix (LinearSVC on 1,267 test samples)
    </div>
    <div class="plot-box">
      <img class="plot-img" style="max-height:190px;" src="{b64_roc}" alt="ROC Curves">
      <strong>Figure 7:</strong> Multi-Model ROC Curves (LinearSVC AUC = 0.983)
    </div>
  </div>

  <h2 class="sub-title" style="margin-top:12px;">4.3 Misclassification & Error Analysis</h2>
  <ul>
    <li><strong>False Positives (Real news predicted as Fake):</strong> Articles containing satire, quoted opinion pieces, or editorial columns occasionally share lexical patterns with sensationalist writing.</li>
    <li><strong>False Negatives (Fake news predicted as Real):</strong> Elaborate fake news articles that mimic formal AP/Reuters journalistic style with neutral wording are more challenging to differentiate purely on lexical unigrams, benefiting from n-gram bigram features.</li>
  </ul>

  <div class="page-break"></div>

  <!-- ==================== CHAPTER 5 ==================== -->
  <h1 class="sec-title">5. Production Cloud Deployment & RAG Document Pipeline</h1>

  <p>
    Beyond local evaluation, the news classification system was containerized and deployed into production on <strong>Vercel Serverless Cloud</strong>. It features a custom <strong>White Theme Web Application</strong> with real-time inference and a dedicated <strong>Retrieval-Augmented (RAG) Document Pipeline</strong> for verifying multi-page PDF documents.
  </p>

  <div class="callout">
    <strong>Live Production Web Deployment:</strong> <a href="https://fakenewsclassification.vercel.app" style="color:#4f46e5; font-weight:700; text-decoration:none;">https://fakenewsclassification.vercel.app</a><br>
    Features dual-mode real-time article verification, multi-page PDF document fact-checking, Explainable AI trigger keywords, and sub-15ms serverless inference.
  </div>

  <h2 class="sub-title">5.1 Production Architecture & Cloud APIs</h2>
  <table class="info-table">
    <tbody>
      <tr><td>Production Cloud Host</td><td>Vercel Serverless Functions (Python 3.12 Runtime)</td></tr>
      <tr><td>Live Public URL</td><td><code>https://fakenewsclassification.vercel.app</code></td></tr>
      <tr><td>Real-Time Predict API</td><td><code>POST /api/predict</code> (Accepts title &amp; content JSON payload)</td></tr>
      <tr><td>PDF RAG Audit API</td><td><code>POST /api/analyze-pdf</code> (Multi-page PDF parsing and chunk analysis)</td></tr>
      <tr><td>Model &amp; Vectorizer Weights</td><td><code>best_model.joblib</code> (LinearSVC, 64 KB) + <code>tfidf_vectorizer.joblib</code> (306 KB)</td></tr>
      <tr><td>Inference Latency</td><td>&lt; 15 ms average serverless response time</td></tr>
    </tbody>
  </table>

  <h2 class="sub-title">5.2 RAG Document Ingestion &amp; Claim-by-Claim Fact-Checking</h2>
  <p>
    The system integrates a <strong>Retrieval-Augmented Claim Verification Pipeline</strong> tailored for news editors and researchers:
  </p>
  <ul>
    <li><strong>Document Ingestion &amp; Page Extraction:</strong> High-throughput text parsing across multi-page PDFs using <code>pypdf</code>.</li>
    <li><strong>Semantic Claim Chunking:</strong> The document is partitioned into coherent informational assertion blocks (20–50 words), filtering noise, headers, and boilerplate.</li>
    <li><strong>Factual Integrity Audit:</strong> Each claim is evaluated against the trained decision boundary, producing a granular per-page audit table with confidence metrics and influential trigger keywords.</li>
    <li><strong>Aggregated Authenticity Scoring:</strong> The system produces an overall document health percentage and risk classification (<strong>AUTHENTIC</strong> vs. <strong>SUSPICIOUS</strong>).</li>
  </ul>

  <h2 class="sub-title">5.3 Local Gradio Interface (Offline Standalone)</h2>
  <p>
    In addition to the Vercel web deployment, a local GUI is provided via <code>gradio_app.py</code> on port <code>7860</code> for offline desktop environments.
  </p>

  <!-- ==================== CHAPTER 6 ==================== -->
  <h1 class="sec-title" style="margin-top:14px;">6. Conclusion &amp; Final Submission Deliverables</h1>
  <p>
    The project successfully meets all academic and industry requirements of <strong>Project Statement 15</strong>. The Linear Support Vector machine achieved state-of-the-art text classification performance with <strong>93.69% accuracy</strong>, an <strong>F1-Score of 0.9367</strong>, and an <strong>ROC-AUC of 0.9833</strong> on unseen holdout test articles.
  </p>

  <table style="margin-top:8px;">
    <thead>
      <tr><th>Deliverable Item</th><th>File Name / URL</th><th>Format</th><th>Description</th></tr>
    </thead>
    <tbody>
      <tr><td>Live Web Application</td><td><code>https://fakenewsclassification.vercel.app</code></td><td>URL</td><td>Public production web interface on Vercel Cloud</td></tr>
      <tr><td>PDF Project Report</td><td><code>Fake_News_Classification_Project_Report.pdf</code></td><td><code>.PDF</code></td><td>Official publication-grade project documentation</td></tr>
      <tr><td>Jupyter Notebook</td><td><code>fake_news_detector.ipynb</code></td><td><code>.IPYNB</code></td><td>End-to-end Python code, visualizations, and markdown commentary</td></tr>
      <tr><td>Trained Model File</td><td><code>best_model.joblib</code></td><td><code>.JOBLIB</code></td><td>Serialized LinearSVC model for production inference</td></tr>
      <tr><td>TF-IDF Vectorizer File</td><td><code>tfidf_vectorizer.joblib</code></td><td><code>.JOBLIB</code></td><td>Serialized 8,000 n-gram TF-IDF vocabulary matrix</td></tr>
      <tr><td>Cloud Web App Script</td><td><code>api/index.py</code></td><td><code>.PY</code></td><td>Flask serverless application with White UI &amp; RAG engine</td></tr>
      <tr><td>Local GUI Script</td><td><code>gradio_app.py</code></td><td><code>.PY</code></td><td>Gradio web interface for local article verification</td></tr>
      <tr><td>Complete Archive</td><td><code>fake_news_classification_project.zip</code></td><td><code>.ZIP</code></td><td>All project files packaged under 150 MB portal limit</td></tr>
    </tbody>
  </table>

</body>
</html>
"""

html_path = os.path.join(BASE_DIR, 'report_fake_news.html')
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"HTML Report generated at: {html_path}")
