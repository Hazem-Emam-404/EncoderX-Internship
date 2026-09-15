# 📊 Task 2: Sentiment Analysis Using NLP
**EncoderX Remote Internship — AI / Machine Learning Track (Batch 02 - Week 02)**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLP-NLTK-green.svg)](https://www.nltk.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 1. Project Overview

Sentiment analysis is a fundamental Natural Language Processing (NLP) task utilized widely in customer support, marketing intelligence, brand monitoring, and product feedback analysis. The objective of this project is to develop an end-to-end NLP classification pipeline that ingests raw customer reviews, performs text preprocessing and feature extraction, trains multiple classification models, evaluates performance using rigorous metrics, and provides practical business insights.

### Core Objectives:
- Implement a robust **text cleaning and preprocessing pipeline** tailored for sentiment polarity.
- Transform unstructured text into meaningful numerical vectors using **TF-IDF with unigrams and bigrams**.
- Train and compare three supervised learning algorithms: **Logistic Regression**, **Multinomial Naive Bayes**, and **Random Forest**.
- Evaluate model performance using **Accuracy, Precision, Recall, F1-Score**, and **Confusion Matrices**.
- Formulate practical **business applications** and identify architectural pathways for future improvement.

---

## 📁 2. Dataset Selection & Understanding

- **Dataset:** [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews)
- **Raw Observations:** 568,454 user reviews across 10 features (`Id`, `ProductId`, `UserId`, `ProfileName`, `HelpfulnessNumerator`, `HelpfulnessDenominator`, `Score`, `Time`, `Summary`, `Text`).
- **Target Formulation:**
  - **Positive Sentiment (`1`)**: Star rating `Score > 3` (4 & 5 stars).
  - **Negative Sentiment (`0`)**: Star rating `Score < 3` (1 & 2 stars).
  - **Neutral (`Score == 3`)**: Dropped to eliminate ambiguous sentiment boundary noise.
- **Class Balancing:**
  - Raw split: 443,777 Positive vs. 82,037 Negative (severe class imbalance).
  - Strategy: Downsampled the Majority (Positive) class to 100,000 samples, combined with all 82,037 Negative samples, resulting in a balanced 182,037-sample dataset with stratified 80/20 train/test split.

---

## ⚙️ 3. NLP Workflow Pipeline

```
Raw Review Text
      │
      ▼
[1. Preprocessing Pipeline]
  ├── HTML & Special Character Removal
  ├── Lowercasing & Tokenization
  ├── Negation-Preserving Stopword Filtering
  └── WordNet Lemmatization
      │
      ▼
[2. Feature Extraction]
  ├── TF-IDF Vectorization (n-grams: 1-2)
  └── Vocabulary Capped at Top 10,000 Features
      │
      ▼
[3. Model Training & Validation (Stratified 80/20)]
  ├── Logistic Regression (L2 Regularization)
  ├── Multinomial Naive Bayes (Laplace Smoothing)
  └── Random Forest Classifier (100 Estimators)
      │
      ▼
[4. Evaluation & Practical Insights]
  ├── Accuracy, Macro Precision, Recall, F1
  ├── Confusion Matrix Analysis
  └── Deployment & Business Recommendations
```

### Text Preprocessing Highlights
1. **Noise Removal**: Strips HTML tags (`<br />`, `<a>`) and punctuation/special symbols using regular expressions.
2. **Case Normalization**: Converts all tokens to lowercase to ensure consistency.
3. **Tokenization**: Uses NLTK's `word_tokenize` to segment text into discrete word units.
4. **Negation Handling**: Standard stopword removal blindly eliminates words like *"not"*, *"no"*, and *"never"*, which inverts sentiment (e.g., *"not good"* becomes *"good"*). Our pipeline preserves essential negation tokens while removing non-informative stopwords.
5. **Lemmatization**: Applies NLTK `WordNetLemmatizer` to reduce words to their morphological base form.

### Feature Extraction: TF-IDF
We transform text into numerical representations using **Term Frequency - Inverse Document Frequency (TF-IDF)**:
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

- **N-gram Range `(1, 2)`**: Incorporates both single words (*unigrams*) and word pairs (*bigrams*), allowing the model to capture multi-word sentiment signals like *"highly recommend"* or *"waste money"*.
- **Max Features `10,000`**: Restricts the feature matrix to the 10,000 most predictive n-grams, reducing memory overhead while avoiding overfitting.

---

## 📈 4. Model Evaluation & Benchmark Results

All models were evaluated on an unseen, stratified test set of **36,408 reviews**:

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Strengths / Trade-offs |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Random Forest** | **90.88%** | **90.97%** | **90.61%** | **90.76%** | Highest overall accuracy; excellent non-linear feature handling; slower training time. |
| **Logistic Regression** | **90.70%** | **90.62%** | **90.59%** | **90.60%** | Near-identical performance to Random Forest; exceptionally fast training and low memory footprint; ideal for real-time production inference. |
| **Multinomial Naive Bayes** | **87.71%** | **87.89%** | **87.29%** | **87.51%** | Strong, lightweight probabilistic baseline; slightly higher false-positive rate due to word independence assumption. |

### Key Evaluation Insights:
- **Champion Model**: **Logistic Regression** is the recommended choice for production deployment due to its high F1-Score (90.60%) combined with sub-millisecond inference latency and minimal resource consumption compared to Random Forest.
- **Error Breakdown**: False negatives primarily stemmed from subtle sarcasm, complex reviews containing mixed praise/criticism, and rare slang.

---

## 💼 5. Practical Business Applications

1. **Customer Support Prioritization (Ticket Triage)**:
   - Automatically route high-urgency negative reviews to senior support agents with SLA guarantees, preventing customer churn.
2. **Product Quality & Defect Detection**:
   - Aggregate negative sentiment clusters across specific Product IDs to alert QA and product development teams about recurring manufacturing or packaging defects.
3. **Marketing Campaign Effectiveness**:
   - Quantify customer sentiment before, during, and after brand campaigns or promotional discounts to measure true customer reception.
4. **Competitor Benchmarking**:
   - Apply the sentiment classifier to publicly available competitor product reviews to identify market vulnerabilities and competitive advantages.
5. **Real-Time Brand Reputation Monitoring**:
   - Ingest live social media streams (e.g., Twitter/X, Reddit) to identify viral PR crises before they escalate.

---

## 🚀 6. Future Improvements

- **Fine-Tuning Transformer Architectures**: Train contextual models such as **DistilBERT** or **RoBERTa** to better capture syntactic dependencies, negation subtleties, and sarcasm.
- **Aspect-Based Sentiment Analysis (ABSA)**: Break down sentiment by individual aspects (e.g., Taste: Positive, Packaging: Negative, Shipping Speed: Neutral) rather than assigning a single document-level label.
- **Handling Emojis & Informal Internet Slang**: Integrate emoji-to-text translators (e.g., `emoji` library) to leverage emoticon sentiment signals.
- **Hyperparameter Optimization**: Conduct Bayesian or Grid Search optimization over vectorizer parameters (`min_df`, `max_df`, `sublinear_tf`) and model regularization strengths (`C` in Logistic Regression).

---

## 🛠️ 7. Project Structure & Setup

### Directory Layout
```
├── AIML_Week02_Batch02.pdf    # EncoderX Internship Task Specification
├── notebook.ipynb             # Complete end-to-end Jupyter Notebook
├── README.md                  # Comprehensive Project Documentation
├── requirements.txt           # Python Package Dependencies
└── SUBMISSION_GUIDE.md        # LinkedIn Post, Video Demo, & Submission Checklist
```

### Installation & Execution

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the notebook:**
   ```bash
   jupyter notebook notebook.ipynb
   ```
   *(Note: When running on Kaggle, the dataset is accessible at `/kaggle/input/datasets/arhamrumi/amazon-product-reviews/Reviews.csv`)*

---

## 👥 8. Author & Internship Details

- **Program:** EncoderX Remote Internship (Batch 02)
- **Track:** Artificial Intelligence & Machine Learning
- **Task:** Week 02 — Task 2: Sentiment Analysis Using NLP
- **Evaluation Criteria Alignment:**
  - Text Preprocessing (20%): Completed with negation preservation & regex cleaning.
  - Feature Engineering (20%): Completed with TF-IDF n-grams `(1, 2)`.
  - Model Performance (25%): 3 models trained, evaluated, and confusion matrices plotted.
  - Analysis & Insights (20%): Business applications, error analysis, and actionable recommendations documented.
  - Documentation (15%): Comprehensive README and detailed notebook markdown.
