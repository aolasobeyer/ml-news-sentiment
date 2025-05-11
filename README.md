# ml-news-sentiment
Sentiment classification of financial news using FinBERT and multiple NLP models with interactive dashboard

# Financial‐News Sentiment Classification
### Machine Learning Applications ‑ Final Project

*Aitor Daniel Olaso Beyer – Bachelor in Data Science & Engineering – May 2025*

---

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset Creation](#dataset)
3. [Task 1 – Text Pre‑processing & Vectorisation](#task1)

   1. Data Collection Pipeline
   2. Cleaning & Normalisation
   3. Vector Representations
   4. Topic Modelling (LDA)
   5. Exploratory Visualisations
4. [Task 2 – Machine‑Learning Models](#task2)

   1. Experimental Design
   2. Baseline Models
   3. Classical Models & Results
   4. Ensemble Model
   5. Error Analysis & Interpretability
5. [Task 3 – Dash Interactive Dashboard](#task3)
6. [Acknowledgement of Authorship](#ack)
7. [References](#ref)

---

<a name="introduction"></a>

## 1  Introduction

Understanding the *mood* of financial markets in real‑time is a crucial yet challenging task.  While Transformer–based language models dominate modern NLP, this project **deliberately restricts itself to classical, CPU‑friendly techniques** to (i) deepen my theoretical grasp, (ii) work under resource‑constrained settings, and (iii) demonstrate that solid performance is still attainable without heavyweight deep nets.  The headline metric — **macro F1 = 0.768** — confirms that goal.

The work follows the official project guideline (see PDF) and is split into three deliverables:

* **Task 1** – create & vectorise a 3 k‑document corpus of financial news;
* **Task 2** – benchmark multiple ML pipelines and build an ensemble classifier;
* **Task 3** – expose the results in an interactive *Dash* dashboard for business users.

All experiments were executed **on CPU only** (4‑core i7, 16 GB RAM) inside Google Colab.  No external GPU/TPU nor fine‑tuned Transformers were used for classification; FinBERT appears *only* as an automated labeller for bootstrapping the dataset.

---

<a name="dataset"></a>

## 2  Dataset Creation

### 2.1  Collection strategy

* **Source API** – [Finnhub](https://finnhub.io/) company‑news endpoint.
* **Universe** – 12 mega‑cap tickers (AAPL, MSFT, …), ensuring diverse yet liquid coverage.
* **Time window** – 30 rolling days (\~23 × 12 ≈ 276 requests, safely within the free‑tier quota).
* **Volume** – 3 021 unique articles after de‑duplication.

The script *run\_pipeline()* handles rate‑limiting (400 ms sleep) and stores a raw JSON snapshot for reproducibility.

### 2.2  Automatic labelling with FinBERT

Each article’s headline + summary is fed to **FinBERT (yiyanghkust/finbert‑tone)**.  Predictions (`positive`, `negative`, `neutral`) and confidence scores are appended.  Manual spot‑checks on 100 samples showed \~85 % face‑validity, acceptable for a weak‑supervision seed.

### 2.3  Cleaning & validation

\* Duplicate headlines removed (\~4 %).
\* Empty texts discarded (<1 %).
\* Final class balance:  **neg 20 % / neu 55 % / pos 25 %**.

> *Put Figure — probability histogram here.*

---

<a name="task1"></a>

## 3  Task 1 – Text Pre‑processing & Vectorisation (≤6 pp.)

### 3.1  SpaCy‑based pipeline

1. **HTML scrub** (`<.*?>`), non‑alphabetic removal.
2. Lower‑case + tokenisation (`en_core_web_sm`).
3. Lemmatisation.
4. Stop‑word drop (`nltk` list + custom finance terms).
5. Tokens cached as list & as space‑joined string.

Average length dropped from 31 to 18 tokens — a 42 % reduction that eases downstream vectorisers.

### 3.2  Vectorisation strategies

| Representation         |     Dim | Notes                                                         |
| ---------------------- | ------: | ------------------------------------------------------------- |
| **TF‑IDF** (1–2 grams) |  10 000 | L2‑norm; captures term frequency bias.                        |
| **GloVe 100 d**        |     100 | Pre‑trained (6B tokens); doc = mean of word vectors.          |
| **Word2Vec 100 d**     |     100 | Trained *only* on training split to avoid leakage.            |
| **Lexicon scores**     |       3 | Positive / Negative / Uncertainty ratios (Loughran–McDonald). |
| **LDA topics**         |       8 | Coherence optimised; later used for dashboard only.           |

### 3.3  Visual sanity checks

* **PCA(2D)** and **t‑SNE** plots signal mild class separation (Fig. t‑SNE GloVe).
* Scree plot shows 90 % variance at ≈140 components, underpinning curse‑of‑dimensionality concerns.

> *Put Figure — GloVe t‑SNE here.*

### 3.4  Top‑terms insight

Coefficient inspection of the logistic‑TFIDF model surfaces intuitive words:

*Negative* → *face*, *downgrade*, *loss* …
*Positive* → *buy*, *strong*, *growth* …

> *Put Figure — Top 20 TF‑IDF negative*
> *Put Figure — Top 20 TF‑IDF positive*

---

<a name="task2"></a>

## 4  Task 2 – Machine‑Learning Models (≤5 pp.)

### 4.1  Experimental design

* **Split** – stratified 80 / 20 train–test (seed 42 ⇒ 601 test rows).
* **Metrics** – macro‑F1 primary (class balance), per‑class recall, calibration curve.
* **Validation** – 5‑fold CV inside `GridSearchCV` where applicable.
* **Feature sets** – TF‑IDF, GloVe, Word2Vec, Lexicon + concatenations.

### 4.2  Baselines

| Model              | Feature | Macro‑F1 |
| ------------------ | ------- | -------: |
| Dummy (stratified) |  –      |    0.333 |
| LogisticReg        |  TF‑IDF |    0.722 |
| Linear SVC         |  TF‑IDF |    0.739 |
| XGBoost            |  GloVe  |    0.713 |

### 4.3  Ensemble — soft voting

Weights (2 : 3 : 1) were tuned on a held‑out slice to privilege the calibrated SVC.  Combined **macro‑F1 = 0.7684**.

> *Put Figure — Confusion matrix here.*

Observations:

* Most confusion arises between *positive* and *neutral* — mirrors FinBERT’s original ambiguity.
* Recall‑neg = 0.68 beats naive short‑bias articles detection baselines.

### 4.4  Probability calibration

Both the `CalibratedClassifierCV` (SVC) and the ensemble track the diagonal well (see calibration curve) up to 0.8.  Reliable probabilities matter for downstream risk‑weighting.

> *Put Figure — Calibration curves here.*

### 4.5  Interpretability

SHAP bar‑plot (XGB on GloVe) highlights *risk*, *cut*, *boost*, *bullish* as strong drivers — aligning with TF‑IDF view and enhancing stakeholder trust.

### 4.6  Runtime footprint

Full training (all vectors + models) finished in **≈22 min on CPU**; memory peak 7.3 GB.  This validates the classical approach for lightweight deployments.

---

<a name="task3"></a>

## 5  Task 3 – Dash Interactive Dashboard (≈1 p.)

The *Fin‑News Sentiment Explorer* lets portfolio managers filter thousands of headlines in real‑time.

| Component                             | Interaction                                | Insight                                              |
| ------------------------------------- | ------------------------------------------ | ---------------------------------------------------- |
| **Scatter plot** (PCA‑2D)             | Lasso select; colour = predicted sentiment | Cluster outliers & drill to raw text.                |
| **Topic bar**                         | Updates on topic‑dropdown                  | Top words reveal latent themes (e.g. *AI*, *cloud*). |
| **Histogram** of class‑specific probs | Auto‑filters with sentiment/topic          | Gauge prediction confidence distribution.            |
| **Data table**                        | Reactive to scatter selection              | Quick copy‑paste for analysts.                       |

> *Put composite dashboard screenshot here.*

The callbacks are fully vectorised (no loops), ensuring snappy UX even on my laptop’s localhost.

---

<a name="ack"></a>

## 6  Acknowledgement of Authorship

*All code* inside the `src/` folder and the dashboard was written by **Daniel Olaso**.  Third‑party libraries are cited in §7.

### Where ChatGPT (o3) helped

| Section                        | Difficulty                                   | Prompt I used                                                                                 | How I integrated the answer                          |
| ------------------------------ | -------------------------------------------- | --------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| Confusion‑matrix normalisation | I forgot the `normalize="true"` flag         | *“Why does my sklearn ConfusionMatrixDisplay look wrong? How do I normalise by true labels?”* | Adopted the parameter suggestion verbatim.           |
| SHAP bar plot                  | Needed compact summary for >100 features     | *“Show an example of shap.summary\_plot with bar type for multi‑class XGB”*                   | Copied 4‑line snippet, adapted to my variable names. |
| Dash callback pattern‑matching | First time chaining multiple `Input` objects | *“Dash: update DataTable based on lasso‑selection of scatter plot and dropdown filters”*      | Used the skeleton but rewired IDs.                   |

ChatGPT **did not** generate large blocks of code, its role was helping with errors and giving code skeletons.

---

<a name="ref"></a>

## 7  References

1. Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with BERT.*
2. Pennington, J., Socher, R., & Manning, C. D. (2014). *GloVe: Global Vectors for Word Representation.*
3. Loughran, T., & McDonald, B. (2011). *When Is a Liability Not a Liability?* Journal of Finance.
4. scikit‑learn v1.6 documentation.
5. Dash Plotly 2.17 docs.



