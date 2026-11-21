# Pattern Recognition & Data Mining — Overview (Mid Exam Note)

> **Source:** "Pattern Recognition and Data Mining: Concepts and Techniques" (Data Mining Overview, 55 slides) + course videos (Intro to DM, Decision Tree, kNN Classifier, Classifier Evaluation, Clustering, Frequent Pattern / Association Rule, kNN Regression, Preprocessing & EDA (PCA, CoV), Privacy & Security).
>
> **Teacher's note:** *"Pattern Recognition & DM overview ⇒ DESCRIPTIVE question."*
> তাই এই chapter থেকে মূলত **definition, process, তুলনা (vs), উদাহরণ** লিখতে হবে — বড় অংক কম, কিন্তু ছোট calculation (accuracy, support/confidence, normalization) দিয়ে answer সাজালে নম্বর বেশি পাওয়া যায়।

---

## সূচিপত্র (Contents)

1. [Why Data Mining?](#1-why-pattern-recognition-and-data-mining)
2. [What is Data Mining? (Definition)](#2-what-is-data-mining)
3. [DM Steps / KDD Process](#3-data-mining-steps--kdd-process)
4. [EDA vs Data Mining (Past vs Future)](#4-eda-vs-data-mining-explaining-the-past-vs-predicting-the-future)
5. [Business Intelligence Pyramid](#5-data-mining-in-the-business-intelligence-bi-pyramid)
6. [Data Mining Tasks (overview table)](#6-data-mining-tasks--overview)
7. [Supervised vs Unsupervised](#7-supervised-vs-unsupervised-learning)
8. [Classification Process + Decision Tree](#8-classification-process-model-construction--model-usage)
9. [k-Nearest Neighbor (kNN) Classifier](#9-k-nearest-neighbor-knn-classifier)
10. [kNN Regression](#10-regression-and-knn-regression)
11. [Classifier Evaluation (Accuracy, Precision, Recall, F1)](#11-classifier-evaluation-metrics)
12. [Clustering (K-Means)](#12-clustering-cluster-analysis)
13. [Frequent Patterns & Association Rules](#13-frequent-pattern-analysis--association-rules)
14. [Anomaly / Outlier Detection](#14-anomaly--outlier-detection)
15. [Data Preprocessing](#15-data-preprocessing)
16. [EDA: PCA, Covariance, Correlation](#16-eda-pca-covariance-correlation)
17. [Ubiquitous & Invisible DM, Privacy & Security](#17-ubiquitous--invisible-data-mining-privacy--security)
18. [Applications](#18-applications-summary)
19. [Likely Descriptive Questions + Model Answers](#19-likely-descriptive-questions--model-answers)
20. [Exam Tips / পরীক্ষার টিপস](#20-exam-tips--পরীক্ষার-টিপস)

---

## 1. Why Pattern Recognition and Data Mining?

- **Explosive growth of data:** Terabytes → Petabytes → Exabytes।
  - $`1\text{ TB} = 1000^4`$ Bytes, $`1\text{ PB} = 1000^5`$ B, $`1\text{ EB} = 1000^6`$ B
- **Data কোথা থেকে আসে (Major sources):**
  - **Business:** Web, E-commerce, Transactions, Stocks
  - **Science:** Remote sensing, Bioinformatics, Scientific simulation
  - **Society & everyone:** News, Digital cameras, YouTube, Facebook
- Automated data collection tools, database systems, web — সব মিলে আমাদের সমাজ এখন "computerized society"।

> **"We are drowning in data, but starving for knowledge!"** — John Naisbitt
> মানে: আমাদের কাছে data এর পাহাড় আছে, কিন্তু সেখান থেকে কাজের **knowledge** বের করতে পারছি না।

> **"Necessity is the mother of invention"** — এই প্রয়োজন থেকেই **Data Mining** এসেছে: massive data set এর উপর **Automated Pattern Recognition & Pattern Analysis**।

**English summary:** Huge amounts of data are collected automatically, but humans cannot analyze it manually — so we need automated tools (data mining) to turn data into knowledge.

---

## 2. What is Data Mining?

### Definition (মুখস্থ রাখো)

> **Data Mining (Pattern & knowledge discovery from data):** Extraction of **interesting** (**non-trivial, implicit, previously unknown and potentially useful**) patterns or knowledge from **huge amounts of data**.

চারটা keyword এর মানে (descriptive answer এ explain করলে ভালো নম্বর):

| Keyword | বাংলা ব্যাখ্যা |
|---|---|
| **Non-trivial** | সহজ/সরাসরি query করে পাওয়া যায় না — computation লাগে |
| **Implicit** | data এর ভিতরে লুকানো, explicitly লেখা নেই |
| **Previously unknown** | আগে কেউ জানত না — নতুন জ্ঞান |
| **Potentially useful** | কাজে লাগানো যায় (decision নিতে সাহায্য করে) |

### "Data mining" — a Misnomer?
নামটা একটু ভুল (misnomer)। আমরা তো "data" mine করছি না — data থেকে **pattern/knowledge** mine করছি (যেমন gold mining মানে মাটি থেকে সোনা তোলা, "rock mining" না)। তাই ভালো নাম হতো **Pattern Mining / Pattern Discovery**।

**Alternative names:** Knowledge Discovery in Databases (**KDD**), knowledge mining, pattern mining, data/pattern analysis, data dredging, information harvesting, business intelligence।

### সবকিছুই কি Data Mining? (Watch out!)

| Activity | Data Mining? | কেন |
|---|---|---|
| Simple search & query processing (e.g., SQL `SELECT`) | **NO** | শুধু existing data খুঁজে আনা — নতুন pattern বের হয় না |
| Deductive expert systems | **YES** (slide অনুযায়ী) | rules/knowledge ব্যবহার করে নতুন conclusion বের করে |

### Pattern Recognition কী?
**Pattern Recognition** মানে data (image, signal, record) এর মধ্যে regularity/pattern চিনে নিয়ে সেটাকে কোনো category তে ফেলা বা predict করা। Data Mining আর Pattern Recognition একই লক্ষ্যে কাজ করে — data থেকে automatic ভাবে pattern খোঁজা। এই course এ দুটোকে একসাথে "Pattern/Data Mining" বলা হয়েছে।

---

## 3. Data Mining Steps / KDD Process

### (a) Slide অনুযায়ী Pattern/Data Mining Steps

1. **Data Preprocessing** — Data cleaning, missing values handle করা
2. **Data Integration** — একাধিক source থেকে data একসাথে করা
3. **Data Warehousing** — data efficiently database/warehouse এ store করা
4. **Data Exploration** — **EDA** (Exploratory Data Analysis) দিয়ে mining এর জন্য data select করা
5. **Data Mining** — Classification, Clustering, Regression ইত্যাদি algorithm চালানো
6. **Visualization** — mining এর result present করা
7. **Knowledge use** — mined patterns/knowledge ব্যবহার করা বা **knowledge-base / decision support system** এ store করা

### (b) KDD Process — ML & Statistics view (slide 11)

```
Input Data ──► Data Pre-Processing ──► Data Mining ──► Post-Processing ──► Pattern / Information / Knowledge
                                        (Pattern discovery,
                                         Association & Correlation,
                                         Classification, Clustering,
                                         Outlier analysis, ...)
```

| Stage | কী হয় |
|---|---|
| **Pre-processing** | cleaning, integration, normalization, feature selection, dimensionality reduction |
| **Data Mining** | আসল algorithm — pattern discovery, association, classification, clustering, outlier analysis |
| **Post-processing** | pattern evaluation, interesting pattern বাছাই, visualization, interpretation |

> **Classic KDD steps (textbook — Han & Kamber, extra clarity):** Data Cleaning → Data Integration → Data Selection → Data Transformation → **Data Mining** → Pattern Evaluation → Knowledge Presentation।
> খেয়াল করো: **Data Mining হলো KDD process এর একটা step (core step)** — পুরো KDD না।

---

## 4. EDA vs Data Mining: "Explaining the Past" vs "Predicting the Future"

> **Pattern/Data Mining is all about Explaining the Past (EDA) and Predicting the Future (DM).**

| | **EDA (Exploratory Data Analysis)** | **Data Mining** |
|---|---|---|
| লক্ষ্য | **Explaining the Past** — অতীতের data বোঝা | **Predicting the Future** — নতুন data এর জন্য predict |
| Tools | Statistical tools: mean, variance, covariance, correlation, histogram, **PCA** | Classification, Regression, Clustering, Association |
| Example | PCA দিয়ে feature সংখ্যা কমানো | নতুন customer loan default করবে কিনা predict |

---

## 5. Data Mining in the Business Intelligence (BI) Pyramid

নিচ থেকে উপরে গেলে **business decision support করার potential বাড়ে**:

| Level (উপর → নিচ) | কাজ | কে করে |
|---|---|---|
| **Decision Making** | চূড়ান্ত সিদ্ধান্ত | End User (Top Management) |
| **Data Presentation** | Visualization Techniques | Business Analyst |
| **Data Mining** | Information Discovery | Data Analyst |
| **Data Exploration (EDA)** | Statistical summary, querying, reporting | Data Analyst |
| **Data Preprocessing / Integration, Data Warehouses** | data তৈরি ও store | DBA |
| **Data Sources** | Paper, files, web documents, scientific experiments, database systems | — |

**মূল কথা:** Data Mining pyramid এর মাঝামাঝি — raw data কে **information/knowledge** এ রূপান্তর করে, যা উপরের level এ decision নিতে কাজে লাগে।

---

## 6. Data Mining Tasks — Overview

| Task | Type | কী করে (বাংলা) | Example | Algorithm (course) |
|---|---|---|---|---|
| **Classification** | Supervised | নতুন data কে আগে থেকে জানা **class/label** এ ফেলা (output = category) | Loan approve হবে কিনা, tumor cancerous/benign | Decision Tree (ID3), kNN |
| **Regression** | Supervised | **continuous numeric value** predict করা | বাড়ির দাম, তাপমাত্রা | kNN Regression, Linear Regression |
| **Clustering** | Unsupervised | label ছাড়া **similar data কে group** করা | Customer segmentation | K-Means |
| **Association Rule / Frequent Pattern** | Unsupervised | কোন item গুলো **একসাথে ঘটে** তা খোঁজা ($`X \Rightarrow Y`$) | Market basket: Bread → Butter | Apriori, FP-Growth |
| **Anomaly / Outlier Detection** | Mostly unsupervised | সাধারণ pattern থেকে **আলাদা/অস্বাভাবিক** data খোঁজা | Credit card fraud | Outlier analysis |

**Classification vs Regression (এক লাইনে):** Classification → output **discrete class** (Yes/No); Regression → output **continuous number** (e.g., 73.5)।

---

## 7. Supervised vs Unsupervised Learning

**Supervised learning** মানে labeled data থেকে শেখা — প্রতিটা training example এর সাথে answer (**class LABEL**) দেওয়া থাকে। "Supervision" = training data এর সাথে label আছে, যেন একজন শিক্ষক উত্তর বলে দিচ্ছে।
- নতুন data (**TEST data**) কে training set এর ভিত্তিতে classify করা হয়।
- Classification process: **(1) Model Creation, (2) Model Usage**।
- Applications: Credit card / loan approval, Medical diagnosis (tumor cancerous or benign), Fraud detection।

**Unsupervised learning** মানে label ছাড়া শেখা — training data এর class LABEL **unknown**। কিছু measurement/observation দেওয়া থাকে, লক্ষ্য হলো data তে **class বা cluster আছে কিনা** তা খুঁজে বের করা।
- Example: **Clustering** ("learning by observation")।

| Point | **Supervised** | **Unsupervised** |
|---|---|---|
| Label | আছে (labeled data) | নেই (unlabeled) |
| শেখার ধরন | Learning **by examples** | Learning **by observations** |
| লক্ষ্য | নতুন data এর label/value predict | data এর ভিতরের structure/group খোঁজা |
| Example task | Classification, Regression | Clustering, Association rules |
| Algorithm | Decision Tree, kNN | K-Means, Apriori |
| Evaluation | Accuracy, Precision, Recall (সহজে মাপা যায়) | মাপা কঠিন (cluster quality measures) |

---

## 8. Classification Process: Model Construction → Model Usage

### Terminology (Training set হলো একটা **TABLE**)
- **Rows** → Patterns / Instances / Examples / Data points
- **Columns** → Features / Attributes
- **Class LABEL** → যে column টা predict করতে হবে

### Step 1: Model Construction (Training)
Training data → **Classification Algorithm** → **Model (Classifier)**।
Model হতে পারে rules, decision tree, বা formula। Slide এর example:

```
IF rank = 'professor' OR years > 6
THEN tenured = 'yes'
```

### Step 2: Model Usage (Testing / Prediction)
- **Test set** দিয়ে model evaluate করা হয় (accuracy ইত্যাদি)। Test set training এ ব্যবহার হয় না।
- Slide example: test set এ **Accuracy = 75%, Precision = 67%, Recall = 100%** (একটা row misclassified)।
- এরপর unseen data predict: যেমন **(Jeff, Professor, 4) → Tenured?** → rule অনুযায়ী rank = professor তাই **Yes**।

### Decision Tree Classifier (Classifier 1)
**Decision Tree** হলো tree-shaped model: প্রতিটা internal node = একটা attribute test, branch = test এর result, leaf = class label। Root থেকে leaf পর্যন্ত গেলে class পাওয়া যায়।
- Slide example: **buys_computer** dataset (4 features: age, income, student, credit_rating) → **ID3** দিয়ে tree বানানো (root = age)।
- **Tree Induction Algorithms:**
  - **ID3** (Iterative Dichotomiser 3)
  - **C4.5** (ID3 এর successor)
  - **CART** (Classification and Regression Tree)
  - **CHAID** (Chi-square Automatic Interaction Detection)
  - **MARS** (numeric data এর জন্য ভালো)
  - **BOAT** (Bootstrapped Optimistic Algorithm for Tree Construction)
- **সুবিধা:** বোঝা সহজ, IF-THEN rule এ রূপান্তর করা যায় (interpretable)।

---

## 9. k-Nearest Neighbor (kNN) Classifier

### Idea: Instance-Based Learning
> **"Similar patterns/instances have similar LABELs."** — নতুন example কে তার মতো (similar) training example গুলোর মতো করে classify করো।

kNN কোনো আলাদা model বানায় না — training data মনে রাখে, আর test এর সময় তুলনা করে (তাই একে **instance-based / lazy learning** বলে)।

### Algorithm
1. Test example **x** দেওয়া আছে, তার class **y** predict করতে হবে।
2. x এর সাথে সব training example এর **distance/similarity** বের করো (Euclidean distance, বা categorical data হলে **Hamming distance** = কয়টা attribute mismatch)।
3. সবচেয়ে কাছের (**most similar**) **k** টা training example নাও।
4. তাদের মধ্যে **MAJORITY** class কে x এর label দাও।

### 1-NN (k = 1)
সবচেয়ে simple classifier — নতুন point কে **সবচেয়ে কাছের একটা** known point এর label দাও। কিন্তু noise এর প্রতি sensitive।

### k-NN (k > 1)
1-NN কে generalize করে training set এর **noise smooth** করে। নতুন point পায় তার k nearest neighbor এর **most frequent label**।

Slide example (Red vs Green):
| k | Neighbors | Prediction |
|---|---|---|
| k = 3 | 2 Red, 1 Green | **Red** |
| k = 7 | 3 Red, 4 Green | **Green** |

→ দেখো, **k এর মান বদলালে result বদলে যেতে পারে।**

### k কেমন হবে?
- **খুব ছোট k** → noise এ sensitive (overfit)।
- **বড় k** → noise এর বিরুদ্ধে **robust**, কিন্তু অনেক বড় হলে দূরের (অন্য class এর) point ও vote দেয়।
- তাই k **not too large, not too small** — dataset এর উপর নির্ভর করে।
- **k সবসময় ODD (1, 3, 5...) নাও, EVEN (যেমন 2) না** — কারণ even k তে **tie** হতে পারে।
  - Slide example: k = 2 তে একটা test example এর দুই neighbor → একজন "yes", একজন "no" → **Yes/No tie** — সিদ্ধান্ত নেওয়া যায় না।

### Advantages / Disadvantages
| ✔ Advantages | ✘ Disadvantages |
|---|---|
| Simple, বোঝা সহজ | Test time এ slow (সব training point এর সাথে distance) |
| খুব কম tuning (শুধু k) | Feature scale এর উপর নির্ভরশীল → **normalization দরকার** |
| Training phase নেই | Irrelevant features এ খারাপ করে |

---

## 10. Regression and kNN Regression

**Regression** মানে numeric (continuous) value predict করা। Regression analysis এ একটা **dependent variable** (response) কে এক বা একাধিক **independent variable** (predictor) দিয়ে model করা হয়; "best fit" সাধারণত **least squares method** দিয়ে বের করা হয়।
- Linear regression: $`Y = b_0 + b_1 X`$
- Multiple linear regression: $`Y = b_0 + b_1 X_1 + b_2 X_2`$
- ব্যবহার: prediction (time-series forecasting সহ), inference, hypothesis testing, causal relationship modeling।

### kNN Regression (k = 3 example in slides)
Classification এর মতোই k nearest neighbor খোঁজো, কিন্তু **majority vote** এর বদলে neighbor দের target value এর **average (mean)** নাও:

```math
\hat{y} = \frac{1}{k}\sum_{i=1}^{k} y_i
```

উদাহরণ: k = 3 nearest বাড়ির দাম 50, 60, 70 লাখ → predicted দাম = 60 লাখ।

> **Note (slide 36):** kNN regression এ **Feature Normalization** দরকার — নাহলে বড় range এর feature (যেমন income) distance কে dominate করবে।

---

## 11. Classifier Evaluation Metrics

### Data Split
শুধু training set দিয়ে accuracy মাপলে ভুল (overly optimistic) হয়। তাই **Training + Validation + Test** set:
- **60% + 20% + 20%** বা **80% + 10% + 10%**

### Confusion Matrix

| Actual \ Predicted | **C1 (Positive)** | **¬C1 (Negative)** | Total |
|---|---|---|---|
| **C1** | **TP** (True Positive) | **FN** (False Negative) | AP |
| **¬C1** | **FP** (False Positive) | **TN** (True Negative) | AN |
| Total | PP | PN | All |

- **TP:** positive ছিল, positive বলেছে ✔
- **TN:** negative ছিল, negative বলেছে ✔
- **FP:** negative ছিল, কিন্তু positive বলেছে ✘ (false alarm)
- **FN:** positive ছিল, কিন্তু negative বলেছে ✘ (miss)
- AP/AN = Actual Positive/Negative; PP/PN = Predicted Positive/Negative

### Formulas

| Metric | Formula | মানে |
|---|---|---|
| **Accuracy** (recognition rate) | $`\dfrac{TP + TN}{All}`$ | মোট কত % সঠিক classify হয়েছে |
| **Error rate** | $`1 - \text{Accuracy} = \dfrac{FP + FN}{All}`$ | কত % ভুল |
| **Sensitivity** (= Recall, TP rate) | $`\dfrac{TP}{AP}`$ | actual positive এর কত % ধরা পড়েছে |
| **Specificity** (TN rate) | $`\dfrac{TN}{AN}`$ | actual negative এর কত % ঠিক negative বলেছে |
| **Precision** (Exactness) | $`\dfrac{TP}{PP} = \dfrac{TP}{TP+FP}`$ | যাদের positive বলেছি তাদের কত % আসলেই positive |
| **Recall** (Completeness) | $`\dfrac{TP}{AP} = \dfrac{TP}{TP+FN}`$ | আসল positive দের কত % positive বলতে পেরেছি |
| **F1 / F-score** | $`\dfrac{2 \times P \times R}{P + R}`$ | Precision ও Recall এর **harmonic mean** |
| **$`F_\beta`$** | $`\dfrac{(1+\beta^2)\, P \times R}{\beta^2 P + R}`$ | Recall কে precision এর চেয়ে **β গুণ** বেশি weight দেয় |

- Perfect score = **1.0**।
- **Precision ও Recall এর মধ্যে inverse relationship** — একটা বাড়ালে অন্যটা সাধারণত কমে। তাই দুটোকে balance করতে **F1** ব্যবহার হয়।

### Worked Example (slide 20: buys_computer)

| Actual \ Predicted | yes | no | Total |
|---|---|---|---|
| **yes** | 6954 (TP) | 46 (FN) | 7000 (AP) |
| **no** | 412 (FP) | 2588 (TN) | 3000 (AN) |
| Total | 7366 (PP) | 2634 (PN) | 10000 |

- Accuracy $`= (6954 + 2588)/10000 = 95.42\%`$
- Error rate $`= (412 + 46)/10000 = 4.58\%`$
- Precision $`= 6954/7366 = 94.41\%`$
- Recall / Sensitivity $`= 6954/7000 = 99.34\%`$
- Specificity $`= 2588/3000 = 86.27\%`$
- F1 $`= 2(0.9441)(0.9934)/(0.9441 + 0.9934) \approx 0.968`$

### Class Imbalance Problem (descriptive এ আসতে পারে!)
যখন একটা class খুব **rare** (যেমন fraud, HIV-positive) — negative class majority, positive class minority। তখন **accuracy misleading**।
- উদাহরণ: 1000 transaction এর মধ্যে 10টা fraud। Model যদি সবসময় "not fraud" বলে → accuracy 99%! কিন্তু একটা fraud ও ধরেনি (Recall = 0)।
- তাই imbalanced data তে **Precision, Recall (Sensitivity), Specificity, F1** দেখতে হয়।

**কখন কোনটা গুরুত্বপূর্ণ?**
- **Recall** বেশি দরকার → cancer diagnosis, fraud detection (positive miss করা বিপজ্জনক)।
- **Precision** বেশি দরকার → spam filter (ভালো mail কে spam বলা খারাপ)।

---

## 12. Clustering (Cluster Analysis)

### Definition
- **Cluster:** data object এর একটা collection যারা
  - একই group এর মধ্যে **similar** (related)
  - অন্য group এর object থেকে **dissimilar** (unrelated)
- **Cluster analysis (clustering, data segmentation):** data এর characteristics অনুযায়ী data point গুলোর similarity বের করে **similar object গুলোকে একই cluster এ group করা**।
- **Unsupervised learning:** কোনো predefined class label নেই (learning by observation)।

**Good clustering:** high **intra-cluster** similarity (ভিতরে কাছাকাছি), low **inter-cluster** similarity (cluster গুলো একে অপর থেকে দূরে)।

**Applications:** Marketing, City planning, Information retrieval, Document clustering, Biology & Bioinformatics, Climate, Geological & Oceanic studies, Health & Drug studies, Economics & Sociology।

### K-Means Algorithm (Partitioning method)
1. Objects কে **k** টা non-empty subset এ ভাগ করো (random assignment / random centroids)।
2. **Repeat:**
   - প্রতিটা partition এর **centroid (mean point)** compute করো।
   - প্রতিটা object কে তার **nearest centroid** এর cluster এ assign করো।
3. **Until no change** (assignment আর বদলায় না)।

(Slide example: K = 2 → arbitrarily partition → update centroids → reassign → loop।)

### Clustering Algorithm Categories

| Category | Algorithms |
|---|---|
| **Partitioning** | K-Means, K-Medoids |
| **Hierarchical** | BIRCH, Chameleon (+ probabilistic hierarchical) |
| **Density-based** | DBSCAN, OPTICS, DENCLU |
| **Grid-based** | STING, CLIQUE (CLIQUE = subspace clustering ও) |
| **Model-based** | EM (statistical distributions), SOM |

### Cluster Models (short)
- **Connectivity** (hierarchical — distance connectivity), **Centroid** (k-means — mean vector), **Distribution** (multivariate normal, EM algorithm), **Density** (DBSCAN, OPTICS — dense region), **Subspace** (biclustering), **Group**, **Graph-based** (clique), **Signed graph**, **Neural** (Self-Organizing Map)।

### Types of Clustering
| Type | মানে |
|---|---|
| **Hard clustering** | প্রতিটা object হয় একটা cluster এ আছে, নয় নেই |
| **Soft (fuzzy) clustering** | প্রতিটা object প্রতিটা cluster এ **কিছু degree** তে আছে (likelihood) |
| Strict partitioning | প্রতিটা object ঠিক **একটা** cluster এ |
| Strict partitioning with outliers | কোনো cluster এ না থাকলে → **outlier** |
| Overlapping | একটা object একাধিক cluster এ থাকতে পারে |
| Hierarchical | child cluster এর object parent cluster এও আছে |
| Subspace | uniquely defined subspace এর ভিতরে overlap হয় না |

### Classification vs Clustering

| | Classification | Clustering |
|---|---|---|
| Learning | Supervised | Unsupervised |
| Label | আগে থেকে জানা | জানা নেই |
| Output | নতুন data এর class | data এর groups |
| Example | Decision Tree, kNN | K-Means |

---

## 13. Frequent Pattern Analysis & Association Rules

### Frequent Pattern
**Frequent pattern** হলো এমন pattern (item set, subsequence, substructure) যা data set এ **বারবার (frequently)** ঘটে।
- প্রথম propose করেন **Agrawal, Imielinski, and Swami** — frequent itemsets ও association rule mining এর context এ।
- **Motivation:** data এর **inherent regularities** খোঁজা —
  - কোন products একসাথে কেনা হয়? (Bread & Butter?)
  - Laptop কেনার পর পরবর্তী purchase কী?
  - কোন ধরনের DNA নতুন drug এ sensitive?
  - Web documents automatically classify করা যায়?
- **Applications:** Market basket analysis, cross-marketing, catalog design, sale campaign analysis, Web log (click stream) analysis, DNA sequence analysis।

### Basic Concepts
- **Itemset:** এক বা একাধিক item এর set। **k-itemset** $`X = \{x_1, \dots, x_k\}`$
- **Absolute support (support count):** itemset X কতবার ঘটেছে।
- **Relative support (s):** কত fraction transaction এ X আছে = P(transaction contains X)।
- **Frequent itemset:** X এর support ≥ **minsup** threshold।

### Association Rule $`X \Rightarrow Y`$
- **Support** $`s = P(X \cup Y)`$ = $`\dfrac{\text{# transactions containing both } X \text{ and } Y}{\text{total transactions}}`$
- **Confidence** $`c = P(Y \mid X)`$ = $`\dfrac{\text{support}(X \cup Y)}{\text{support}(X)}`$
- লক্ষ্য: **minimum support** এবং **minimum confidence** পূরণ করে এমন সব rule বের করা।

### Worked Example (slide 31–32)

| Tid | Items bought |
|---|---|
| 10 | Beer, Nuts, Diaper |
| 20 | Beer, Coffee, Diaper |
| 30 | Beer, Diaper, Eggs |
| 40 | Nuts, Eggs, Milk |
| 50 | Nuts, Coffee, Diaper, Eggs, Milk |

minsup = 50%, minconf = 50% (অর্থাৎ ≥ 3 টা transaction এ থাকতে হবে, 5 এর 50% = 2.5)
- **Frequent patterns:** Beer:3, Nuts:3, Diaper:4, Eggs:3, {Beer, Diaper}:3
- **Beer ⇒ Diaper (60%, 100%)** → support = 3/5 = 60%, confidence = 3/3 = 100%
- **Diaper ⇒ Beer (60%, 75%)** → support = 3/5 = 60%, confidence = 3/4 = 75%

> লক্ষ্য করো: **support দুই দিকে একই**, কিন্তু **confidence আলাদা** (কারণ denominator আলাদা)।

### Mining Methods
- **Apriori** — *Candidate Generation-and-Test approach*: আগে frequent itemsets বের করো → তারপর association rules → প্রতিটা rule এর confidence compute করো। (Apriori principle: frequent itemset এর সব subset ও frequent।)
- **FP-Growth, FP-Growth+** — Frequent Pattern-Growth approach (candidate generation ছাড়া)
- **ECLAT** — vertical data format
- **MaxMiner** — closed frequent patterns & max-patterns; **CLOSET, FPClose, FPMax**
- Graph patterns: **gSpan, CloseGraph**; Clustering: **MaPle**; Classification: **PGClass**

---

## 14. Anomaly / Outlier Detection

**Outlier (anomaly)** হলো এমন data object যা বাকি data এর general behavior থেকে অনেক আলাদা। KDD process এ **Outlier analysis** একটা data mining task (slide 11)।
- **Data cleaning** এ outlier identify/remove করা হয় (noise হিসেবে) — যেমন Salary = "−10"।
- কিন্তু কখনো outlier ই আসল interest — যেমন **fraud detection** (অস্বাভাবিক credit card transaction), network intrusion, medical anomaly।
- Clustering এর সাথে সম্পর্ক: "**Strict partitioning with outliers**" — যে object কোনো cluster এ পড়ে না সেটা outlier। Density-based (DBSCAN) method এ low-density point outlier হয়।

**English summary:** Anomaly detection finds rare objects that deviate from the normal pattern; they can be noise (remove) or the most valuable finding (fraud).

---

## 15. Data Preprocessing

### কেন দরকার? — "Data in the real world is dirty"
Instrument faulty, human/computer error, transmission error এর কারণে data খারাপ হয়:

| Problem | মানে | Example |
|---|---|---|
| **Incomplete** | attribute value missing | Occupation = " " |
| **Noisy** | noise, error, outlier | Salary = "−10", Age = "7" (আসলে 70–79) |
| **Inconsistent** | code/name এ অমিল | Age = "42" কিন্তু Birthday = "03/07/2010"; rating আগে "1,2,3" এখন "A,B,C"; duplicate record এ অমিল |
| **Intentional** (disguised missing) | ইচ্ছাকৃত ভুল value | সবার birthday Jan 1 বা Dec 30 |

> "Garbage in, garbage out" — খারাপ data দিলে mining result ও খারাপ হবে।

### Major Tasks of Data Preprocessing

| Task | কী করে |
|---|---|
| **1. Data Cleaning** | missing values fill, noisy data smooth, outlier identify/remove, inconsistency resolve |
| **2. Data Integration** | একাধিক database, data cube, file একসাথে করা |
| **3. Data Reduction** | Dimensionality reduction (**PCA, LDA**), Numerosity reduction (**sampling**, regression, histogram, clustering), Data compression |
| **4. Data Transformation & Discretization** | **Normalization** (min-max, z-score), concept hierarchy generation, smoothing, aggregation, feature construction |

### Handling Missing Data
1. **Ignore the tuple** — সাধারণত যখন class label missing; কিন্তু attribute প্রতি missing % অনেক বেশি হলে effective না।
2. **Manually fill** — tedious, অনেক সময় infeasible।
3. **Automatically fill:**
   - **Global constant** (যেমন "unknown" — কিন্তু এটা নতুন class হয়ে যেতে পারে!)
   - Attribute এর **mean**
   - **Same class** এর sample গুলোর attribute mean (smarter)
   - **Most probable value** — inference-based (Bayesian formula, Decision tree)
   - Numeric হলে **interpolation** (e.g., Lagrange formula)

### Data Reduction techniques
- **Regression:** data কে একটা equation ($`Y = b_0 + b_1X`$) দিয়ে represent — শুধু parameter store।
- **Histogram:** data কে **buckets** এ ভাগ করে প্রতিটার average/sum store। Partitioning: **Equal-width** (সমান range) বা **Equal-frequency/equal-depth** (প্রতিটায় সমান সংখ্যক data)।
- **Clustering:** শুধু cluster representation (centroid, diameter) store — data clustered হলে effective, "smeared" হলে না।
- **Sampling:** বড় data set N কে represent করতে ছোট sample s নেওয়া → algorithm sub-linear complexity তে চলে। Key: **representative subset**।
  - **SRSWOR** (without replacement): একবার নির্বাচিত হলে population থেকে বাদ — একই object একাধিকবার আসে না।
  - **SRSWR** (with replacement): নির্বাচিত object ফেরত যায় — একাধিকবার আসতে পারে।
  - **Stratified sampling:** data কে partition (strata) করে প্রতিটা থেকে **proportionally** sample — **skewed data** এর জন্য (simple random sampling skew এ খারাপ করে)।
  - **Cluster sampling:** cluster বানিয়ে cluster থেকে sample।

### Normalization (Data Transformation)

**Min-max normalization** — $`[new\_min, new\_max]`$ range এ আনা:

```math
v' = \frac{v - min_A}{max_A - min_A}(new\_max_A - new\_min_A) + new\_min_A
```

Example: income range \$12,000 – \$98,000 → [0.0, 1.0]। \$73,000 → $`\frac{73000-12000}{98000-12000} \approx 0.709`$ (slide এ textbook অনুযায়ী 0.716 লেখা — textbook এ value \$73,600 ধরা হয়েছে, তখন ≈ 0.716)।

**Z-score normalization** ($`\mu`$ = mean, $`\sigma`$ = standard deviation):

```math
v' = \frac{v - \mu_A}{\sigma_A}
```

Example: $`\mu = 54{,}000`$, $`\sigma = 16{,}000`$ → \$73,600 → $`\frac{73600 - 54000}{16000} = 1.225`$

**কেন normalization?** kNN, clustering এর মতো **distance-based** method এ বড় range এর attribute distance কে dominate করে; normalization সব attribute কে সমান গুরুত্ব দেয়।

---

## 16. EDA: PCA, Covariance, Correlation

### Principal Component Analysis (PCA)
- **Goal:** এমন একটা **projection** খোঁজা যা data এর **largest amount of variation (variance)** ধরে রাখে।
- n-dimensional N টা data vector থেকে **k ≤ n** টা **orthogonal vector (principal components)** খোঁজা যা data কে সবচেয়ে ভালো represent করে।
- Covariance matrix এর **eigenvectors** নতুন space define করে।
- ফলাফল: **dimensionality reduction** (feature সংখ্যা কমে)।
- Slide example: 2-D point $`(x_1, x_2)`$ → maximum variance vector **e** এর উপর project → শুধু **1-D (z)** দিয়ে represent।
- (পিছনের math: covariance matrix, eigenvalues, eigenvectors, SVD, projection)

### Covariance (numeric data)

```math
Cov(A,B) = E[(A-\bar{A})(B-\bar{B})] = \frac{\sum_{i=1}^{n}(a_i - \bar{A})(b_i - \bar{B})}{n}
```

Simplified: $`Cov(A,B) = E(A \cdot B) - \bar{A}\,\bar{B}`$

| Value | মানে |
|---|---|
| $`Cov > 0`$ | A ও B একসাথে বাড়ে (positive) |
| $`Cov < 0`$ | A বাড়লে B কমে (negative) |
| $`Cov = 0`$ | independent হলে Cov = 0, **কিন্তু উল্টোটা সত্য না** (Cov = 0 মানেই independent না; শুধু multivariate normal এর মতো assumption থাকলে) |

**Example (slide 49):** Stock A, B: (2,5), (3,8), (5,10), (4,11), (6,14)
- $`E(A) = 20/5 = 4`$, $`E(B) = 48/5 = 9.6`$
- $`E(A \cdot B) = (10 + 24 + 50 + 44 + 84)/5 = 212/5 = 42.4`$
- $`Cov(A,B) = 42.4 - 4 \times 9.6 = 4 > 0`$ → **A ও B একসাথে rise করে।**

### Correlation Coefficient (Pearson's)

```math
r_{A,B} = \frac{\sum_{i=1}^{n}(a_i - \bar{A})(b_i - \bar{B})}{n\,\sigma_A \sigma_B} = \frac{Cov(A,B)}{\sigma_A \sigma_B}
```

- সবসময় **[−1, 1]** এর মধ্যে।
- $`r > 0`$: positively correlated (যত বেশি, তত strong); $`r = 0`$: uncorrelated (slide: independent); $`r < 0`$: negatively correlated।
- Scatter plot দিয়ে visually correlation দেখা যায় (−1 থেকে 1)।
- **Covariance vs Correlation:** Correlation হলো **normalized covariance** — scale-independent, তাই তুলনা করা সহজ।
- **Non-numeric (nominal/ordinal) data** হলে → **$`\chi^2`$ (chi-square) test**।

---

## 17. Ubiquitous & Invisible Data Mining, Privacy & Security

### Ubiquitous Data Mining
Data mining **সব জায়গায়** ব্যবহার হয় — online shopping, **CRM (Customer Relationship Management)**।

### Invisible Data Mining
Data mining function দৈনন্দিন কাজের ভিতরে **built-in** — user বুঝতেই পারে না।
- Example: **Google search** — user জানে না যে সে data mining এর result দেখছে।
- Invisible DM highly desirable, **কিন্তু Privacy ও Security issue** তৈরি করে।

### Privacy Concern
- **আসল privacy concern:** individual record এ **unconstrained access**, বিশেষ করে privacy-sensitive information।
- অনেক DM application personal data ছোঁয়ই না — meteorology, astronomy, geography, geology, biology, scientific/engineering data।
- অনেক DM study scalable algorithm দিয়ে **general / statistically significant pattern** খোঁজে, individual কে না।

### Privacy/Security রক্ষার Methods
1. **Removing sensitive IDs** — data থেকে sensitive identifier (নাম, NID) বাদ দেওয়া।
2. **Data security-enhancing methods:**
   - **Multi-level security model** — শুধু authorized level এর access।
   - **Encryption** — blind signatures, biometric encryption, anonymous databases (personal info encrypt করে বিভিন্ন জায়গায় store)।
3. **Privacy-preserving data mining methods** (নিচে)।

### Privacy-Preserving Data Mining (PPDM)
> **Definition:** underlying **sensitive data values disclose না করে** valid mining result পাওয়া। প্রায়ই **information loss ও privacy** এর মধ্যে **trade-off** লাগে।

| Method | কী করে |
|---|---|
| **Randomization (perturbation)** | data তে **noise add** করে attribute value mask করা |
| **k-anonymity** | record এমনভাবে alter করা যাতে প্রতিটা record **অন্তত k টা** অন্য record এর সাথে মিলে যায় → uniquely identify করা যায় না |
| **l-diversity** | একই group এর ভিতরে sensitive value এর **diversity** নিশ্চিত করা (k-anonymity র দুর্বলতা দূর করে) |
| **Distributed privacy preservation** | data **horizontally, vertically** বা দুইভাবে partition করে বিভিন্ন site এ রাখা |
| **Downgrading effectiveness of DM** | DM এর **output** ও privacy violate করতে পারে → কিছু association rule hide করা বা classification model সামান্য distort করা |

### Social Impacts
DM এর সুফল (better service, fraud detection, science) আছে, আবার অপব্যবহার হলে privacy হারানো, discrimination, surveillance এর ঝুঁকি আছে — তাই responsible use দরকার।

---

## 18. Applications (Summary)

| Domain | Application | Task |
|---|---|---|
| Banking/Finance | Credit card / loan approval | Classification |
| Banking | Fraud detection | Classification / Anomaly detection |
| Medicine | Tumor cancerous or benign; drug-sensitive DNA | Classification / Frequent patterns |
| Retail | Market basket analysis, cross-marketing, catalog design | Association rules |
| Marketing | Customer segmentation, CRM | Clustering |
| Web | Search (Google), click-stream analysis, document clustering | Invisible DM, clustering, association |
| Science | Remote sensing, bioinformatics, climate, geology | Clustering, regression |
| Real estate/Economy | Price prediction, stock trend | Regression |
| City planning | Area grouping | Clustering |

---

## 19. Likely Descriptive Questions + Model Answers

### Q1. What is Data Mining? Why is it needed? / Data Mining কী? কেন দরকার?
**Answer:** Data mining is the extraction of interesting — **non-trivial, implicit, previously unknown and potentially useful** — patterns or knowledge from huge amounts of data. It is also called Knowledge Discovery in Databases (KDD).
বাংলায়: বিশাল data থেকে লুকানো, নতুন ও কাজের pattern automatic ভাবে বের করাই data mining।
**কেন দরকার:** Business (e-commerce, stocks), science (bioinformatics, remote sensing) আর society (YouTube, Facebook) থেকে Terabyte–Exabyte data জমছে। মানুষ manually এগুলো analyze করতে পারে না — "We are drowning in data, but starving for knowledge"। তাই automated pattern analysis দরকার।
**Not DM:** simple search/query processing। **DM:** deductive expert systems।

### Q2. Why is "data mining" called a misnomer?
**Answer:** কারণ আমরা data কে mine করি না, data **থেকে knowledge/pattern** mine করি (gold mining এ যেমন মাটি থেকে সোনা)। তাই সঠিক নাম হওয়া উচিত "knowledge mining from data" / **pattern mining / pattern discovery**। Other names: KDD, knowledge mining, data dredging, information harvesting, business intelligence।

### Q3. Describe the steps of the KDD / data mining process.
**Answer:**
1. **Data preprocessing/cleaning** — noise, missing values দূর করা
2. **Data integration** — multiple source একত্রিত করা
3. **Data warehousing / selection** — data store ও relevant data select (EDA দিয়ে)
4. **Data transformation** — normalization, reduction (PCA)
5. **Data mining** — classification, clustering, regression, association, outlier analysis
6. **Pattern evaluation / post-processing** — interesting pattern বাছাই
7. **Visualization & knowledge presentation** — result present করে knowledge-base / decision support system এ ব্যবহার

ML/Statistics view: *Input Data → Pre-processing → Data Mining → Post-processing → Pattern/Information/Knowledge*। Data mining হলো KDD এর core step।

### Q4. Differentiate supervised and unsupervised learning with examples.
**Answer:** Supervised learning এ training data এর সাথে **class label** থাকে; model training set থেকে শিখে নতুন test data classify করে (learning by examples)। Example: classification — loan approval, tumor cancerous/benign, fraud detection; algorithm: Decision Tree, kNN।
Unsupervised learning এ class label **unknown**; লক্ষ্য data তে class/cluster আছে কিনা খুঁজে বের করা (learning by observation)। Example: clustering (K-Means) — customer segmentation; association rules।
(পাশে comparison table দাও — Section 7।)

### Q5. Explain the two-step classification process.
**Answer:** (1) **Model construction:** labeled training set (rows = instances, columns = features, একটা class label column) থেকে classification algorithm একটা model বানায় — যেমন rule *IF rank = 'professor' OR years > 6 THEN tenured = 'yes'*, বা decision tree। (2) **Model usage:** আলাদা **test set** দিয়ে model এর accuracy/precision/recall মাপা হয় (test set training এ use হয় না); acceptable হলে unseen data predict — (Jeff, Professor, 4) → Tenured = Yes।

### Q6. Explain the kNN classifier. Why should k be odd?
**Answer:** kNN হলো **instance-based learning** — "similar instances have similar labels"। Test example x এর জন্য training set থেকে সবচেয়ে similar (nearest, e.g., Euclidean/Hamming distance) **k** টা example নিয়ে তাদের **majority class** x কে দেওয়া হয়। k = 1 হলে 1-NN (closest point এর label)। বড় k noise smooth করে (robust), কিন্তু খুব বড় হলে অন্য class এর দূরের point ও vote দেয় — তাই k not too large, not too small। Example: k = 3 → 2 Red, 1 Green → Red; k = 7 → 3 Red, 4 Green → Green।
**Odd k:** even k (যেমন k = 2) তে 1 "yes" ও 1 "no" হলে **tie** হয়, সিদ্ধান্ত নেওয়া যায় না; odd k (binary class এ) tie এড়ায়।
Advantages: simple, little tuning। Need: feature normalization।

### Q7. How does kNN regression differ from kNN classification?
**Answer:** দুটোতেই k nearest neighbor খোঁজা হয়। Classification এ neighbor দের **majority vote** (discrete class), regression এ neighbor দের target value এর **average** (continuous value): $`\hat{y} = \frac{1}{k}\sum y_i`$। দুটোতেই feature normalization গুরুত্বপূর্ণ।

### Q8. Define accuracy, precision, recall and F1. Why is accuracy not enough?
**Answer:**
- Accuracy $`= (TP+TN)/All`$ — মোট কত % সঠিক।
- Precision $`= TP/(TP+FP)`$ — **exactness**: positive predict করাদের কত % আসলেই positive।
- Recall (sensitivity) $`= TP/(TP+FN)`$ — **completeness**: actual positive দের কত % ধরা পড়েছে।
- F1 $`= 2PR/(P+R)`$ — precision ও recall এর harmonic mean।
**Accuracy যথেষ্ট না** কারণ **class imbalance problem**: fraud বা HIV-positive এর মতো rare class এ সবকিছু negative বললেও accuracy খুব বেশি আসে, অথচ একটা positive ও ধরা পড়ে না। তাই precision, recall, specificity, F1 দরকার। Precision ও recall এর inverse relationship থাকায় F1 দিয়ে balance করা হয়।
(Example দাও: 6954/46/412/2588 → Acc 95.42%, P 94.41%, R 99.34%।)

### Q9. Why do we split data into training, validation and test sets?
**Answer:** Training data দিয়েই accuracy মাপলে model "মুখস্থ" করে ফেলায় accuracy অতিরিক্ত ভালো দেখায় (overfitting ধরা পড়ে না)। **Training set** দিয়ে model বানানো, **validation set** দিয়ে parameter (যেমন kNN এর k) tune/model select, আর **test set** দিয়ে unseen data এ final performance মাপা হয়। Typical split: 60/20/20 বা 80/10/10।

### Q10. What is clustering? Explain the K-Means algorithm.
**Answer:** Clustering হলো unsupervised learning যেখানে data point গুলোর similarity অনুযায়ী similar object কে একই cluster এ এবং dissimilar object কে আলাদা cluster এ রাখা হয় (high intra-cluster, low inter-cluster similarity)।
**K-Means:** (1) objects কে k টা non-empty subset এ ভাগ করো; (2) প্রতিটা cluster এর centroid (mean) বের করো; (3) প্রতিটা object কে nearest centroid এর cluster এ reassign করো; (4) কোনো change না হওয়া পর্যন্ত 2–3 repeat।
Applications: marketing, city planning, document clustering, bioinformatics, climate।
Categories: partitioning (K-means, K-medoids), hierarchical (BIRCH, Chameleon), density-based (DBSCAN, OPTICS, DENCLU), grid-based (STING, CLIQUE), model-based।

### Q11. Differentiate hard and soft (fuzzy) clustering.
**Answer:** Hard clustering এ প্রতিটা object একটা cluster এ থাকে অথবা থাকে না (yes/no)। Soft/fuzzy clustering এ প্রতিটা object প্রতিটা cluster এ কোনো একটা **degree/likelihood** নিয়ে থাকে (যেমন 0.7 cluster A, 0.3 cluster B)।

### Q12. What is frequent pattern analysis? Define support and confidence with an example.
**Answer:** Frequent pattern হলো data set এ বারবার ঘটা itemset/subsequence/substructure (Agrawal, Imielinski & Swami প্রথম propose করেন)। এর লক্ষ্য data এর inherent regularity খোঁজা — যেমন কোন products একসাথে কেনা হয় (market basket analysis)।
Rule $`X \Rightarrow Y`$ এর জন্য — **Support** $`= P(X \cup Y)`$ (X ও Y দুটোই আছে এমন transaction এর fraction), **Confidence** $`= P(Y|X) = sup(X \cup Y)/sup(X)`$।
Example (5 transactions): Beer 3 বার, Diaper 4 বার, {Beer, Diaper} 3 বার → Beer ⇒ Diaper (support 60%, confidence 100%); Diaper ⇒ Beer (60%, 75%)। minsup = minconf = 50% হলে দুটোই strong rule।
Algorithm: **Apriori** (candidate generation-and-test), FP-Growth, ECLAT।

### Q13. What is data preprocessing? Why is it needed? Describe its major tasks.
**Answer:** Real-world data **dirty** — incomplete (Occupation = ""), noisy (Salary = −10), inconsistent (Age = 42 কিন্তু Birthday 2010), intentional/disguised missing (সবার birthday Jan 1)। খারাপ data দিলে mining result ও খারাপ হয়, তাই preprocessing দরকার।
Major tasks: (1) **Data cleaning** — missing value fill, noise smooth, outlier remove, inconsistency resolve; (2) **Data integration** — multiple DB/files merge; (3) **Data reduction** — dimensionality (PCA, LDA), numerosity (sampling, histogram, clustering, regression), compression; (4) **Data transformation & discretization** — normalization (min-max, z-score), concept hierarchy।

### Q14. How can missing values be handled?
**Answer:** (i) tuple ignore করা (class label missing হলে); (ii) manually fill (tedious); (iii) global constant ("unknown"); (iv) attribute mean; (v) same class এর attribute mean (smarter); (vi) most probable value — Bayesian/decision tree inference; (vii) numeric হলে interpolation।

### Q15. Explain sampling. Differentiate SRSWOR, SRSWR and stratified sampling.
**Answer:** Sampling মানে বড় data set N কে represent করতে ছোট representative subset s নেওয়া, যাতে mining algorithm দ্রুত চলে। **SRSWOR**: নির্বাচিত object population থেকে বাদ যায়, একাধিকবার আসে না। **SRSWR**: নির্বাচিত object ফেরত যায়, একাধিকবার আসতে পারে। **Stratified**: data কে partition করে প্রতিটা থেকে proportional sample — skewed data তে কাজে লাগে (simple random sampling skew এ খারাপ করে)।

### Q16. What is normalization? Why is it important?
**Answer:** Attribute value গুলোকে একটা common scale এ আনা। **Min-max:** $`v' = \frac{v-min}{max-min}(new\_max - new\_min) + new\_min`$; **Z-score:** $`v' = (v-\mu)/\sigma`$ (e.g., μ = 54,000, σ = 16,000 → 73,600 ↦ 1.225)। Distance-based method (kNN, K-means) এ বড় range এর attribute (income) ছোট range এর attribute (age) কে dominate করে — normalization এটা ঠেকায়।

### Q17. What is PCA? / EDA কী?
**Answer:** **EDA** (Exploratory Data Analysis) statistical tools দিয়ে data বোঝে — "explaining the past"; data mining "predicting the future"। **PCA** একটা EDA/dimensionality reduction technique: covariance matrix এর eigenvector ব্যবহার করে k ≤ n টা orthogonal **principal component** খোঁজে যারা data এর **maximum variance** ধরে রাখে; data কে এই ছোট space এ project করে feature সংখ্যা কমায় (e.g., 2-D → 1-D)।

### Q18. Explain covariance and correlation. Is zero covariance always independence?
**Answer:** Covariance দুই numeric attribute একসাথে কীভাবে বদলায় তা মাপে: Cov > 0 → একসাথে বাড়ে; Cov < 0 → একটা বাড়লে অন্যটা কমে। Correlation coefficient $`r = Cov/(\sigma_A\sigma_B)`$, সবসময় [−1, 1] এর মধ্যে — normalized, তাই strength তুলনা করা যায়। Independent হলে Cov = 0, কিন্তু **Cov = 0 মানেই independent না** (শুধু multivariate normal এর মতো extra assumption এ)। Nominal data র জন্য $`\chi^2`$ test।
Example: stock (2,5),(3,8),(5,10),(4,11),(6,14) → Cov = 42.4 − 4×9.6 = 4 > 0 → একসাথে rise।

### Q19. What is ubiquitous and invisible data mining?
**Answer:** **Ubiquitous DM:** data mining সব জায়গায় ব্যবহৃত — online shopping, CRM। **Invisible DM:** DM function দৈনন্দিন operation এ built-in, user টেরই পায় না — যেমন Google search result। Invisible DM desirable কারণ user কে কিছু শিখতে হয় না, কিন্তু এটা **privacy ও security** concern তৈরি করে।

### Q20. Discuss privacy and security issues in data mining. What is privacy-preserving data mining?
**Answer:** আসল privacy concern হলো individual record এ **unconstrained access**, বিশেষ করে sensitive information। (অনেক DM application — meteorology, astronomy, biology — personal data ছোঁয় না, আর অনেক study general pattern খোঁজে, individual না।)
Protection methods: (1) sensitive IDs remove; (2) security-enhancing — multi-level security model, encryption (blind signatures, biometric encryption, anonymous databases); (3) **privacy-preserving data mining**।
**PPDM** = sensitive data value disclose না করে valid mining result পাওয়া; information loss ও privacy এর trade-off থাকে। Methods: **randomization/perturbation** (noise add), **k-anonymity** (প্রতিটা record অন্তত k টা record এর সাথে indistinguishable), **l-diversity** (group এর ভিতরে sensitive value diverse), **distributed privacy preservation** (horizontal/vertical partition), **downgrading DM effectiveness** (কিছু association rule hide বা model সামান্য distort)।

### Q21. Explain the role of data mining in the Business Intelligence pyramid.
**Answer:** নিচ থেকে: Data sources → Data preprocessing/integration & warehouses (DBA) → Data exploration/EDA (statistical summary, querying, reporting) → **Data mining (information discovery, Data Analyst)** → Data presentation/visualization (Business Analyst) → Decision making (Top management)। যত উপরে, business decision support এর potential তত বেশি। DM raw data কে decision-ready knowledge এ পরিণত করে।

### Q22. Name the main data mining tasks with one application each.
**Answer:** Classification (loan approval), Regression (house price), Clustering (customer segmentation), Association rules (market basket — bread ⇒ butter), Anomaly/outlier detection (credit card fraud)। (Section 6 এর table টা লিখে দাও।)

---

## 20. Exam Tips / পরীক্ষার টিপস

1. **Definition হুবহু লেখো:** DM definition এর চারটা keyword — *non-trivial, implicit, previously unknown, potentially useful* — অবশ্যই লিখবে।
2. **Descriptive answer এর structure:** Definition → ব্যাখ্যা (2–3 লাইন) → **Example** → (সম্ভব হলে) **ছোট table/diagram**। Table দিলে examiner দ্রুত নম্বর দেয়।
3. **"Differentiate" প্রশ্নে সবসময় table:** Supervised vs Unsupervised, Classification vs Clustering, Classification vs Regression, EDA vs DM, Hard vs Soft clustering, SRSWOR vs SRSWR, Precision vs Recall।
4. **KDD process এ flow diagram আঁকো:** Input Data → Preprocessing → Data Mining → Post-processing → Knowledge।
5. **ছোট number example দিয়ে impress করো:**
   - Beer ⇒ Diaper (60%, 100%), Diaper ⇒ Beer (60%, 75%)
   - Confusion matrix 6954/46/412/2588 → Accuracy 95.42%
   - Covariance stock example → Cov = 4
   - kNN k = 3 → Red, k = 7 → Green
6. **Famous quote ব্যবহার করো:** "We are drowning in data, but starving for knowledge" — John Naisbitt। Introduction এ দারুণ লাগে।
7. **Formula গুলো মনে রাখো (এক জায়গায়):**
   - Accuracy $`= \frac{TP+TN}{All}`$, Precision $`= \frac{TP}{TP+FP}`$, Recall $`= \frac{TP}{TP+FN}`$, F1 $`= \frac{2PR}{P+R}`$
   - Support $`= P(X \cup Y)`$, Confidence $`= P(Y|X)`$
   - Min-max, Z-score, $`Cov = E(AB) - \bar{A}\bar{B}`$, $`r = \frac{Cov}{\sigma_A\sigma_B}`$
8. **Common traps:**
   - Simple query/search **data mining না**।
   - Data mining = KDD এর **একটা step**, পুরো KDD না।
   - kNN এ **odd k** — কারণ tie এড়ানো।
   - Cov = 0 মানেই independent **না**।
   - Imbalanced data তে শুধু accuracy দেখা **ভুল**।
   - Support দুই দিকে (X⇒Y, Y⇒X) একই, **confidence আলাদা**।
9. **Privacy প্রশ্নে তিনটা method + PPDM এর techniques** (randomization, k-anonymity, l-diversity, distributed, downgrading) লিস্ট করলেই পূর্ণ নম্বরের কাছাকাছি।
10. **Algorithm নাম মনে রাখো** (short note এ কাজে লাগে): Decision tree — ID3, C4.5, CART; Clustering — K-Means, DBSCAN, BIRCH; Association — Apriori, FP-Growth; Dimensionality reduction — PCA, LDA।
11. **Time management:** Descriptive প্রশ্নে বেশি লিখে সময় নষ্ট কোরো না — point আকারে, heading দিয়ে লেখো।

---

### Quick Revision (এক নজরে)
- **DM** = interesting pattern from huge data; KDD এর core step।
- **Steps:** Preprocess → Integrate → Warehouse → EDA → Mine → Visualize → Knowledge।
- **Supervised** (label আছে): Classification, Regression | **Unsupervised** (label নেই): Clustering, Association।
- **kNN:** majority vote of k nearest; odd k; normalize features; regression = average।
- **Metrics:** Acc, Error, Sensitivity, Specificity, Precision, Recall, F1; class imbalance এ accuracy misleading।
- **K-Means:** assign → compute centroid → reassign → until no change।
- **Association:** support & confidence; Apriori।
- **Preprocessing:** cleaning, integration, reduction, transformation।
- **EDA:** PCA (max variance), covariance, correlation [−1, 1], χ²।
- **Privacy:** remove IDs, security (multi-level, encryption), PPDM (randomization, k-anonymity, l-diversity, distributed, downgrading)।
