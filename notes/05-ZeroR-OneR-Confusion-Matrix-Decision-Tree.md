# 05 — ZeroR, OneR, Confusion Matrix & Decision Tree

> **Exam focus (teacher's note):** ZeroR, OneR, CM (Confusion Matrix), Decision Tree — **MATH (quiz-like), ~3 marks**; Decision Tree **~3 marks**.
> তাই এই note-এ সব formula + step-by-step worked example দেওয়া আছে। সব arithmetic Python দিয়ে verify করা।

**Sources:** SaedSayad (ZeroR, OneR, Decision Tree Classification and Regression), dataschool.io confusion-matrix guide, the class Confusion Matrix Example ppt and whiteboards, the Kaggle Fβ post, Medium (Madhu Sanjeevi on ID3/CART, Rishabh Jain on Entropy/Gini/Chi-square/Variance).

---

## 0. Running Dataset — "Play Golf" (14 rows)

ZeroR, OneR, আর Decision Tree, তিনটাতেই **এই একই dataset** ব্যবহার হয়েছে (SaedSayad version)। এটা ভালো করে চিনে রাখো।

| # | Outlook | Temp | Humidity | Windy | **Play Golf** |
|---|---|---|---|---|---|
| 1 | Rainy | Hot | High | False | No |
| 2 | Rainy | Hot | High | True | No |
| 3 | Overcast | Hot | High | False | Yes |
| 4 | Sunny | Mild | High | False | Yes |
| 5 | Sunny | Cool | Normal | False | Yes |
| 6 | Sunny | Cool | Normal | True | No |
| 7 | Overcast | Cool | Normal | True | Yes |
| 8 | Rainy | Mild | High | False | No |
| 9 | Rainy | Cool | Normal | False | Yes |
| 10 | Sunny | Mild | Normal | False | Yes |
| 11 | Rainy | Mild | Normal | True | Yes |
| 12 | Overcast | Mild | High | True | Yes |
| 13 | Overcast | Hot | Normal | False | Yes |
| 14 | Sunny | Mild | High | True | No |

- **Predictors (features):** Outlook, Temp, Humidity, Windy
- **Target (class):** Play Golf, যেখানে **Yes = 9, No = 5**

> ⚠️ এই version-এ **Sunny = 3 Yes / 2 No** আর **Rainy = 2 Yes / 3 No**। Quinlan-এর original weather data-তে এটা উল্টা। পরীক্ষায় প্রশ্নে যে table দেয় সেটাই follow করবে।

---

## 1. ZeroR (Zero Rule)

### Concept
**ZeroR** সবচেয়ে simple classifier। এটা **কোনো feature (predictor) দেখে না**, শুধু target-এর **majority class** টাই সবসময় predict করে।
- "Zero" rule, মানে zero predictor ব্যবহার করে।
- এর কোনো predictive power নেই। তবে এটা **baseline / benchmark** হিসেবে কাজে লাগে। অন্য যেকোনো classifier-এর accuracy অন্তত ZeroR-এর চেয়ে বেশি হওয়া উচিত।
- **Predictors contribution:** কিছুই না, কারণ ZeroR কোনো predictor use করে না।

### Algorithm
1. Target-এর **frequency table** বানাও।
2. সবচেয়ে frequent value (majority class) নাও। এটাই model।

### Worked Example (Play Golf)

| Play Golf | Yes | No |
|---|---|---|
| Count | **9** | 5 |

- Majority = **Yes**, তাই ZeroR model হলো **"Play Golf = Yes"** (সবার জন্য)।
- Accuracy $`= \dfrac{9}{14} = \mathbf{0.64}`$

**ZeroR-এর Confusion Matrix** (rows = Prediction, columns = Actual):

| ZeroR Prediction ↓ / Actual → | Yes | No | |
|---|---|---|---|
| **Yes** | TP = 9 | FP = 5 | Precision (PPV) = 9/14 = **0.64** |
| **No** | FN = 0 | TN = 0 | NPV = 0/0 → **0.00** (source লিখেছে 0) |
| | Sensitivity = 9/9 = **1.00** | Specificity = 0/5 = **0.00** | **Accuracy = 0.64** |

> ZeroR শুধু majority class-টাই ঠিক predict করে, তাই Recall = 1 আর Specificity = 0।
> **Null Error Rate** = সবসময় majority predict করলে যত ভুল হয় = 5/14 = 0.36 (= 1 − ZeroR accuracy)।

---

## 2. OneR (One Rule)

### Concept
**OneR** প্রতিটা predictor-এর জন্য **একটা করে rule** বানায়। তারপর যে predictor-এর rule-এর **total error সবচেয়ে কম**, সেটাকেই final "one rule" হিসেবে নেয়।
- Simple, কিন্তু accuracy প্রায় state-of-the-art algorithm-এর কাছাকাছি, আর rule মানুষ সহজে বুঝতে পারে।
- OneR কোনো score/probability দেয় না, তাই Gain, Lift, K-S, ROC chart এখানে প্রযোজ্য না।
- **Predictor contribution:** total error যত কম, contribution তত বেশি।

### Algorithm
```
For each predictor:
    For each value of that predictor:
        Count how often each class (target value) appears
        Find the most frequent class
        Make rule: assign that class to this value
    Calculate total error of this predictor's rules
Choose the predictor with the smallest total error
```

### Worked Example (Play Golf), Frequency Tables

প্রতিটা value-র জন্য majority class-ই rule। **Error = minority count** (যেগুলো rule ভুল বলবে)।

**Outlook**

| Outlook | Yes | No | Rule | Error |
|---|---|---|---|---|
| Sunny | 3 | 2 | → Yes | 2 |
| Overcast | 4 | 0 | → Yes | 0 |
| Rainy | 2 | 3 | → No | 2 |
| | | | **Total** | **4/14** |

**Temp**

| Temp | Yes | No | Rule | Error |
|---|---|---|---|---|
| Hot | 2 | 2 | → tie (Yes/No) | 2 |
| Mild | 4 | 2 | → Yes | 2 |
| Cool | 3 | 1 | → Yes | 1 |
| | | | **Total** | **5/14** |

**Humidity**

| Humidity | Yes | No | Rule | Error |
|---|---|---|---|---|
| High | 3 | 4 | → No | 3 |
| Normal | 6 | 1 | → Yes | 1 |
| | | | **Total** | **4/14** |

**Windy**

| Windy | Yes | No | Rule | Error |
|---|---|---|---|---|
| False | 6 | 2 | → Yes | 2 |
| True | 3 | 3 | → tie | 3 |
| | | | **Total** | **5/14** |

**Summary:**

| Predictor | Total Error |
|---|---|
| **Outlook** | **4/14** ★ |
| Temp | 5/14 |
| **Humidity** | **4/14** (tie) |
| Windy | 5/14 |

Outlook আর Humidity দুটোরই error 4/14 (tie)। Tie হলে যেকোনো একটা নেওয়া যায়। Source (SaedSayad) **Outlook** নিয়েছে (★)।

**Chosen OneR rule:**
```
IF Outlook = Sunny    THEN Play = Yes
IF Outlook = Overcast THEN Play = Yes
IF Outlook = Rainy    THEN Play = No
```

### OneR-এর Confusion Matrix
- Predicted Yes = Sunny + Overcast = 5 + 4 = 9 rows, এর মধ্যে actual Yes = 3 + 4 = **7 (TP)**, actual No = **2 (FP)**
- Predicted No = Rainy = 5 rows, এর মধ্যে actual Yes = **2 (FN)**, actual No = **3 (TN)**

| OneR Prediction ↓ / Actual → | Yes | No | |
|---|---|---|---|
| **Yes** | TP = 7 | FP = 2 | Precision = 7/9 = **0.78** |
| **No** | FN = 2 | TN = 3 | NPV = 3/5 = **0.60** |
| | Sensitivity = 7/9 = **0.78** | Specificity = 3/5 = **0.60** | **Accuracy = 10/14 = 0.71** |

ZeroR-এর accuracy 0.64 থেকে OneR-এ 0.71, অর্থাৎ improvement হয়েছে।

> **Hierarchy (SaedSayad):** ZeroR-এ **কোনো predictor নেই**। OneR **single best predictor** নেয়। Naive Bayes **সব predictor** নেয় (independence assumption ধরে)। Decision Tree **সব predictor** নেয়, তবে predictor-গুলোর মধ্যে **dependence** ধরে।

---

## 3. Confusion Matrix (CM)

### Concept
**Confusion matrix** একটা $`N \times N`$ table (N = class সংখ্যা)। এটা দেখায় model কতগুলো prediction ঠিক করেছে আর কতগুলো ভুল, actual label-এর সাথে তুলনা করে। যেকোনো classifier-এর **performance evaluate** করতে এটা use হয়।

### The 4 basic terms (whiteboard rule)

> **Positive** মানে classifier **YES predict করেছে।**
> - Actually YES হলে **TP** (True Positive)
> - Actually NO হলে **FP** (False Positive), যাকে **Type I error** বলে
>
> **Negative** মানে classifier **NO predict করেছে।**
> - Actually NO হলে **TN** (True Negative)
> - Actually YES হলে **FN** (False Negative), যাকে **Type II error** বলে

মনে রাখার trick: **2nd letter (P/N) = model কী predict করেছে**, আর **1st letter (T/F) = prediction সঠিক ছিল কি না।**

### Layout
এখানে **rows = Actual, columns = Predicted** (dataschool style):

| | **Predicted: Yes** | **Predicted: No** | Total |
|---|---|---|---|
| **Actual: Yes** | TP | FN | AP = TP + FN |
| **Actual: No** | FP | TN | AN = FP + TN |
| Total | PP = TP + FP | PN = FN + TN | N |

> ⚠️ SaedSayad-এর table-এ **rows = Model (Predicted), columns = Target (Actual)**, অর্থাৎ transpose। তাই সবসময় label পড়ে নিয়ে TP/FP/FN/TN চিনবে। Position মুখস্থ করবে না।
> Notation: **AP, AN** = Actual Positive/Negative; **PP, PN** = Predicted Positive/Negative.

### Formulas (সব একসাথে)

| Metric | Formula | মানে (বাংলায়) |
|---|---|---|
| **Accuracy** (recognition rate) | $`\dfrac{TP+TN}{N}`$ | মোট prediction-এর মধ্যে কতগুলো ঠিক |
| **Error rate** / Misclassification rate | $`\dfrac{FP+FN}{N} = 1 - \text{Accuracy}`$ | মোট কতগুলো ভুল |
| **Recall** = Sensitivity = **TPR** | $`\dfrac{TP}{TP+FN} = \dfrac{TP}{AP}`$ | আসল Yes-গুলোর কতটা ধরতে পেরেছি (**completeness**) |
| **Precision** = PPV | $`\dfrac{TP}{TP+FP} = \dfrac{TP}{PP}`$ | যেগুলোকে Yes বলেছি, তার কতগুলো সত্যিই Yes (**exactness**) |
| **Specificity** = TNR | $`\dfrac{TN}{TN+FP} = \dfrac{TN}{AN}`$ | আসল No-গুলোর কতটা ঠিকভাবে No বলেছি |
| **FPR** (False Positive Rate) | $`\dfrac{FP}{FP+TN} = 1 - \text{Specificity}`$ | আসল No-কে ভুল করে Yes বলার হার |
| **NPV** | $`\dfrac{TN}{TN+FN} = \dfrac{TN}{PN}`$ | যেগুলোকে No বলেছি, তার কতগুলো সত্যিই No |
| **Prevalence** | $`\dfrac{AP}{N}`$ | data-তে Yes কত ঘন ঘন আসে |
| **Null error rate** | $`\dfrac{\text{minority count}}{N}`$ | সবসময় majority predict করলে error কত (ZeroR-এর error) |
| **F1 score** | $`\dfrac{2 \cdot P \cdot R}{P + R}`$ | Precision আর Recall-এর **harmonic mean** |
| **Fβ score** | $`\dfrac{(1+\beta^2)\, P \cdot R}{\beta^2 P + R}`$ | Weighted F, যেখানে Recall-কে β গুণ বেশি গুরুত্ব দেওয়া হয় |

- Perfect score = 1.0। Precision আর Recall-এর মধ্যে সাধারণত **inverse relationship** থাকে (একটা বাড়ালে অন্যটা কমে)।
- ROC curve মানে TPR (y-axis) vs FPR (x-axis), threshold বদলে বদলে plot করা হয়।

### Fβ Score: β কী করে?
- **β = 1** হলে F1, যেখানে Precision আর Recall সমান গুরুত্ব পায়।
- **β > 1** (যেমন F2, F3) হলে **Recall-কে বেশি গুরুত্ব**। Medical diagnosis-এ এটা দরকার, কারণ কোনো রোগী miss করা (FN) চলবে না।
- **β < 1** (যেমন F0.5) হলে **Precision-কে বেশি গুরুত্ব**। এখানে false alarm (FP) কমানো জরুরি, যেমন spam filter।
- Fβ "assigns **β times as much weight to recall** as to precision."
- F2 formula: $`F_2 = \dfrac{5PR}{4P + R}`$। F0.5 formula: $`F_{0.5} = \dfrac{1.25\,PR}{0.25P + R}`$।

---

### ✍️ Worked Example 1 (Class Example / dataschool): **খুব গুরুত্বপূর্ণ**

> **Q:** A classifier made a total of **165** predictions, out of which it predicted 'Yes' **110** times. In reality **105** sample points actually belonged to 'Yes'. Given **TP = 100**. Construct the confusion matrix and calculate all metrics.

**Step 1: Missing values বের করো**
- Predicted No (PN) = 165 − 110 = **55**
- Actual No (AN) = 165 − 105 = **60**
- FP = PP − TP = 110 − 100 = **10**
- FN = AP − TP = 105 − 100 = **5**
- TN = PN − FN = 55 − 5 = **50** (check: AN − FP = 60 − 10 = 50 ✓)

**Step 2: CM**

| N = 165 | Predicted: Yes | Predicted: No | Total |
|---|---|---|---|
| **Actual: Yes** | TP = 100 | FN = 5 | AP = 105 |
| **Actual: No** | FP = 10 | TN = 50 | AN = 60 |
| Total | PP = 110 | PN = 55 | 165 |

**Step 3: Metrics**

| Metric | Calculation | Value |
|---|---|---|
| Accuracy | (100 + 50)/165 = 150/165 | **0.91** |
| Misclassification / Error rate | (10 + 5)/165 = 15/165 | **0.09** |
| TPR = Sensitivity = Recall | 100/105 | **0.95** (0.9524) |
| FPR | 10/60 | **0.17** |
| Specificity | 50/60 | **0.83** |
| Precision | 100/110 | **0.91** (0.9091) |
| NPV | 50/55 | 0.91 |
| Prevalence | 105/165 | **0.64** |
| Null error rate | 60/165 | 0.36 |
| **F1** | 2(0.9091)(0.9524)/(0.9091 + 0.9524) | **0.93** (0.9302) |
| **F2** | 5(0.9091)(0.9524)/(4·0.9091 + 0.9524) | **0.943** |
| **F3** | 10PR/(9P + R) | **0.948** |
| F0.5 | 1.25PR/(0.25P + R) | 0.917 |

> লক্ষ্য করো: এখানে Recall > Precision, তাই β বাড়ালে Fβ বাড়ে (F0.5 < F1 < F2 < F3)।
> **F score / F measure / F1 score**, তিনটা নামই একই জিনিস বোঝায় (β = 1)।

### ✍️ Worked Example 2 (SaedSayad)

| Model ↓ / Target → | Positive | Negative | |
|---|---|---|---|
| **Positive** | a = TP = 70 | b = FP = 20 | Precision = 70/90 = **0.78** |
| **Negative** | c = FN = 30 | d = TN = 80 | NPV = 80/110 = **0.73** |
| | Recall = 70/100 = **0.70** | Specificity = 80/100 = **0.80** | Accuracy = 150/200 = **0.75** |

- Generic formula (a, b, c, d): Precision = a/(a + b), NPV = d/(c + d), Sensitivity = a/(a + c), Specificity = d/(b + d), Accuracy = (a + d)/(a + b + c + d)
- F1 = 2(0.7778)(0.70)/(0.7778 + 0.70) = **0.737**. F2 = 0.714, F0.5 = 0.761।

---

## 4. Decision Tree (DT)

### 4.1 Basics & Terminology
**Decision tree** classification বা regression model-কে **tree structure** আকারে বানায়। Dataset-কে ধাপে ধাপে ছোট ছোট subset-এ ভাঙে।
- **Root Node:** সবচেয়ে উপরের node, যেটা best predictor (পুরো population)
- **Decision Node:** যে node আরও split হয় (যেমন Outlook → Sunny/Overcast/Rainy)
- **Leaf / Terminal Node:** আর split হয় না, এটাই final class/value (যেমন Play = Yes)
- **Splitting:** একটা node-কে sub-nodes-এ ভাগ করা। **Pruning** হলো এর উল্টা, node কেটে tree ছোট করা।
- **Branch / Sub-tree**, **Parent / Child node**
- **Types:** Categorical target হলে **Classification tree**, continuous target হলে **Regression tree**।

**Algorithms ও তাদের split metric:**

| Algorithm | Split metric | Target |
|---|---|---|
| **ID3** (Quinlan) | **Entropy + Information Gain** | Categorical |
| **C4.5** | **Normalized** Information Gain (gain ratio); continuous attributes, missing values, pruning handle করে | Categorical |
| **CART** | **Gini Index** (binary split) | Classification + Regression |
| **CHAID** | **Chi-square** | Categorical, multi-way split |
| Reduction in Variance / **SDR** | Variance / Standard Deviation | Continuous (Regression) |

ID3 হলো **top-down, greedy search, no backtracking**। প্রতিটা ধাপে সবচেয়ে ভালো attribute বেছে নেয়, পরে ফিরে গিয়ে আর বদলায় না।

---

### 4.2 Entropy

```math
E(S) = \sum_{i=1}^{c} -p_i \log_2 p_i
```

- **Entropy** মানে sample-এর **impurity / uncertainty** কতটা।
- পুরো homogeneous (সব Yes বা সব No) হলে **E = 0**
- 50% Yes / 50% No হলে **E = 1** (binary-তে maximum)
- নিয়ম: $`0 \log_2 0 = 0`$

**Two-attribute entropy** (split-এর পরে weighted average):
```math
E(T, X) = \sum_{c \in X} P(c)\, E(c)
```

**Quick values মুখস্থ রাখো (binary):**

| Split (a, b) | Entropy |
|---|---|
| (k, 0) | 0 |
| (1, 1), (2, 2), (3, 3) | 1.000 |
| (3, 2) / (2, 3) | 0.971 |
| (9, 5) | 0.940 |
| (3, 1) / (6, 2) | 0.811 |
| (4, 2) / (2, 1) | 0.918 |
| (4, 3) / (3, 4) | 0.985 |
| (6, 1) | 0.592 |

### 4.3 Information Gain (IG)

```math
IG(T, X) = E(T) - E(T, X)
```

Split-এর পরে entropy কতটা **কমলো**, সেটাই IG। **যার IG সবচেয়ে বেশি, সেই attribute হবে decision node।**

---

### 4.4 ✍️ Full ID3 Worked Example (Play Golf), **must practice**

#### Step 1: Target-এর Entropy
Yes = 9, No = 5, N = 14

```math
E(\text{PlayGolf}) = E(9,5) = -\tfrac{9}{14}\log_2\tfrac{9}{14} - \tfrac{5}{14}\log_2\tfrac{5}{14}
```
```math
= -(0.643)\log_2(0.643) - (0.357)\log_2(0.357) = 0.4098 + 0.5305 = \mathbf{0.940}
```

#### Step 2: প্রতিটা attribute-এর জন্য E(T, X) আর IG

**(a) Outlook**

| Outlook | Yes | No | Total | Entropy |
|---|---|---|---|---|
| Sunny | 3 | 2 | 5 | E(3,2) = 0.4422 + 0.5288 = **0.971** |
| Overcast | 4 | 0 | 4 | **0** |
| Rainy | 2 | 3 | 5 | **0.971** |

```math
E(T,\text{Outlook}) = \tfrac{5}{14}(0.971) + \tfrac{4}{14}(0) + \tfrac{5}{14}(0.971) = 0.347 + 0 + 0.347 = \mathbf{0.693}
```
```math
IG(\text{Outlook}) = 0.940 - 0.693 = \mathbf{0.247}
```

**(b) Temp**

| Temp | Yes | No | Total | Entropy |
|---|---|---|---|---|
| Hot | 2 | 2 | 4 | 1.000 |
| Mild | 4 | 2 | 6 | 0.390 + 0.528 = 0.918 |
| Cool | 3 | 1 | 4 | 0.811 |

```math
E(T,\text{Temp}) = \tfrac{4}{14}(1) + \tfrac{6}{14}(0.918) + \tfrac{4}{14}(0.811) = 0.286 + 0.394 + 0.232 = \mathbf{0.911}
```
```math
IG(\text{Temp}) = 0.940 - 0.911 = \mathbf{0.029}
```

**(c) Humidity**

| Humidity | Yes | No | Total | Entropy |
|---|---|---|---|---|
| High | 3 | 4 | 7 | 0.5239 + 0.4613 = 0.985 |
| Normal | 6 | 1 | 7 | 0.1906 + 0.4011 = 0.592 |

```math
E(T,\text{Humidity}) = \tfrac{7}{14}(0.985) + \tfrac{7}{14}(0.592) = \mathbf{0.788}
```
```math
IG(\text{Humidity}) = 0.940 - 0.788 = \mathbf{0.152}
```

**(d) Windy**

| Windy | Yes | No | Total | Entropy |
|---|---|---|---|---|
| False | 6 | 2 | 8 | 0.311 + 0.500 = 0.811 |
| True | 3 | 3 | 6 | 1.000 |

```math
E(T,\text{Windy}) = \tfrac{8}{14}(0.811) + \tfrac{6}{14}(1) = 0.464 + 0.429 = \mathbf{0.892}
```
```math
IG(\text{Windy}) = 0.940 - 0.892 = \mathbf{0.048}
```

#### Step 3: Highest IG বেছে নাও

| Attribute | IG |
|---|---|
| **Outlook** | **0.247** ★ |
| Humidity | 0.152 |
| Windy | 0.048 |
| Temp | 0.029 |

**Root node = Outlook.** (Source-এ লেখা আছে "IG = 0.94 − 0.69 = 0.24", আসলে rounding-এর কারণে, exact মান 0.247।)

#### Step 4a: Overcast branch
Overcast = (4 Yes, 0 No), E(4,0) = 0, তাই এটা **LEAF: Play = Yes**

#### Step 4b: Sunny branch (5 rows: 3 Yes, 2 No; E = 0.971)

| Sunny rows | Temp | Humidity | Windy | Play |
|---|---|---|---|---|
| 4 | Mild | High | False | Yes |
| 5 | Cool | Normal | False | Yes |
| 6 | Cool | Normal | True | No |
| 10 | Mild | Normal | False | Yes |
| 14 | Mild | High | True | No |

- **Temp:** Cool (1Y, 1N) E = 1; Mild (2Y, 1N) E = 0.918, তাই E = (2/5)(1) + (3/5)(0.918) = 0.4 + 0.551 = 0.951, **IG = 0.971 − 0.951 = 0.020**
- **Humidity:** High (1Y, 1N) E = 1; Normal (2Y, 1N) E = 0.918, তাই E = 0.951, **IG = 0.020**
- **Windy:** False (3Y, 0N) E = 0; True (0Y, 2N) E = 0, তাই E = 0, **IG = 0.971** ★

তাই Sunny-র নিচে **Windy** বসবে। False হলে Yes, True হলে No (দুটোই leaf)।

#### Step 4c: Rainy branch (5 rows: 2 Yes, 3 No; E = 0.971)

| Rainy rows | Temp | Humidity | Windy | Play |
|---|---|---|---|---|
| 1 | Hot | High | False | No |
| 2 | Hot | High | True | No |
| 8 | Mild | High | False | No |
| 9 | Cool | Normal | False | Yes |
| 11 | Mild | Normal | True | Yes |

- **Temp:** Hot (0Y, 2N) = 0; Mild (1Y, 1N) = 1; Cool (1Y, 0N) = 0, তাই E = (2/5)(1) = 0.4, **IG = 0.571**
- **Humidity:** High (0Y, 3N) = 0; Normal (2Y, 0N) = 0, তাই E = 0, **IG = 0.971** ★
- **Windy:** False (1Y, 2N) = 0.918; True (1Y, 1N) = 1, তাই E = (3/5)(0.918) + (2/5)(1) = 0.951, **IG = 0.020**

তাই Rainy-র নিচে **Humidity** বসবে। High হলে No, Normal হলে Yes।

#### Step 5: Final Tree
সব branch-এর entropy 0 হলে ID3 থামে (সব leaf)।

```
                    Outlook
          /            |            \
       Sunny        Overcast        Rainy
         |             |              |
       Windy       Play=Yes       Humidity
       /    \                     /       \
   FALSE    TRUE               High      Normal
  Play=Yes  Play=No           Play=No   Play=Yes
```

#### Decision Tree to Decision Rules
Root থেকে প্রতিটা leaf পর্যন্ত একটা path মানে একটা rule:
- R1: IF Outlook = Sunny AND Windy = FALSE THEN Play = Yes
- R2: IF Outlook = Sunny AND Windy = TRUE THEN Play = No
- R3: IF Outlook = Overcast THEN Play = Yes
- R4: IF Outlook = Rainy AND Humidity = High THEN Play = No
- R5: IF Outlook = Rainy AND Humidity = Normal THEN Play = Yes

---

### 4.5 Gini Index (CART)

দুইভাবে লেখা হয়, তাই **convention সাবধানে খেয়াল করবে:**

**(i) Gini Impurity** (CART / Madhu Sanjeevi). এখানে **কম মানে ভালো**:
```math
Gini = 1 - \sum_{i=1}^{k} p_i^2 \quad\quad (\text{binary: } 1 - p^2 - q^2)
```
- Pure node হলে 0। Binary-তে max = 1 − 0.5² − 0.5² = **0.5**। k class হলে max = **1 − 1/k**।
- Split-এর Gini = sub-node Gini-গুলোর **weighted average**। **Gini gain = Gini(parent) − weighted Gini**, যেটা সবচেয়ে বেশি সেটা নাও।

**Play Golf-এ Gini impurity:**
- Root: $`1 - (9/14)^2 - (5/14)^2 = 1 - 0.413 - 0.128 = \mathbf{0.459}`$
- Outlook: Sunny = 1 − 0.6² − 0.4² = 0.48, Overcast = 0, Rainy = 0.48, তাই weighted = (5/14)(0.48)·2 = **0.343**, gain = **0.116** ★
- Humidity: High = 1 − (3/7)² − (4/7)² = 0.490, Normal = 1 − (6/7)² − (1/7)² = 0.245, তাই weighted = **0.367**, gain = 0.092
- Windy: weighted = (8/14)(0.375) + (6/14)(0.5) = **0.429**, gain = 0.031
- Temp: weighted = (4/14)(0.5) + (6/14)(0.444) + (4/14)(0.375) = **0.440**, gain = 0.019

Gini দিয়েও **Outlook** best। ID3-এর result-এর সাথে মিলে যায়।

**(ii) "Gini score" = $`p^2 + q^2`$** (Rishabh Jain / Analytics-Vidhya style). এখানে **বেশি মানে বেশি homogeneous**, তাই বেশি মানে ভালো।

**Cricket example:** 30 জন student, 15 জন cricket খেলে (50%)।
- Gender: Female 10 জন (2 play, 8 not), Male 20 জন (13 play, 7 not)
- Class: IX 14 জন (6 play, 8 not), X 16 জন (9 play, 7 not)

| Split | Sub-node | $`p^2+q^2`$ | Weighted |
|---|---|---|---|
| Gender | Female (0.2, 0.8) | 0.04 + 0.64 = **0.68** | |
| | Male (0.65, 0.35) | 0.4225 + 0.1225 = **0.545 ≈ 0.55** | (10/30)(0.68) + (20/30)(0.55) = **0.59** |
| Class | IX (0.43, 0.57) | **0.51** | |
| | X (0.56, 0.44) | **0.51** | (14/30)(0.51) + (16/30)(0.51) = **0.51** |

Gender-এর score (0.59) > Class-এর score (0.51), তাই **Gender-এ split হবে।**

---

### 4.6 Chi-Square (CHAID)

Sub-node আর parent node-এর পার্থক্য statistically significant কি না, সেটা মাপে। Observed (Actual) আর Expected frequency-র standardized difference দেখে।
- Categorical target-এ কাজ করে। **2 বা তার বেশি split** করতে পারে। **Chi-square যত বেশি, split তত significant (ভালো)।** এভাবে যে tree বানায় তার নাম **CHAID**।
- Source-এর formula (প্রতিটা cell-এর জন্য):
```math
\chi = \sqrt{\frac{(\text{Actual} - \text{Expected})^2}{\text{Expected}}}
```
(Standard Pearson χ²-এ square root নেওয়া হয় না: $`\chi^2 = \sum \frac{(A-E)^2}{E}`$। Exam-এ source-এর version দিলে সেটাই ব্যবহার করবে।)
- Split-এর chi-square = সব node-এর Success আর Failure-এর chi value যোগ করে।

**Cricket example** (parent-এ play-এর probability 50%, তাই Expected = node size × 0.5):

| Node | Play (A) | Not (A) | Expected (each) | Dev (Play / Not) | χ Play | χ Not | Sum |
|---|---|---|---|---|---|---|---|
| Female | 2 | 8 | 5 | −3 / +3 | √(9/5) = 1.34 | 1.34 | 2.68 |
| Male | 13 | 7 | 10 | +3 / −3 | √(9/10) = 0.95 | 0.95 | 1.90 |
| **Gender total** | | | | | | | **4.58** |
| Class IX | 6 | 8 | 7 | −1 / +1 | √(1/7) = 0.38 | 0.38 | 0.76 |
| Class X | 9 | 7 | 8 | +1 / −1 | √(1/8) = 0.35 | 0.35 | 0.71 |
| **Class total** | | | | | | | **1.46** |

Gender (4.58) > Class (1.46), তাই **Gender split বেশি significant।**

---

### 4.7 Reduction in Variance (continuous target)

```math
\text{Variance} = \frac{\sum (X - \bar{X})^2}{n}
```
**Variance যত কম, node তত homogeneous।** Split-এর variance = sub-node variance-গুলোর **weighted average**। যে split-এ variance সবচেয়ে কম, সেটা নাও।

**Cricket example** (play = 1, not play = 0):
- Root: mean 0.5, Var = (15(0.5)² + 15(0.5)²)/30 = **0.25**
- Female: mean = 2/10 = 0.2, Var = (2(0.8)² + 8(0.2)²)/10 = **0.16**
- Male: mean = 13/20 = 0.65, Var = (13(0.35)² + 7(0.65)²)/20 = **0.2275 ≈ 0.23**
- **Gender split** = (10/30)(0.16) + (20/30)(0.23) = **0.21**
- Class IX: mean 0.43, Var ≈ **0.24**. Class X: mean 0.56, Var ≈ **0.25**
- **Class split** = (14/30)(0.24) + (16/30)(0.25) = **0.25**

Gender-এর variance (0.21) সবচেয়ে কম, তাই **Gender-এ split।**

---

### 4.8 Decision Tree for Regression: Standard Deviation Reduction (SDR)

Target যখন **numeric** (যেমন Hours Played), তখন ID3-তে **Information Gain-এর জায়গায় SDR** ব্যবহার হয়।

- **Standard Deviation (S):** tree building/branching-এর জন্য
- **Coefficient of Variation, CV = S/x̄ × 100%:** কখন branching থামাতে হবে (যেমন CV < 10% হলে বা n ≤ 3 হলে stop)
- **Average:** leaf node-এর value (prediction)

```math
S = \sqrt{\frac{\sum (x-\bar{x})^2}{n}}, \qquad S(T,X) = \sum_{c\in X} P(c)\,S(c), \qquad SDR(T,X) = S(T) - S(T,X)
```

**Dataset:** একই Play Golf predictors, target = Hours Played:
25, 30, 46, 45, 52, 23, 43, 35, 38, 46, 48, 52, 44, 30 (row 1 থেকে 14)

**Step 1: Target**
n = 14, x̄ = 557/14 = **39.8**, S = **9.32**, CV = 9.32/39.8 = **23%**

**Step 2: Outlook-এর SDR**

| Outlook | Hours | Mean | S | n |
|---|---|---|---|---|
| Overcast | 46, 43, 52, 44 | 46.3 | 3.49 | 4 |
| Rainy | 25, 30, 35, 38, 48 | 35.2 | 7.78 | 5 |
| Sunny | 45, 52, 23, 46, 30 | 39.2 | 10.87 | 5 |

```math
S(\text{Hours},\text{Outlook}) = \tfrac{4}{14}(3.49) + \tfrac{5}{14}(7.78) + \tfrac{5}{14}(10.87) = \mathbf{7.66}
```
```math
SDR = 9.32 - 7.66 = \mathbf{1.66}
```

**Step 3: সব attribute**

| Attribute | Branch S values | SDR |
|---|---|---|
| **Outlook** | 3.49 / 7.78 / 10.87 | **1.66** ★ |
| Temp | Cool 10.51, Hot 8.95, Mild 7.65 | 0.48 (source-এ 0.17 লেখা, কিন্তু source-এর নিজের S values দিয়ে হিসাব করলে 0.48 আসে) |
| Humidity | High 9.36, Normal 8.37 (৭ জন করে) | 0.28 |
| Windy | False 7.87, True 10.59 | 0.29 |

তাই Root = **Outlook**।

**Step 4: Branches**
- **Overcast:** CV = 3.49/46.3 ≈ **8% < 10%**, তাই LEAF = average = **46.3**
- **Sunny:** CV = 10.87/39.2 ≈ 28% > 10%, তাই আরও split লাগবে। SDR হিসাব করলে Windy = **7.62** (best), Temp = 0.68, Humidity = 0.37। তাই **Windy** নেওয়া হলো।
  - FALSE {45, 52, 46}, avg **47.7**; TRUE {23, 30}, avg **26.5**. দুটোতেই n ≤ 3, তাই stop।
- **Rainy:** CV = 7.78/35.2 ≈ 22% > 10%, তাই split। SDR: **Temp = 4.18** (best), Humidity = 3.33, Windy = 0.85। (Source text-এ "Windy" লেখা, কিন্তু final tree আর হিসাব দুটোই **Temp** দেখায়।)
  - Cool {38}, avg **38**; Hot {25, 30}, avg **27.5**; Mild {35, 48}, avg **41.5**

```
                 Outlook
        /           |            \
     Sunny       Overcast        Rainy
       |          46.3            |
     Windy                       Temp
    /     \                /      |      \
 FALSE    TRUE           Cool    Hot     Mild
 47.7     26.5            38     27.5    41.5
```

**Prediction:** ⟨Sunny, Hot, Normal, Windy = True⟩ হলে path হবে Sunny → Windy = TRUE, তাই **26.5 hours**।

---

### 4.9 DT Issues (short)
- **Continuous attributes:** ID3 শুধু categorical নিতে পারে, তাই আগে **binning (discretization)** করতে হয়।
  - Unsupervised binning: **Equal width** (w = (max − min)/k), **Equal frequency**, rank, quantiles, floor(log x)
  - Supervised binning: **Entropy-based**, যেখানে যে cut point-এ IG সবচেয়ে বেশি সেটা নেওয়া হয়।
- **Overfitting:** এড়াতে **Pruning** করা হয় (reduced error pruning, cost complexity pruning)।
- **Super attributes:** যে attribute-এর অনেক unique value (যেমন ID), সেটা IG-কে ভুলভাবে বাড়িয়ে দেয়। C4.5-এর gain ratio এটা ঠিক করে।
- **Missing values** (C4.5-এ "?" দিয়ে mark করা হয়, gain হিসাবে বাদ দেওয়া হয়)।

**Entropy-based binning example (O-ring failure, 24 records: 7 Y, 17 N):**
- $`E(7,17) = E(0.29, 0.71) = 0.871`$
- Cut "≤60 / >60": (3Y, 0N) আর (4Y, 17N). E(3,0) = 0, E(4,17) = E(0.19, 0.81) = 0.702
  - $`E = \tfrac{3}{24}(0) + \tfrac{21}{24}(0.702) = 0.615`$, তাই **Gain = 0.871 − 0.615 = 0.256** ★
- Cut ≤70: Gain = 0.101. Cut ≤75: Gain = 0.148
- তাই best cut হলো **(≤60, >60)**।

**Advantages:** সহজে interpret করা যায় (black box না), কম data cleaning লাগে, outlier আর missing value-তে কম প্রভাবিত হয়, numerical আর categorical দুটোই handle করে, **non-parametric**।

---

## 5. Exam Tips / পরীক্ষার টিপস

1. **CM বানানোর আগে orientation ঠিক করো।** Rows Actual নাকি Predicted, table-এর label দেখে বুঝবে। তারপর TP, FP, FN, TN লিখে নাও।
2. **Missing cell বের করার trick:** FP = PP − TP, FN = AP − TP, TN = N − TP − FP − FN। Row আর column totals মিলিয়ে check করো।
3. **Precision বনাম Recall:** দুটোরই numerator TP। Precision-এর denominator **Predicted** Yes, Recall-এর denominator **Actual** Yes।
4. **Specificity = TN/AN**, আর **FPR = 1 − Specificity**।
5. **F1 হলো harmonic mean, arithmetic mean না।** Fβ formula-তে **β² শুধু denominator-এর P-এর সাথে** গুণ হয়।
6. Entropy হিসাবে $`\log_2`$ ব্যবহার করবে। Calculator-এ $`\log_2 x = \ln x / \ln 2`$ বা $`\log_{10}x / 0.30103`$।
7. Pure node মানে E = 0 আর leaf। 50/50 split মানে E = 1 (binary)।
8. **IG বেশি মানে ভালো। Gini impurity কম মানে ভালো। Chi-square বেশি মানে ভালো। Variance কম মানে ভালো। SDR বেশি মানে ভালো।**
9. OneR-এ **error = প্রতিটা value-র minority count-এর যোগফল**। Tie হলে যেকোনো একটা বলে দিয়ে উল্লেখ করো যে tie হয়েছে।
10. ZeroR accuracy = majority/N। এটাই baseline।
11. সব step দেখাবে (fraction আর decimal দুটোই), আর 3 decimal পর্যন্ত রাখবে।

---

## 6. Practice Problems (Quiz-style) with Answers

**Q1 (ZeroR).** একটা dataset-এ 10টা record আছে: 7 Yes, 3 No। ZeroR model কী, আর accuracy কত?
<details><summary>Answer</summary>

Model: সবসময় **Yes** predict করে। Accuracy = 7/10 = **0.70**। CM: TP = 7, FP = 3, FN = 0, TN = 0। Recall = 1, Specificity = 0।
</details>

**Q2 (OneR + ZeroR).** নিচের data দিয়ে OneR rule বের করো, আর ZeroR-এর সাথে তুলনা করো।

| # | Weather | Wind | Play |
|---|---|---|---|
| 1 | Sunny | Weak | Yes |
| 2 | Sunny | Strong | Yes |
| 3 | Sunny | Weak | Yes |
| 4 | Rain | Weak | No |
| 5 | Rain | Strong | No |
| 6 | Sunny | Strong | No |
| 7 | Rain | Weak | No |
| 8 | Rain | Strong | No |

<details><summary>Answer</summary>

- **Weather:** Sunny (3Y, 1N) → Yes, err 1; Rain (0Y, 4N) → No, err 0. Total = **1/8**
- **Wind:** Weak (2Y, 2N) → tie, err 2; Strong (1Y, 3N) → No, err 1. Total = **3/8**
- **OneR rule:** IF Weather = Sunny THEN Yes; IF Weather = Rain THEN No. Accuracy = **7/8 = 0.875**
- **ZeroR:** No (5 of 8), accuracy = **5/8 = 0.625**
</details>

**Q3 (CM).** TP = 40, FP = 10, FN = 20, TN = 30। Accuracy, Error rate, Precision, Recall, Specificity, F1, F2 বের করো।
<details><summary>Answer</summary>

N = 100
- Accuracy = 70/100 = **0.70**; Error = **0.30**
- Precision = 40/50 = **0.80**; Recall = 40/60 = **0.667**
- Specificity = 30/40 = **0.75**; FPR = 0.25
- F1 = 2(0.8)(0.667)/(0.8 + 0.667) = **0.727**
- F2 = 5(0.8)(0.667)/(4(0.8) + 0.667) = 2.667/3.867 = **0.690** (Recall কম, তাই F2 < F1)
</details>

**Q4 (CM from totals).** Total 100টা prediction। Classifier 60 বার Yes বলেছে। Actual Yes = 50। TP = 45। CM বানাও, তারপর Accuracy, Precision, Recall, Specificity, F1, F2, F0.5 বের করো।
<details><summary>Answer</summary>

PN = 40, AN = 50, FP = 60 − 45 = 15, FN = 50 − 45 = 5, TN = 40 − 5 = 35

| | Pred Yes | Pred No | |
|---|---|---|---|
| Act Yes | 45 | 5 | 50 |
| Act No | 15 | 35 | 50 |
| | 60 | 40 | 100 |

- Accuracy = 80/100 = **0.80**; Precision = 45/60 = **0.75**; Recall = 45/50 = **0.90**; Specificity = 35/50 = **0.70**
- F1 = 2(0.75)(0.9)/1.65 = **0.818**; F2 = 5(0.675)/(3.0 + 0.9) = **0.865**; F0.5 = 1.25(0.675)/(0.1875 + 0.9) = **0.776**
</details>

**Q5 (Entropy / Gini).** একটা node-এ 6 Yes আর 2 No। Entropy আর Gini impurity কত?
<details><summary>Answer</summary>

p = 0.75, q = 0.25
- E = −0.75 log₂0.75 − 0.25 log₂0.25 = 0.311 + 0.5 = **0.811**
- Gini = 1 − 0.5625 − 0.0625 = **0.375**
</details>

**Q6 (Information Gain).** Parent node-এ 8 Yes, 8 No। Attribute A data-কে দুই ভাগ করে: (6Y, 2N) আর (2Y, 6N)। IG(A) কত?
<details><summary>Answer</summary>

E(parent) = 1। প্রতিটা child-এর E = 0.811। Weighted = (8/16)(0.811) + (8/16)(0.811) = 0.811। **IG = 1 − 0.811 = 0.189**
</details>

**Q7 (ID3 recall).** Play Golf data-তে root কোনটা, আর কেন? Sunny-র নিচে কোন attribute?
<details><summary>Answer</summary>

IG: Outlook 0.247, Humidity 0.152, Windy 0.048, Temp 0.029। তাই **Outlook** root। Sunny-র নিচে **Windy** (IG = 0.971, কারণ দুই branch-ই pure)। Rainy-র নিচে **Humidity**।
</details>

**Q8 (SDR).** Target values: 10, 20, 30, 40। Attribute B এগুলোকে {10, 20} আর {30, 40}-এ ভাগ করে। SDR কত?
<details><summary>Answer</summary>

Mean = 25, S = √((225 + 25 + 25 + 225)/4) = √125 = **11.18**। প্রতিটা child-এর S = 5, তাই weighted = 5। **SDR = 11.18 − 5 = 6.18**
</details>

**Q9 (Chi-square, source formula).** Parent-এ 50% success। একটা node-এ 10 জন: 2 success, 8 failure। Node-এর chi-square কত?
<details><summary>Answer</summary>

Expected = 5 আর 5। Deviation হলো −3 আর +3। χ = √(9/5) + √(9/5) = 1.342 + 1.342 = **2.68**
</details>

**Q10 (Concept).** Medical screening-এ কোন metric বেশি গুরুত্বপূর্ণ, আর কোন Fβ নেবে?
<details><summary>Answer</summary>

**Recall (Sensitivity)** বেশি গুরুত্বপূর্ণ, কারণ রোগী miss করা (FN) খুব বিপজ্জনক। তাই **β > 1**, যেমন F2।
</details>
