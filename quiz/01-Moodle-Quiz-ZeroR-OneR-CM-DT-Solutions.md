# Moodle Quiz — ZeroR, OneR, Confusion Matrix, Decision Tree (Solved)

> Mid syllabus: "ZeroR, OneR, CM, DT ⇒ Math (**Moodle like**), 3 marks" — exam এর প্রশ্ন ঠিক এই ধরনের হবে।
> নিচে প্রতিটা প্রশ্ন **হুবহু (full question + সব option)** দেওয়া, তারপর **Answer** আর **কিভাবে solve করবে (step by step)**।
> Answer key এর 11টা answer ই course এর formula দিয়ে Python এ verify করা — সব সঠিক ✅
> Theory: [05-ZeroR-OneR-Confusion-Matrix-Decision-Tree](../notes/05-ZeroR-OneR-Confusion-Matrix-Decision-Tree.md)

## Formula Cheat Sheet

| কী | Formula (course এর version) |
|---|---|
| OneR error | প্রতিটা partition এ **minority class এর count** = error। সব partition এর error যোগ করো। সবচেয়ে কম total error যার → সেই predictor |
| OneR prediction | প্রতিটা partition এ **majority class** predict করে |
| ZeroR | সবসময় **majority class** predict করে, কোনো feature দেখে না |
| Precision | $\dfrac{TP}{TP+FP}$ |
| Recall | $\dfrac{TP}{TP+FN}$ |
| F1 | $\dfrac{2 \cdot P \cdot R}{P+R}$ |
| Entropy | $H = -p\log_2 p - q\log_2 q$ |
| Combined (weighted) entropy | $\sum \dfrac{n_i}{n} H(S_i)$ |
| Gini (course version) | $p^2 + q^2$ (বেশি মানে বেশি pure) |
| Chi-square (per class, per node) | $\sqrt{\dfrac{(\text{Actual}-\text{Expected})^2}{\text{Expected}}}$; Expected = node size × parent এর ratio। সব যোগ = total |

> **YES = Positive class** (TP মানে actual YES, predicted YES)।

---

## Question 1

**A Training Data Set has two Labels or Targets {Yes, No} and three features or predictors {X1, X2, X3}. The frequency table for X1 is {6 Yes, 2 No}, {3 Yes, 4 No} and {2 Yes, 3 No}. The frequency table for X2 is {10 Yes, 2 No} and {1 Yes, 7 No}. The frequency table for X3 is {9 Yes, 2 No} and {2 Yes, 7 No}. Which one of the three predictors will be selected by the OneR classifier?**

- A. X1
- B. X2
- C. X3
- D. Insufficient Data

### ✅ Answer: B. X2

### কিভাবে solve করবে

**Step 1:** OneR প্রতিটা partition এ majority class predict করে। তাই প্রতিটা partition এ **minority class এর সংখ্যা = ভুল (error)**।

**Step 2:** প্রতিটা predictor এর total error বের করো:

| Predictor | Partition | Majority (predict) | Error (minority) |
|---|---|---|---|
| X1 | {6 Yes, 2 No} | Yes | 2 |
| | {3 Yes, 4 No} | No | 3 |
| | {2 Yes, 3 No} | No | 2 |
| | | **Total** | **7 / 20** |
| X2 | {10 Yes, 2 No} | Yes | 2 |
| | {1 Yes, 7 No} | No | 1 |
| | | **Total** | **3 / 20** ← সবচেয়ে কম |
| X3 | {9 Yes, 2 No} | Yes | 2 |
| | {2 Yes, 7 No} | No | 2 |
| | | **Total** | **4 / 20** |

**Step 3:** সবচেয়ে কম error → **X2** (accuracy = 17/20 = 0.85)।

---

## Question 2

**Suppose, OneR selects an attribute as the best predictor which splits the original data set into following three partitions - {7 Yes, 12 No}, {10 Yes, 4 No} and {5 Yes, 11 No}. What is the value of FN (False Negative) for the OneR Classifier?**

- A. 09
- B. 11
- C. 12
- D. 15

### ✅ Answer: C. 12

### কিভাবে solve করবে

**Step 1:** প্রতিটা partition এ majority class predict করো।

**Step 2:** Confusion matrix এর cell গুলো ভরো (YES = positive):

| Partition | Predict | TP | FP | FN | TN |
|---|---|---|---|---|---|
| {7 Yes, 12 No} | **No** | 0 | 0 | **7** | 12 |
| {10 Yes, 4 No} | **Yes** | 10 | 4 | 0 | 0 |
| {5 Yes, 11 No} | **No** | 0 | 0 | **5** | 11 |
| **Total** | | 10 | 4 | **12** | 23 |

**Step 3:** FN = actual YES কিন্তু predicted NO = 7 + 5 = **12**

> FN শুধু সেই partition গুলো থেকে আসে যেখানে prediction **No** — ঐ partition এর Yes গুলোই FN।

---

## Question 3

**Suppose, OneR selects an attribute as the best predictor which splits the original data set into following three partitions - {10 Yes, 4 No}, {5 Yes, 6 No} and {4 Yes, 2 No}. What is the Precision of the OneR Classifier?**

- A. 0.60
- B. 0.65
- C. 0.667
- D. 0.70

### ✅ Answer: D. 0.70

### কিভাবে solve করবে

**Step 1:** প্রতিটা partition এ majority predict করো:

| Partition | Predict | TP | FP | FN | TN |
|---|---|---|---|---|---|
| {10 Yes, 4 No} | Yes | 10 | 4 | 0 | 0 |
| {5 Yes, 6 No} | **No** | 0 | 0 | 5 | 6 |
| {4 Yes, 2 No} | Yes | 4 | 2 | 0 | 0 |
| **Total** | | **14** | **6** | 5 | 6 |

**Step 2:** Precision = TP / (TP + FP) = 14 / (14 + 6) = 14/20 = **0.70**

> ভুল করার জায়গা: {5 Yes, 6 No} এ prediction **No**, তাই এর 5টা Yes কে TP ধরবে না (ওগুলো FN)।

---

## Question 4

**A Training data set has 50 instances in total, among which 10 are YES instances and the others are NO instances. What will be the F1-score of a ZeroR classifier on this Training Set?**

- A. 0.80
- B. 0.667
- C. Indefinite
- D. 0.20

### ✅ Answer: C. Indefinite

### কিভাবে solve করবে

**Step 1:** YES = 10, NO = 40 → majority = **NO**। ZeroR সবসময় **NO** predict করে।

**Step 2:** Confusion matrix: TP = 0, FP = 0, FN = 10, TN = 40

**Step 3:** Precision = TP/(TP+FP) = 0/0 → **undefined**। Recall = 0/10 = 0।
F1 = 2PR/(P+R) — P undefined, তাই F1 ও **Indefinite**।

> (Accuracy অবশ্য হিসাব করা যায় = 40/50 = 0.80 — option A একটা trap।)

---

## Question 5

**A Training data set has 20 instances in total, among which 05 are YES instances and the others are NO instances. What will be the Precision of a ZeroR classifier on this Training Set?**

- A. 0.75
- B. 0.667
- C. 0.33
- D. Indefinite

### ✅ Answer: D. Indefinite

### কিভাবে solve করবে

**Step 1:** YES = 5, NO = 15 → majority = **NO** → ZeroR সবসময় NO predict করে।

**Step 2:** কোনো YES predict হয় না → TP = 0, FP = 0।

**Step 3:** Precision = 0 / (0 + 0) = 0/0 → **Indefinite**।

> **Trick:** ZeroR যদি **NO** (negative) predict করে → Precision আর F1 সবসময় **Indefinite**। যদি YES majority হয় → Precision = YES/total, Recall = 1। (Option A 0.75 = accuracy, trap।)

---

## Question 6

**A Training Data Set has 10 YES instances and 20 NO instances. By using a feature X, the data set is splitted into Two sets S1 and S2. The set S1 has 8 YES and 5 NO instances, while the set S2 has 2 YES and 15 NO instances. What is the Total CHI-SQUARE value for the split with X?**

- A. 3.78
- B. 5.64
- C. 6.45
- D. 3.97

### ✅ Answer: B. 5.64

### কিভাবে solve করবে

**Step 1 — Parent ratio:** YES = 10/30 = 1/3, NO = 20/30 = 2/3

**Step 2 — S1 (8 + 5 = 13 instances):**
- Expected YES = 13 × 1/3 = 4.333, Expected NO = 13 × 2/3 = 8.667
- YES: $\sqrt{(8-4.333)^2/4.333} = \sqrt{13.44/4.333} = \sqrt{3.103} = 1.762$
- NO: $\sqrt{(5-8.667)^2/8.667} = \sqrt{13.44/8.667} = \sqrt{1.551} = 1.245$

**Step 3 — S2 (2 + 15 = 17 instances):**
- Expected YES = 17 × 1/3 = 5.667, Expected NO = 17 × 2/3 = 11.333
- YES: $\sqrt{(2-5.667)^2/5.667} = \sqrt{2.373} = 1.540$
- NO: $\sqrt{(15-11.333)^2/11.333} = \sqrt{1.186} = 1.089$

**Step 4 — Total:** 1.762 + 1.245 + 1.540 + 1.089 = **5.64**

---

## Question 7

**A Training Data Set has 8 YES instances and 5 NO instances. Now, We Split the Data Set using an attribute X into two new sets S1 and S2. The Set S1 has 5 YES and 2 NO instances, while the other set S2 has 3 YES and 3 NO instances. What is the combined Entropy after the split?**

- A. 0.83
- B. 0.88
- C. 0.93
- D. 0.98

### ✅ Answer: C. 0.93

### কিভাবে solve করবে

**Step 1 — H(S1), 7 instances (5 Yes, 2 No):**
$H(S_1) = -\frac57\log_2\frac57 - \frac27\log_2\frac27 = 0.347 + 0.516 = 0.863$

**Step 2 — H(S2), 6 instances (3 Yes, 3 No):**
3:3 সমান → $H(S_2) = 1$ (maximum impurity)

**Step 3 — Weighted (combined) entropy:**
$\frac{7}{13}(0.863) + \frac{6}{13}(1) = 0.465 + 0.462 = $ **0.93**

> Check: split এর আগে parent (8Y, 5N) এর entropy = 0.961। Split এর পরে 0.93 — কমেছে, তাই Information Gain = 0.961 − 0.926 = 0.035।

---

## Question 8

**A Training Data Set has two Labels YES and NO and two features X and Y. If we split the Data Set using the feature Y, we get two sets S1 and S2. If we split using the attribute X, we get two sets {6YES, 3NO} and {6YES, 3NO}. Which of the two features will be picked by REDUCTION IN VARIANCE to split the tree?**

- A. X
- B. Y
- C. either of X or Y (Both equivalent)
- D. Insufficient data

### ✅ Answer: D. Insufficient data

### কিভাবে solve করবে

**Step 1:** X এর split এর data দেওয়া আছে: {6Y,3N}, {6Y,3N} → variance হিসাব করা যায়।

**Step 2:** Y এর split এ শুধু বলা আছে "S1 and S2" — **কোনো YES/NO count দেওয়া নেই**।

**Step 3:** Y এর variance হিসাব করা সম্ভব না → X আর Y compare করা যাবে না → **Insufficient data**।

> Trick question — হিসাব শুরু করার আগে দেখো সব data দেওয়া আছে কিনা।

---

## Question 9

**A Training Data Set has two Labels YES and NO and two features X and Y. If we split the Data Set using the feature Y, we get two sets {8YES, 4NO} and {4YES, 2NO}. If we split using the attribute X, we get two sets {6YES, 3NO} and {6YES, 3NO}. Which of the two features will be picked by CHI-SQUARE INDEX to split the tree?**

- A. X
- B. Y
- C. either of X or Y (Both equivalent)
- D. Insufficient data

### ✅ Answer: C. either of X or Y (Both equivalent)

### কিভাবে solve করবে

**Step 1 — Parent:** 8 + 4 = 12 YES, 4 + 2 = 6 NO → ratio YES : NO = 2/3 : 1/3

**Step 2 — Split on Y:**
- {8Y, 4N}: Expected = 12 × 2/3 = 8 Y, 4 N → Actual = Expected → χ = 0
- {4Y, 2N}: Expected = 6 × 2/3 = 4 Y, 2 N → χ = 0
- Total = **0**

**Step 3 — Split on X:**
- {6Y, 3N}: Expected = 9 × 2/3 = 6 Y, 3 N → χ = 0 (দুইবার)
- Total = **0**

**Step 4:** দুইটাই 0 → সমান → **Both equivalent**।

> Shortcut: child এর ratio = parent এর ratio হলে chi-square **0**।

---

## Question 10

**A Training Data Set has 1 YES instances and 24 NO instances. What is the Gini Index of the Data Set?**

- A. 0.89
- B. 0.92
- C. 0.95
- D. 0.97

### ✅ Answer: B. 0.92

### কিভাবে solve করবে

**Step 1:** Total = 25 → p(YES) = 1/25 = 0.04, q(NO) = 24/25 = 0.96

**Step 2:** Course এর Gini = $p^2 + q^2$ = 0.04² + 0.96² = 0.0016 + 0.9216 = **0.9232 ≈ 0.92**

> ⚠️ অন্য বইয়ে Gini impurity = $1 - (p^2+q^2) = 0.077$ — কিন্তু option এ নেই। এই course **$p^2 + q^2$** version use করে।

---

## Question 11

**A Training Data Set has 20 YES instances and 25 NO instances. By using a feature X, the data set is splitted into Two sets S1 and S2. The set S1 has ALL the YES instances, while the set S2 has ALL the NO instances. What is the Total CHI-SQUARE value for the set S1?**

- A. 5.49
- B. 5.25
- C. 7.06
- D. 6.07

### ✅ Answer: C. 7.06

### কিভাবে solve করবে

**Step 1 — Parent ratio:** YES = 20/45 = 0.444, NO = 25/45 = 0.556

**Step 2 — S1 = সব YES = {20 Yes, 0 No}, 20 instances:**
- Expected YES = 20 × 0.444 = 8.889, Expected NO = 20 × 0.556 = 11.111

**Step 3 — Chi values:**
- YES: $\sqrt{(20-8.889)^2/8.889} = \sqrt{123.46/8.889} = \sqrt{13.889} = 3.727$
- NO: $\sqrt{(0-11.111)^2/11.111} = \sqrt{11.111} = 3.333$

**Step 4:** Total for S1 = 3.727 + 3.333 = **7.06**

> প্রশ্নে "for the **set S1**" — তাই শুধু S1 এর দুইটা value যোগ, S2 না।

---

## Answer Summary

| Q | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Ans | B | C | D | C | D | B | C | D | C | B | C |
| Value | X2 | 12 | 0.70 | Indefinite | Indefinite | 5.64 | 0.93 | Insufficient | Both | 0.92 | 7.06 |

এরপর practice: [02-Extra-Practice-Questions](02-Extra-Practice-Questions.md)
