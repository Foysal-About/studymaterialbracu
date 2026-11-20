# Moodle Quiz — ZeroR, OneR, Confusion Matrix, Decision Tree (Solved)

> Mid syllabus: "ZeroR, OneR, CM, DT ⇒ Math (**Moodle like**), 3 marks" — তাই exam এর প্রশ্ন ঠিক এই ধরনের হবে।
> প্রতিটা প্রশ্নের answer key এর answer আমরা course এর formula দিয়ে নিজে হিসাব করে **verify** করেছি — 11টাই সঠিক ✅
> Theory বুঝতে: [05-ZeroR-OneR-Confusion-Matrix-Decision-Tree](../notes/05-ZeroR-OneR-Confusion-Matrix-Decision-Tree.md)

## Formula Cheat Sheet (এই quiz এর জন্য যা লাগবে)

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
| Gini (course/Analytics Vidhya version) | $p^2 + q^2$ (বেশি মানে বেশি pure) |
| Chi-square (per class, per node) | $\sqrt{\dfrac{(\text{Actual}-\text{Expected})^2}{\text{Expected}}}$, Expected = node size × parent এর ratio। সব যোগ করলে total |

> **YES = Positive class** ধরা হয় (TP মানে actual YES, predicted YES)।

---

## Q1. OneR — কোন predictor select হবে?

> Frequency tables: **X1** = {6Y,2N}, {3Y,4N}, {2Y,3N}; **X2** = {10Y,2N}, {1Y,7N}; **X3** = {9Y,2N}, {2Y,7N}
> A. X1 B. X2 C. X3 D. Insufficient Data

**Solution:** প্রতিটা partition এ minority = error.

| Predictor | Partition errors | Total error |
|---|---|---|
| X1 | 2 + 3 + 2 | **7** |
| X2 | 2 + 1 | **3** ← সবচেয়ে কম |
| X3 | 2 + 2 | **4** |

**Answer: B (X2)** ✅ — OneR সবচেয়ে কম error এর predictor নেয়।

---

## Q2. OneR — FN (False Negative) কত?

> Partitions: {7Y,12N}, {10Y,4N}, {5Y,11N}
> A. 09 B. 11 C. 12 D. 15

**Solution:** প্রতিটা partition এ majority predict করো, তারপর CM বানাও।

| Partition | Predict | TP | FP | FN | TN |
|---|---|---|---|---|---|
| {7Y, 12N} | **No** | 0 | 0 | **7** | 12 |
| {10Y, 4N} | **Yes** | 10 | 4 | 0 | 0 |
| {5Y, 11N} | **No** | 0 | 0 | **5** | 11 |
| **Total** | | 10 | 4 | **12** | 23 |

FN = actual YES কিন্তু predicted NO = 7 + 5 = **12**

**Answer: C (12)** ✅

---

## Q3. OneR — Precision কত?

> Partitions: {10Y,4N}, {5Y,6N}, {4Y,2N}
> A. 0.60 B. 0.65 C. 0.667 D. 0.70

| Partition | Predict | TP | FP |
|---|---|---|---|
| {10Y, 4N} | Yes | 10 | 4 |
| {5Y, 6N} | **No** | 0 | 0 |
| {4Y, 2N} | Yes | 4 | 2 |
| **Total** | | **14** | **6** |

Precision = 14 / (14 + 6) = 14/20 = **0.70**

**Answer: D (0.70)** ✅

> ভুল করার জায়গা: {5Y,6N} partition এ prediction **No**, তাই এর 5টা Yes কে TP ধরবে না (ওগুলো FN)।

---

## Q4. ZeroR — F1-score কত? (50 instances, 10 YES, 40 NO)

> A. 0.80 B. 0.667 C. Indefinite D. 0.20

ZeroR majority predict করে → সবসময় **NO**।
তাহলে কোনো YES predict হয় না → TP = 0, FP = 0।

Precision = 0 / (0 + 0) = **0/0 → undefined** → F1 ও undefined।

**Answer: C (Indefinite)** ✅

---

## Q5. ZeroR — Precision কত? (20 instances, 5 YES, 15 NO)

> A. 0.75 B. 0.667 C. 0.33 D. Indefinite

Majority = NO → সব prediction NO → TP + FP = 0 → Precision = 0/0।

**Answer: D (Indefinite)** ✅

> **Trick:** ZeroR যদি **NO** (negative) predict করে, তাহলে Precision আর F1 সবসময় **Indefinite**। যদি YES majority হয়, তাহলে Precision = YES/total, Recall = 1।

---

## Q6. Total Chi-square for split X

> Parent: 10 YES, 20 NO। S1 = {8Y, 5N}, S2 = {2Y, 15N}
> A. 3.78 B. 5.64 C. 6.45 D. 3.97

Parent ratio: YES = 10/30 = 1/3, NO = 2/3

**S1 (13 instances):** Expected YES = 13/3 = 4.333, Expected NO = 8.667
- YES: $\sqrt{(8-4.333)^2/4.333} = \sqrt{13.44/4.333} = \sqrt{3.103} = 1.762$
- NO: $\sqrt{(5-8.667)^2/8.667} = \sqrt{1.551} = 1.245$

**S2 (17 instances):** Expected YES = 5.667, Expected NO = 11.333
- YES: $\sqrt{(2-5.667)^2/5.667} = \sqrt{2.373} = 1.540$
- NO: $\sqrt{(15-11.333)^2/11.333} = \sqrt{1.186} = 1.089$

Total = 1.762 + 1.245 + 1.540 + 1.089 = **5.64**

**Answer: B (5.64)** ✅

---

## Q7. Combined Entropy after split

> Parent: 8 YES, 5 NO। S1 = {5Y, 2N}, S2 = {3Y, 3N}
> A. 0.83 B. 0.88 C. 0.93 D. 0.98

- $H(S_1) = -\frac57\log_2\frac57 - \frac27\log_2\frac27 = 0.347 + 0.516 = 0.863$
- $H(S_2) = 1$ (3/3 সমান ভাগ → সবচেয়ে impure)
- Combined = $\frac{7}{13}(0.863) + \frac{6}{13}(1) = 0.465 + 0.462 = $ **0.93**

**Answer: C (0.93)** ✅

---

## Q8. Reduction in Variance — X নাকি Y?

> Y split এর sets এর **কোনো data দেওয়া নেই** (শুধু "S1 and S2")। X = {6Y,3N}, {6Y,3N}
> A. X B. Y C. Either D. Insufficient data

Y এর split এর class count জানা নেই → Y এর variance হিসাব করা সম্ভব না → compare করা যাবে না।

**Answer: D (Insufficient data)** ✅

> Trick question — প্রশ্ন ভালো করে পড়ো, সব data দেওয়া আছে কিনা।

---

## Q9. Chi-square index — X নাকি Y?

> Y → {8Y,4N}, {4Y,2N}; X → {6Y,3N}, {6Y,3N}
> A. X B. Y C. Either (Both equivalent) D. Insufficient data

Parent = 12 YES, 6 NO → ratio 2/3 : 1/3

- Y এর দুইটা set: 8/12 = 2/3, 4/6 = 2/3 → parent এর **একই ratio**
- X এর দুইটা set: 6/9 = 2/3 → parent এর **একই ratio**

Actual = Expected সবখানে → সব chi-square = 0। দুইটাই 0 → সমান।

**Answer: C (Both equivalent)** ✅

> Shortcut: child এর ratio parent এর সমান হলে chi-square **0** — split কোনো কাজের না।

---

## Q10. Gini Index of the dataset (1 YES, 24 NO)

> A. 0.89 B. 0.92 C. 0.95 D. 0.97

Course এর Gini = $p^2 + q^2$:

$p = 1/25 = 0.04,\ q = 24/25 = 0.96$
Gini = $0.04^2 + 0.96^2 = 0.0016 + 0.9216 = $ **0.9232 ≈ 0.92**

**Answer: B (0.92)** ✅

> ⚠️ অন্য বইয়ে Gini impurity = $1 - (p^2+q^2) = 0.077$। কিন্তু option এ 0.077 নেই — এই course **$p^2+q^2$** (purity) version use করে।

---

## Q11. Chi-square for set S1 (all YES)

> Parent: 20 YES, 25 NO। S1 = সব YES = {20Y, 0N}
> A. 5.49 B. 5.25 C. 7.06 D. 6.07

Parent ratio: YES = 20/45 = 0.444, NO = 25/45 = 0.556
S1 এ 20 instances → Expected YES = 8.889, Expected NO = 11.111

- YES: $\sqrt{(20-8.889)^2/8.889} = \sqrt{123.46/8.889} = \sqrt{13.889} = 3.727$
- NO: $\sqrt{(0-11.111)^2/11.111} = \sqrt{11.111} = 3.333$

Total (S1) = 3.727 + 3.333 = **7.06**

**Answer: C (7.06)** ✅

---

## Answer Summary

| Q | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Ans | B | C | D | C | D | B | C | D | C | B | C |

এরপর practice: [02-Extra-Practice-Questions](02-Extra-Practice-Questions.md)
