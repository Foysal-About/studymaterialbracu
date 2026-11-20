# Extra Practice Questions (Moodle-style MCQ)

> নিজে খাতায় solve করো, তারপর **"Answer দেখো"** click করে মিলাও।
> সব answer Python দিয়ে verify করা। Formula গুলো: [01-Moodle-Quiz Solutions এর cheat sheet](01-Moodle-Quiz-ZeroR-OneR-CM-DT-Solutions.md)

---

## Part A — ZeroR, OneR, Confusion Matrix, Decision Tree (3 marks)

### E1. OneR predictor selection
A dataset has 9 YES and 9 NO. Frequency tables: **X1** = {4Y,1N}, {2Y,5N}, {3Y,3N}; **X2** = {7Y,2N}, {2Y,7N}; **X3** = {5Y,4N}, {4Y,5N}. Which predictor does OneR select, and what is its accuracy?

A. X1, 0.667 B. X2, 0.778 C. X3, 0.556 D. X2, 0.667

<details><summary>Answer দেখো</summary>

Errors: X1 = 1 + 2 + 3 = 6, **X2 = 2 + 2 = 4**, X3 = 4 + 4 = 8 → X2।
Accuracy = (18 − 4)/18 = 14/18 = **0.778** → **B**

({3Y,3N} এ tie — যেকোনোটা predict করলেও error 3।)
</details>

### E2. OneR — Recall and F1
OneR partitions: {12Y,3N}, {4Y,9N}, {2Y,5N}. Find Recall and F1.

A. 0.667, 0.727 B. 0.80, 0.727 C. 0.667, 0.80 D. 0.75, 0.70

<details><summary>Answer দেখো</summary>

Predictions: Yes, No, No → TP = 12, FP = 3, FN = 4 + 2 = 6, TN = 9 + 5 = 14
Precision = 12/15 = 0.80, Recall = 12/18 = **0.667**
F1 = 2(0.8)(0.667)/(0.8 + 0.667) = **0.727** → **A**
(Accuracy = 26/35 = 0.743)
</details>

### E3. ZeroR when YES is the majority
40 instances: 30 YES, 10 NO. What is the F1-score of ZeroR?

A. 0.75 B. Indefinite C. 0.857 D. 1.00

<details><summary>Answer দেখো</summary>

ZeroR → সবসময় YES। TP = 30, FP = 10, FN = 0
Precision = 30/40 = 0.75, Recall = 30/30 = 1
F1 = 2(0.75)(1)/(1.75) = **0.857** → **C**

(Q4/Q5 এর উল্টো — এখানে majority YES, তাই indefinite না।)
</details>

### E4. Entropy of a dataset
9 YES, 5 NO. Entropy = ?

A. 0.863 B. 0.940 C. 0.985 D. 1.00

<details><summary>Answer দেখো</summary>

$H = -\frac{9}{14}\log_2\frac{9}{14} - \frac{5}{14}\log_2\frac{5}{14} = 0.410 + 0.530 = $ **0.940** → **B**
</details>

### E5. Information Gain
Parent: 9 YES, 5 NO. Split: S1 = {6Y, 2N}, S2 = {3Y, 3N}. Information Gain = ?

A. 0.048 B. 0.152 C. 0.247 D. 0.029

<details><summary>Answer দেখো</summary>

H(S1) = 0.811, H(S2) = 1.000
Weighted = (8/14)(0.811) + (6/14)(1) = 0.464 + 0.429 = 0.892
IG = 0.940 − 0.892 = **0.048** → **A** (Play-Golf এর **Windy** attribute এর মতো)
</details>

### E6. Gini of a split (course formula $p^2+q^2$)
S1 = {6Y, 2N}, S2 = {3Y, 3N}. Weighted Gini of the split = ?

A. 0.429 B. 0.500 C. 0.571 D. 0.625

<details><summary>Answer দেখো</summary>

Gini(S1) = (6/8)² + (2/8)² = 0.5625 + 0.0625 = 0.625
Gini(S2) = 0.5² + 0.5² = 0.5
Weighted = (8/14)(0.625) + (6/14)(0.5) = 0.357 + 0.214 = **0.571** → **C**
</details>

### E7. Total Chi-square of a split
Parent: 12 YES, 8 NO. S1 = {9Y, 1N}, S2 = {3Y, 7N}. Total chi-square = ?

A. 4.24 B. 5.45 C. 6.00 D. 3.67

<details><summary>Answer দেখো</summary>

Parent ratio 0.6 : 0.4. দুইটা set ই 10 instance → Expected YES = 6, NO = 4
- S1: √((9−6)²/6) + √((1−4)²/4) = 1.225 + 1.5 = 2.725
- S2: √((3−6)²/6) + √((7−4)²/4) = 1.225 + 1.5 = 2.725

Total = **5.45** → **B**
</details>

### E8. F-beta score
Precision = 0.8, Recall = 0.5. F2 = ?

A. 0.615 B. 0.541 C. 0.667 D. 0.588

<details><summary>Answer দেখো</summary>

$F_\beta = \dfrac{(1+\beta^2)PR}{\beta^2 P + R} = \dfrac{5(0.8)(0.5)}{4(0.8)+0.5} = \dfrac{2}{3.7} = $ **0.541** → **B**
(β = 2 → recall কে বেশি গুরুত্ব দেয়, তাই F1 = 0.615 এর চেয়ে কম।)
</details>

### E9. Specificity
TP = 40, FP = 10, FN = 20, TN = 30. Specificity = ?

A. 0.667 B. 0.80 C. 0.75 D. 0.60

<details><summary>Answer দেখো</summary>

Specificity = TN/(TN + FP) = 30/40 = **0.75** → **C**
</details>

### E10. Pure node
A node has 0 YES and 6 NO. Entropy and Gini ($p^2+q^2$) are:

A. 1, 0 B. 0, 1 C. 0, 0 D. 1, 0.5

<details><summary>Answer দেখো</summary>

Pure node → Entropy = **0**, Gini (p²+q²) = 0² + 1² = **1** → **B**
</details>

---

## Part B — Anomaly, Perceptron, Regression, Chi-square test (bonus)

### E11. Anomaly — determinant and inverse
$\Sigma = \begin{bmatrix}9 & 3\\ 3 & 5\end{bmatrix}$. Find $\lvert\Sigma\rvert$ and $\Sigma^{-1}$.

<details><summary>Answer দেখো</summary>

$\lvert\Sigma\rvert = 9(5) - 3(3) = 45 - 9 = $ **36**

$\Sigma^{-1} = \dfrac{1}{36}\begin{bmatrix}5 & -3\\ -3 & 9\end{bmatrix} = \begin{bmatrix}0.139 & -0.083\\ -0.083 & 0.25\end{bmatrix}$
</details>

### E12. Perceptron — one update
$w = [1, 1, 1]$, normalized augmented sample $y = [1, -2, 1]$, $\alpha = 0.5$. Is $y$ misclassified? What is the new $w$?

<details><summary>Answer দেখো</summary>

$w^Ty = 1 - 2 + 1 = 0 \le 0$ → **misclassified** (boundary এর উপর = correct না)
$w_{new} = w + \alpha y = [1, 1, 1] + 0.5[1, -2, 1] = $ **[1.5, 0, 1.5]**
</details>

### E13. Least-squares regression line
x = [1, 2, 3, 4], y = [2, 4, 5, 7]. Find the line and predict y at x = 5.

<details><summary>Answer দেখো</summary>

$\bar x = 2.5,\ \bar y = 4.5$
$\sum(x-\bar x)(y-\bar y) = 3.75 + 0.25 + 0.25 + 3.75 = 8$, $\sum(x-\bar x)^2 = 5$
$m = 8/5 = 1.6$, $c = 4.5 - 1.6(2.5) = 0.5$ → **ŷ = 1.6x + 0.5**
x = 5 → ŷ = **8.5**
</details>

### E14. Gradient descent — one single-mode step
$\theta_0 = 0,\ \theta_1 = 0,\ \alpha = 0.01$, first sample (x = 1, y = 2). Find new $\theta_0, \theta_1$.

<details><summary>Answer দেখো</summary>

Prediction = 0 + 0(1) = 0, error = 0 − 2 = −2
$\theta_0 = 0 - 0.01(-2) = $ **0.02**, $\theta_1 = 0 - 0.01(-2)(1) = $ **0.02**
</details>

### E15. Chi-square test of independence (EDA)
100 students: Male — 30 Pass, 20 Fail; Female — 20 Pass, 30 Fail. At α = 0.05, is result independent of gender?

<details><summary>Answer দেখো</summary>

Expected প্রতিটা cell = (50 × 50)/100 = 25
$\chi^2 = \sum \frac{(O-E)^2}{E} = 4 \times \frac{(5)^2}{25} = $ **4.0**
df = (2−1)(2−1) = 1, critical value = 3.841
4.0 > 3.841 → **Reject H₀** → result আর gender **independent না** (related)।
</details>
