# Anomaly Detection (4 Marks) + Perceptron (4 Marks) — Question & Answer + All Formulas

> Sir বলেছেন: **Anomaly Detection = 4 marks**, **Perceptron = 4 marks** — দুইটাই **"Mainly Math + Short Que"**।
> এই file এ course এর Drive folder এর এই files থেকে **সব formula** আর **exam-style question + full answer** দেওয়া:
> `15 Anomaly Detection Theory v3`, `15b How to Find Covariance Matrix`, `15c How Find Inverse of Matrix`, `15d Math Step-by-Step for Anomaly Detection v2`, `15e Question List (Anomaly Detection) v2`, `Perceptron Tutorial (Final) v4`।
> সব math Python এ verify করা ✅। নতুন practice math গুলোও (যেগুলো Drive এ নেই) একই method এ বানানো।
> বিস্তারিত theory: [03-Anomaly-Detection](../notes/03-Anomaly-Detection.md), [04-Perceptron](../notes/04-Perceptron.md)

**Teacher's hints (Lecture-07):**
- Anomaly: **15e question list**, **15d step-by-step**। "**Cov matrix দেওয়া থাকলে পরের part করতে হবে, অথবা শুধু Cov matrix পর্যন্ত**"।
- Perceptron: **Linear boundary**; **limitation – linear**। "**Many at a time → more stable, better convergence**"।

---

## সূচি

- [Part 1 — Formula Sheet](#part-1--formula-sheet)
- [Part 2 — Anomaly Detection: Short Questions](#part-2--anomaly-detection-short-questions)
- [Part 3 — Anomaly Detection: Math Questions](#part-3--anomaly-detection-math-questions)
- [Part 4 — Perceptron: Short Questions](#part-4--perceptron-short-questions)
- [Part 5 — Perceptron: Math Questions](#part-5--perceptron-math-questions)
- [Part 6 — Answer Key + Exam Tips](#part-6--answer-key--exam-tips)

---

## Part 1 — Formula Sheet

### 1A. Anomaly Detection Formulas

| # | কী | Formula | Symbol এর মানে |
|---|---|---|---|
| A1 | Mean (প্রতিটা feature এর) | $`\mu_j = \dfrac{1}{m}\sum_{i=1}^{m} x_j^{(i)}`$ | m = training example সংখ্যা, j = কোন feature |
| A2 | Variance | $`\sigma_j^2 = \dfrac{1}{m}\sum_{i=1}^{m}(x_j^{(i)}-\mu_j)^2`$ | course এ **1/m** (1/(m−1) ও চলে, পার্থক্য সামান্য) |
| A3 | Standard deviation | $`\sigma_j = \sqrt{\sigma_j^2}`$ | spread / width |
| A4 | Gaussian (1 feature) | $`p(x;\mu,\sigma^2) = \dfrac{1}{\sigma\sqrt{2\pi}}\, e^{-\frac{(x-\mu)^2}{2\sigma^2}}`$ | প্রথম অংশ = height factor, $`e^{...}`$ অংশ = μ থেকে দূরত্বের penalty |
| A5 | Univariate model (n features) | $`p(x) = \prod_{j=1}^{n} p(x_j;\mu_j,\sigma_j^2)`$ | **প্রতিটা feature আলাদা Gaussian, তারপর গুণ** (independence assumption) |
| A6 | Covariance matrix | $`\Sigma = \dfrac{1}{m}\sum_{i=1}^{m}(x^{(i)}-\mu)(x^{(i)}-\mu)^T = \dfrac{a^T a}{m}`$ | $`a = M - \mu`$ (প্রতিটা row থেকে mean বাদ); Σ হলো **n×n** |
| A7 | 2×2 Determinant | $`\lvert\Sigma\rvert = ad - bc`$ for $`\begin{bmatrix}a&b\\c&d\end{bmatrix}`$ | একটা single number |
| A8 | 2×2 Inverse | $`\Sigma^{-1} = \dfrac{1}{ad-bc}\begin{bmatrix}d&-b\\-c&a\end{bmatrix}`$ | diagonal **swap**, off-diagonal **sign change**, ÷ det |
| A9 | Multivariate Gaussian | $`p(x) = \dfrac{1}{(2\pi)^{n/2}\lvert\Sigma\rvert^{1/2}}\exp\!\left(-\tfrac12 (x-\mu)^T\Sigma^{-1}(x-\mu)\right)`$ | n = feature সংখ্যা; n = 2 হলে $`(2\pi)^{n/2} = 2\pi`$ |
| A10 | Decision rule | $`p(x_{test}) < \varepsilon`$ → **Anomaly**; $`p(x_{test}) \ge \varepsilon`$ → **Normal** | ε = threshold probability |

**Covariance matrix এর গঠন (2 features):**
```math
\Sigma = \begin{bmatrix}\sigma_1^2 & \text{cov}(x_1,x_2)\\ \text{cov}(x_1,x_2) & \sigma_2^2\end{bmatrix}, \qquad \text{cov}(x_1,x_2) = \frac1m\sum (x_1^{(i)}-\mu_1)(x_2^{(i)}-\mu_2)
```
- Diagonal = variance, off-diagonal = covariance, matrix **symmetric**।
- Off-diagonal সব 0 হলে multivariate = univariate (দুইটা model identical)।

**Dimension check (A9 এর exponent):** $`(1\times n)\cdot(n\times n)\cdot(n\times 1) = (1\times 1)`$ → একটা scalar number।

**3×3 Inverse (15c):** Minors → Cofactors (sign pattern $`+\,-\,+`$) → Adjugate (transpose) → $`\times\frac{1}{\det}`$। Exam এ সাধারণত 2×2।

### 1B. Perceptron Formulas

| # | কী | Formula | মানে |
|---|---|---|---|
| P1 | Discriminant function | $`g(x) = w^T x + w_0`$ | straight line $`ax_1 + bx_2 + c = 0`$ এর মতো; $`w^T = [a\ b]`$, $`w_0 = c`$ |
| P2 | Decision rule | $`g(x) > 0 \Rightarrow \omega_1`$; $`g(x) < 0 \Rightarrow \omega_2`$; $`g(x) = 0`$ → **decision boundary** | |
| P3 | Augmented vector | $`x = [x_1\ x_2\ 1]^T`$, $`w = [w_1\ w_2\ w_0]`$ → $`g = w^T x`$ | শেষের **1** দিয়ে $`w_0`$ (constant/bias) vector এর ভিতরে ঢোকে |
| P4 | φ (high-dim projection) | $`y = \varphi(x) = [x_1^2,\ x_2^2,\ x_1x_2,\ x_1,\ x_2,\ 1]^T`$ | 2-D → 6-D; **শেষ component সবসময় 1** |
| P5 | Class normalization | $`\omega_2`$ এর সব pattern: $`y \leftarrow -y`$ | তাহলে সবার জন্য শুধু একটা condition: $`g(y) > 0`$ |
| P6 | Update (One at a time) | $`g(y) = w^T y \le 0`$ হলে **সাথে সাথে** $`w \leftarrow w + \alpha\, y`$ | misclassified y এর α অংশ যোগ |
| P7 | Update (Many at a time) | iteration শেষে **একবার**: $`w \leftarrow w + \alpha \sum y_{mis}`$ | পুরো iteration এ **একই w**; সব misclassified y যোগ করে update |
| P8 | Learning rate | $`0 < \alpha \le 1`$; সাধারণত ~0.5 অথবা $`\alpha = 1/n`$ | α = 0 → কোনো update হয় না; α > 1 → unstable |
| P9 | Initial weights | ছোট random মান (যেমন $`[-1, 1]`$) বা সব 1 | **সব 0 দেবে না** |
| P10 | Convergence | একটা full iteration এ **কোনো y misclassified না** (সব $`g(y) > 0`$, w বদলায়নি) → stop | |

> **g(y) = 0 হলে?** Normalization এর পর condition হলো $`g(y) > 0`$। তাই $`g = 0`$ মানে point টা line এর উপর → **misclassified ধরে update করো**।

---

## Part 2 — Anomaly Detection: Short Questions

> 15e question list থেকে (★ = বেশি important, ★★★ = অবশ্যই)। প্রতিটা answer 1–2 marks এর জন্য যথেষ্ট।

### SA1. What do you understand by Anomaly detection? What are its applications? *(15e, Theory p.1, 3–4)*

**Answer:** Anomaly detection হলো এমন data point খুঁজে বের করা যেটা **normal data থেকে খুব আলাদা**। Normal data দিয়ে একটা probability model $`p(x)`$ বানানো হয়; নতুন point এর $`p(x_{test}) < \varepsilon`$ হলে সেটা **anomaly/outlier**।
- এটা মূলত **unsupervised** (anomaly label থাকে না), কিন্তু normal points "supervisor" এর মতো কাজ করে — কী anomaly **না** সেটা জানায়।

**Applications:**
1. **Fraud detection** — login, credit card usage (login location, frequency, spending) অস্বাভাবিক হলে block
2. **Manufacturing / QA** — aircraft engine (heat, vibration) faulty কিনা
3. **Data center monitoring** — memory use, disk access, CPU load, network traffic দেখে কোন machine fail করবে আগে ধরা

### SA2. What is the role of Gaussian distribution in anomaly detection? Explain the role of μ and σ.

**Answer:** Normal data কে **Gaussian (normal) distribution** $`x \sim N(\mu, \sigma^2)`$ দিয়ে model করা হয়। Gaussian curve থেকে যেকোনো x এর probability পাওয়া যায় — μ এর কাছে probability বেশি, দূরে গেলে কমে। খুব কম probability = anomaly।
- **μ (mean)** — curve এর **peak/center** কোথায়।
- **σ (standard deviation)** — curve এর **width/spread**। σ ছোট হলে curve সরু ও উঁচু, বড় হলে চওড়া ও নিচু। (Area সবসময় = 1।)

### SA3. ★★★ Write the formal anomaly detection algorithm using the uni-variate Gaussian model. *(Theory p.9)*

**Answer:**
1. **Choose features** $`x_j`$ যেগুলো anomaly বুঝতে সাহায্য করবে।
2. **Fit parameters** training set $`\{x^{(1)},\dots,x^{(m)}\}`$ থেকে, প্রতিটা feature j = 1…n এর জন্য:
   $`\mu_j = \frac1m\sum_i x_j^{(i)}`$, $`\quad \sigma_j^2 = \frac1m\sum_i (x_j^{(i)}-\mu_j)^2`$
3. **Given new example x, compute**
   $`p(x) = \prod_{j=1}^{n}\frac{1}{\sqrt{2\pi}\,\sigma_j}\exp\!\left(-\frac{(x_j-\mu_j)^2}{2\sigma_j^2}\right)`$
4. **Decide:** $`p(x) < \varepsilon`$ → **Anomaly**; otherwise **Normal**।

### SA4. ★ What are the three considerations with the uni-variate Gaussian anomaly detection algorithm? *(Theory p.9)*

**Answer:**
1. **How to choose features** — এমন feature যেগুলো anomaly তে অস্বাভাবিক বড়/ছোট মান নেয়।
2. **How to fit parameters** — প্রতিটা feature এর $`\mu_j`$ ও $`\sigma_j^2`$ হিসাব করা (vectorized হলে ভালো)।
3. **How to compute p(x) for the test point** — Gaussian formula দিয়ে, তারপর ε এর সাথে তুলনা।

### SA5. Briefly describe the issues for developing and evaluating an anomaly detection system (labelling, splitting, evaluation, choosing ε).

**Answer:**
- **Data labelling:** normal → $`y = 0`$, anomalous → $`y = 1`$। Training set এ শুধু (বা প্রায় শুধু) normal data।
- **Splitting** (example: 10,000 good + 20 flawed engines, ratio 3:1:1):
  - Training: 6000 good ($`y=0`$)
  - CV: 2000 good + 10 anomalous
  - Test: 2000 good + 10 anomalous
  - CV আর Test এ **একই anomaly reuse করা bad practice**।
- **Evaluation:** data খুব skewed ($`y=0`$ অনেক বেশি), তাই **accuracy ভালো metric না** (সবসময় "normal" বললেও 2000/2010 ≈ 99.5%)। বরং **TP/FP/FN/TN, Precision, Recall, F1-score** use করো।
- **Choosing ε:** CV set এ বিভিন্ন ε try করো, যেটা **F1-score maximize** করে সেটা নাও; final evaluation **test set** এ।

### SA6. ★ When to choose an anomaly detection algorithm vs a supervised learning algorithm? *(Theory p.13–14)*

| Anomaly Detection use করো যখন | Supervised Learning use করো যখন |
|---|---|
| Positive (anomaly) example **খুব কম** (0–20), negative (normal) **অনেক বেশি** | Positive আর negative **দুইটাই যথেষ্ট** |
| Anomaly **অনেক ধরনের** — ভবিষ্যতের anomaly আগের গুলোর মতো নাও হতে পারে | ভবিষ্যতের positive গুলো training এর positive এর **মতোই দেখতে** |
| Example: **fraud detection, manufacturing (engine), data center monitoring** | Example: **spam email, weather prediction, cancer classification** |

> Spam অনেক ধরনের হলেও common keyword থাকে আর positive example যথেষ্ট — তাই supervised।

### SA7. ★ How to deal with non-Gaussian features? *(Theory p.14–15)*

**Answer:** Feature এর **histogram plot** করে দেখো Gaussian (bell-shaped) কিনা। না হলে **transformation** দিয়ে Gaussian এর মতো বানাও:
- $`x \to \log(x)`$
- $`x \to \log(x + c)`$ (x negative/0 হলে; c adjust করো)
- $`x \to x^{1/2},\ x^{1/3},\ x^{1/k}`$

### SA8. ★ Give an example of feature engineering. What are the assumptions of the univariate model? *(Theory p.16–18, p.8)*

**Answer (feature engineering):** Data center এ $`x_3`$ = CPU load, $`x_4`$ = network traffic — সাধারণত একসাথে বাড়ে-কমে। নতুন feature $`x_5 = \dfrac{\text{CPU load}}{\text{network traffic}}`$:
- CPU high + network low → **infinite loop/deadlock** (machine fail করবে)
- CPU low + network high → **botnet/malicious host**

দুইটা ক্ষেত্রেই $`x_5`$ অস্বাভাবিক হয়ে যায়, তাই anomaly ধরা পড়ে। (Error analysis: CV set এ যেগুলো ভুল হয়েছে সেগুলো দেখে নতুন feature বানানো।)

**Assumptions:** (1) প্রতিটা feature **Gaussian distributed**; (2) features **independent** (তাই আলাদা probability গুণ করা যায়)।

### SA9. ★★★ Demonstrate an example where the multivariate model detects an outlier but the univariate model fails. *(Theory p.17–18)*

**Answer:** CPU load ($`x_1`$) আর memory use ($`x_2`$) সাধারণত **positively correlated** (একসাথে বাড়ে)। একটা test machine এর **memory high কিন্তু CPU low**।
- **Univariate:** প্রতিটা feature **আলাদাভাবে** দেখলে দুইটাই normal range এ → probability contour গুলো **axis-aligned concentric circle** → point টা normal ধরা পড়ে (**fail**)।
- **Multivariate:** Σ এর off-diagonal (covariance) correlation ধরে → contour **tilted ellipse** → point টা ellipse এর বাইরে → **anomaly**।

> Numerical example: [Math AM5](#am5-4-marks--univariate-vs-multivariate-same-test-point) — univariate p = 0.0060 ≥ ε (normal) কিন্তু multivariate p = 4.7 × 10⁻⁷ < ε (anomaly)।

### SA10. ★★★ Write the formal anomaly detection algorithm using the multivariate Gaussian model. *(Theory p.21–22)*

**Answer:**
1. **Fit model:** training set থেকে
   $`\mu = \frac1m\sum_{i=1}^{m}x^{(i)}`$ (n-dim vector), $`\quad \Sigma = \frac1m\sum_{i=1}^{m}(x^{(i)}-\mu)(x^{(i)}-\mu)^T`$ (n×n)
2. **Given new example x, compute**
   $`p(x) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}}\exp\!\left(-\frac12(x-\mu)^T\Sigma^{-1}(x-\mu)\right)`$
3. **Decide:** $`p(x) < \varepsilon`$ → **Anomaly**; otherwise **Normal**।

### SA11. Explain the relation between the univariate and multivariate models. *(Theory p.23)*

**Answer:** Univariate model হলো multivariate model এর **special case**, যেখানে Σ এর **off-diagonal সব 0** (features এর মধ্যে কোনো covariance নেই):
```math
\Sigma = \begin{bmatrix}\sigma_1^2 & 0\\ 0 & \sigma_2^2\end{bmatrix}
```
তখন দুইটা model **identical** p(x) দেয়, আর contour গুলো **axis-aligned**।

### SA12. ★ Compare the univariate and multivariate models. Mention the non-invertible matrix issue. *(Theory p.23–24)*

| Univariate (original) | Multivariate |
|---|---|
| বেশি ব্যবহার হয় | কম ব্যবহার হয় |
| Correlation ধরতে **manually extra feature** বানাতে হয় ($`x_1/x_2`$) | **Correlation automatically ধরে** |
| **Computationally cheap** (Σ, inverse লাগে না) | **Expensive** — n×n matrix এর inverse লাগে |
| n = 100,000 feature এও চলে | n বড় হলে খারাপ |
| ছোট training set (m = 50, 100) এও চলে | **m > n লাগবে** (ভালো হয় m ≫ n, যেমন m ≥ 10n) |

**Non-invertible (singular) Σ — কারণ ও সমাধান:**
1. **m < n** (data কম) → univariate model use করো, অথবা আরো data নাও।
2. **Redundant features** (linearly dependent, যেমন $`x_3 = x_1 + x_2`$ বা একই feature দুইবার) → redundant feature বাদ দাও / **PCA** দিয়ে dimension কমাও।

---

## Part 3 — Anomaly Detection: Math Questions

### Main Problem (15d) — এই data দিয়ে AM1 আর AM2

> An aircraft testing facility detects whether an aircraft is faulty based on **engine heat ($`x_1`$)** and **vibration ($`x_2`$)**. Training set = 10 non-faulty aircrafts:
> $`(2,2), (3,4), (1,3), (4,4), (5,7), (6,9), (7,8), (10,14), (12,15), (15,19)`$.
> A new aircraft has **heat = 10, vibration = 5**. Is it faulty? Use **ε = 0.1**.

### AM1 *(4 marks)* — Univariate Gaussian model

**Step 1 — Mean (A1):**
- $`\mu_1 = \frac{2+3+1+4+5+6+7+10+12+15}{10} = \frac{65}{10} = 6.5`$
- $`\mu_2 = \frac{2+4+3+4+7+9+8+14+15+19}{10} = \frac{85}{10} = 8.5`$

**Step 2 — Squared differences from mean (A2):**

| $`x_1`$ | $`x_2`$ | $`(x_1-6.5)^2`$ | $`(x_2-8.5)^2`$ |
|---|---|---|---|
| 2 | 2 | 20.25 | 42.25 |
| 3 | 4 | 12.25 | 20.25 |
| 1 | 3 | 30.25 | 30.25 |
| 4 | 4 | 6.25 | 20.25 |
| 5 | 7 | 2.25 | 2.25 |
| 6 | 9 | 0.25 | 0.25 |
| 7 | 8 | 0.25 | 0.25 |
| 10 | 14 | 12.25 | 30.25 |
| 12 | 15 | 30.25 | 42.25 |
| 15 | 19 | 72.25 | 110.25 |
| | **Sum** | **186.5** | **298.5** |
| | **Variance = Sum/10** | **18.65** | **29.85** |
| | **σ = √Variance** | **4.32** | **5.46** |

**Step 3 — Each feature's Gaussian (A4), test point (10, 5):**
- $`x_1`$: $`\frac{1}{4.32\sqrt{2\pi}} = \frac{1}{10.829} = 0.09235`$; $`\quad e^{-\frac{(10-6.5)^2}{2(18.65)}} = e^{-\frac{12.25}{37.3}} = e^{-0.3284} = 0.72022`$
- $`x_2`$: $`\frac{1}{5.46\sqrt{2\pi}} = \frac{1}{13.686} = 0.07307`$; $`\quad e^{-\frac{(5-8.5)^2}{2(29.85)}} = e^{-\frac{12.25}{59.7}} = e^{-0.2052} = 0.81428`$

**Step 4 — Multiply (A5):**
```math
p(10,5) = 0.09235 \times 0.72022 \times 0.07307 \times 0.81428 = \mathbf{0.00396}
```

**Step 5 — Decide (A10):** $`0.00396 < \varepsilon = 0.1`$ → **Anomaly → the aircraft is FAULTY**।

### AM2 *(4 marks)* — Multivariate Gaussian model (15d, 10 steps)

**Step 1 — Mean vector:** $`\mu = [6.5,\ 8.5]`$

**Step 2 — $`a = M - \mu`$** (প্রতিটা row থেকে μ বাদ):
$`a = [(-4.5,-6.5), (-3.5,-4.5), (-5.5,-5.5), (-2.5,-4.5), (-1.5,-1.5), (-0.5,0.5), (0.5,-0.5), (3.5,5.5), (5.5,6.5), (8.5,10.5)]`$

**Step 3 — $`a^T`$** (2×10): row 1 = $`x_1`$ deviations, row 2 = $`x_2`$ deviations

**Step 4 — $`a^T a`$** (2×2):
- (1,1) = $`\sum(x_1-\mu_1)^2 = 186.5`$
- (2,2) = $`\sum(x_2-\mu_2)^2 = 298.5`$
- (1,2) = (2,1) = $`\sum(x_1-\mu_1)(x_2-\mu_2) = 29.25+15.75+30.25+11.25+2.25-0.25-0.25+19.25+35.75+89.25 = 232.5`$

```math
a^T a = \begin{bmatrix}186.5 & 232.5\\ 232.5 & 298.5\end{bmatrix}
```

**Step 5 — Covariance matrix (A6):** $`\Sigma = \frac{a^T a}{m} = \frac{a^T a}{10} = \begin{bmatrix}18.65 & 23.25\\ 23.25 & 29.85\end{bmatrix}`$

**Step 6 — Determinant (A7):** $`|\Sigma| = 18.65 \times 29.85 - 23.25 \times 23.25 = 556.70 - 540.56 = 16.14`$; $`\quad \sqrt{|\Sigma|} = 4.017`$

**Step 7 — Inverse (A8):**
```math
\Sigma^{-1} = \frac{1}{16.14}\begin{bmatrix}29.85 & -23.25\\ -23.25 & 18.65\end{bmatrix} = 0.06196 \times \begin{bmatrix}29.85 & -23.25\\ -23.25 & 18.65\end{bmatrix} = \begin{bmatrix}1.85 & -1.44\\ -1.44 & 1.16\end{bmatrix}
```

**Step 8 — $`x - \mu`$ and $`(x-\mu)^T\Sigma^{-1}`$:**
- $`x - \mu = [10-6.5,\ 5-8.5] = [3.5,\ -3.5]`$
- $`(x-\mu)^T\Sigma^{-1} = [3.5(1.85) + (-3.5)(-1.44),\ \ 3.5(-1.44) + (-3.5)(1.16)] = [6.475+5.04,\ -5.04-4.06] = [11.515,\ -9.1]`$

**Step 9 — $`(x-\mu)^T\Sigma^{-1}(x-\mu)`$:** $`11.515(3.5) + (-9.1)(-3.5) = 40.30 + 31.85 = 72.15`$

**Step 10 — p(x) (A9, n = 2 → $`(2\pi)^{n/2} = 2\pi`$):**
```math
p(10,5) = \frac{1}{2\pi \times 4.017}\, e^{-0.5 \times 72.15} = 0.0396 \times e^{-36.08} = 0.0396 \times 2.149\times10^{-16} = \mathbf{8.6\times 10^{-18}}
```

**Decide:** $`8.6\times10^{-18} \ll 0.1`$ → **Anomaly → FAULTY**।

> কেন multivariate এ এত ছোট? Training data তে heat বাড়লে vibration ও বাড়ে (covariance 23.25, positive)। Test point এ heat বেশি (10) কিন্তু vibration কম (5) — এই **combination** টা অস্বাভাবিক, multivariate এটা ধরে।

### AM3 *(4 marks)* — Pattern A: "Σ দেওয়া আছে, পরের part করো"

> A model is trained with mean $`\mu = [5,\ 6]`$ and covariance matrix $`\Sigma = \begin{bmatrix}4 & 2\\ 2 & 3\end{bmatrix}`$. Is the test point $`x = (7, 3)`$ an anomaly? Use **ε = 0.01** and the multivariate Gaussian model.

**Step 1 — Determinant:** $`|\Sigma| = 4(3) - 2(2) = 12 - 4 = 8`$; $`\quad\sqrt{8} = 2.828`$

**Step 2 — Inverse:** $`\Sigma^{-1} = \frac18\begin{bmatrix}3 & -2\\ -2 & 4\end{bmatrix} = \begin{bmatrix}0.375 & -0.25\\ -0.25 & 0.5\end{bmatrix}`$

**Step 3 — $`x - \mu`$:** $`[7-5,\ 3-6] = [2,\ -3]`$

**Step 4 — $`(x-\mu)^T\Sigma^{-1}`$:** $`[2(0.375) + (-3)(-0.25),\ \ 2(-0.25) + (-3)(0.5)] = [0.75+0.75,\ -0.5-1.5] = [1.5,\ -2.0]`$

**Step 5 — Quadratic term:** $`[1.5,\ -2.0]\cdot[2,\ -3] = 3 + 6 = 9`$

**Step 6 — p(x):**
```math
p(x) = \frac{1}{2\pi\times 2.828}\,e^{-0.5\times 9} = 0.05627 \times e^{-4.5} = 0.05627 \times 0.01111 = \mathbf{0.000625}
```

**Decide:** $`0.000625 < 0.01`$ → **Anomaly**।

### AM4 *(2 marks)* — Pattern B: "শুধু Cov matrix পর্যন্ত"

> Five normal samples with two features: $`(2,3), (4,5), (6,4), (8,9), (10,9)`$. Find the variance-covariance matrix Σ.

**Step 1 — Mean:** $`\mu_1 = \frac{2+4+6+8+10}{5} = 6`$, $`\quad \mu_2 = \frac{3+5+4+9+9}{5} = 6`$

**Step 2 — $`a = M - \mu`$:**

| $`x_1-6`$ | $`x_2-6`$ | $`(x_1-6)^2`$ | $`(x_2-6)^2`$ | $`(x_1-6)(x_2-6)`$ |
|---|---|---|---|---|
| −4 | −3 | 16 | 9 | 12 |
| −2 | −1 | 4 | 1 | 2 |
| 0 | −2 | 0 | 4 | 0 |
| 2 | 3 | 4 | 9 | 6 |
| 4 | 3 | 16 | 9 | 12 |
| | **Sum** | **40** | **32** | **32** |

**Step 3 — $`a^T a = \begin{bmatrix}40 & 32\\ 32 & 32\end{bmatrix}`$, তারপর ÷ m = 5:**
```math
\Sigma = \begin{bmatrix}8 & 6.4\\ 6.4 & 6.4\end{bmatrix}
```

### AM5 *(4 marks)* — Univariate vs Multivariate, same test point

> Using the data of AM4, test the point $`x = (10, 4)`$ with **ε = 0.005**. (a) Use the univariate model. (b) Use the multivariate model. (c) Explain the difference.

**(a) Univariate:** $`\mu = (6, 6)`$, $`\sigma_1 = \sqrt8 = 2.828`$, $`\sigma_2 = \sqrt{6.4} = 2.530`$
- $`x_1`$: $`\frac{1}{2.828\sqrt{2\pi}} = 0.14105`$; $`\quad e^{-\frac{(10-6)^2}{2(8)}} = e^{-1} = 0.36788`$
- $`x_2`$: $`\frac{1}{2.530\sqrt{2\pi}} = 0.15770`$; $`\quad e^{-\frac{(4-6)^2}{2(6.4)}} = e^{-0.3125} = 0.73162`$
- $`p(x) = 0.14105 \times 0.36788 \times 0.15770 \times 0.73162 = \mathbf{0.00599}`$ ≥ 0.005 → **Normal**

**(b) Multivariate:** Σ from AM4
- $`|\Sigma| = 8(6.4) - 6.4(6.4) = 51.2 - 40.96 = 10.24`$; $`\sqrt{10.24} = 3.2`$
- $`\Sigma^{-1} = \frac{1}{10.24}\begin{bmatrix}6.4 & -6.4\\ -6.4 & 8\end{bmatrix} = \begin{bmatrix}0.625 & -0.625\\ -0.625 & 0.78125\end{bmatrix}`$
- $`x - \mu = [4,\ -2]`$
- $`(x-\mu)^T\Sigma^{-1} = [4(0.625)+(-2)(-0.625),\ 4(-0.625)+(-2)(0.78125)] = [3.75,\ -4.0625]`$
- Quadratic term $`= 3.75(4) + (-4.0625)(-2) = 15 + 8.125 = 23.125`$
- $`p(x) = \frac{1}{2\pi(3.2)}e^{-11.5625} = 0.04974 \times 9.516\times10^{-6} = \mathbf{4.73\times10^{-7}}`$ < 0.005 → **Anomaly**

**(c) কেন আলাদা?** Covariance 6.4 (positive) — $`x_1`$ বাড়লে $`x_2`$ ও বাড়ে। Test point এ $`x_1 = 10`$ (high) কিন্তু $`x_2 = 4`$ (low)। আলাদাভাবে দুইটাই "acceptable" তাই univariate ধরতে পারেনি; multivariate **combination** টা দেখে anomaly ধরেছে। (এটাই SA9 এর numerical example।)

---

## Part 4 — Perceptron: Short Questions

### SP1. What is a linear decision boundary / discriminant function?

**Answer:** দুই class এর মাঝের **straight line (বা high-dim এ hyperplane)** যেখানে $`g(x) = w^T x + w_0 = 0`$।
- একটা class এর সব point এ $`g(x) > 0`$, অন্য class এ $`g(x) < 0`$।
- $`w^T`$ = line এর coefficient ($`[a\ b]`$), $`w_0`$ = constant ($`c`$, line এর shift/intercept)। Perceptron এর কাজ হলো $`w`$ আর $`w_0`$ খুঁজে বের করা।

### SP2. Why do we use the augmented feature vector (last component = 1)?

**Answer:** $`g(x) = w^Tx + w_0`$ কে একটা single dot product $`[w^T\ w_0]\cdot[x\ \ 1]^T`$ হিসেবে লিখতে। শেষের **1** এর সাথে $`w_0`$ গুণ হয়ে constant term থাকে — যেটা কোনো feature এর উপর নির্ভর করে না (straight line এর c এর মতো)। এটা 1 না হলে $`w_0`$ বদলে যাবে।

### SP3. Why do we project the patterns to a higher dimension (φ function)?

**Answer:** Low dimension এ **linearly separable না** এমন data অনেক সময় **higher dimension এ linearly separable** হয়ে যায়। তাই $`\varphi(x) = [x_1^2, x_2^2, x_1x_2, x_1, x_2, 1]^T`$ দিয়ে 2-D → 6-D নিয়ে যাই। তখন 6-D তে linear boundary, কিন্তু original 2-D তে সেটা **curve (quadratic)** হয়।

### SP4. What is class normalization? Why is it done?

**Answer:** একটা class ($`\omega_2`$) এর সব pattern কে **negate (reflect)** করা: $`y \leftarrow -y`$।
- কারণ: $`g(y) < 0`$ condition টা $`g(-y) > 0`$ এর সমান। Normalize করলে **দুইটা condition এর জায়গায় শুধু একটা condition** ($`g(y) > 0`$ সবার জন্য) check করলেই হয় → algorithm simple হয়।

### SP5. Write the perceptron weight update rule.

**Answer:** যদি $`g(y) = w^T y \le 0`$ (misclassified), তাহলে $`w(i+1) = w(i) + \alpha\, y`$; আর $`g(y) > 0`$ হলে w অপরিবর্তিত।

### SP6. What is the role of the learning rate α? What happens if α = 0 or α > 1?

**Answer:** α ঠিক করে প্রতি update এ weight **কতটা** বদলাবে। Range: $`0 < \alpha \le 1`$।
- **α = 0** → কোনো update **কখনো হবে না**, শেখা বন্ধ।
- **α > 1** → step size খুব বড়, algorithm **unstable** হতে পারে।
- ভালো মান: ~0.5, অথবা $`\alpha = 1/n`$ (n = training pattern সংখ্যা; tutorial এ 1/6 ≈ 0.17)।

### SP7. How should the initial weights be chosen?

**Answer:** Arbitrary (যেমন সব 1), তবে ভালো হলো **ছোট random মান** (Uniform $`[-1, 1]`$)। **সব 0 দেওয়া যাবে না** — তাহলে training আটকে যেতে পারে।

### SP8. ★ Compare One-at-a-time and Many-at-a-time weight update. Which is better?

| One at a time | Many at a time (batch) |
|---|---|
| প্রতিটা misclassified y পাওয়া মাত্র **সাথে সাথে** update | পুরো iteration এ **একই w** দিয়ে সব y check; শেষে **একবার** update |
| $`w \leftarrow w + \alpha y`$ | $`w \leftarrow w + \alpha\sum y_{mis}`$ |
| একটা noisy point weight কে ভুল দিকে টানতে পারে | সব misclassified point এর **sum** নেয়, তাই balanced |

**Teacher's point:** **Many at a time → more stable, better convergence।** (Tutorial: simple problem + random initial weight + many-at-a-time → কম iteration লাগে।)

### SP9. When does the perceptron algorithm stop (convergence)?

**Answer:** যখন একটা **complete iteration** এ **কোনো y misclassified হয় না** (সব $`g(y) > 0`$) এবং তাই কোনো weight বদলায় না। (অথবা max iteration শেষ হলে।) সাধারণত শত বা হাজার iteration লাগতে পারে — নির্ভর করে problem এর difficulty, α, initial weights, আর strategy এর উপর।

### SP10. ★ What is the main limitation of the perceptron?

**Answer:** Perceptron শুধু **linear decision boundary** শিখতে পারে, তাই শুধু **linearly separable** data তে converge করে।
- Classic example: **XOR** — (0,0),(1,1) এক class, (0,1),(1,0) আরেক class; কোনো একটা straight line দিয়ে আলাদা করা যায় না → perceptron **কখনো converge করবে না**।
- Solution: **φ দিয়ে higher dimension এ project** করা (যেমন $`x_1x_2`$ feature যোগ), বা multi-layer network।

---

## Part 5 — Perceptron: Math Questions

### PM1 *(2 marks)* — Given g(x), classify

> A trained perceptron gives the decision boundary $`g(x) = 2.8x_1 + x_2 - 8.9`$. Classify the points $`(3, 1)`$ and $`(1, 2)`$. Where is the decision boundary?

- $`(3,1)`$: $`g = 2.8(3) + 1 - 8.9 = 8.4 + 1 - 8.9 = 0.5 > 0`$ → **Class $`\omega_1`$**
- $`(1,2)`$: $`g = 2.8(1) + 2 - 8.9 = 2.8 + 2 - 8.9 = -4.1 < 0`$ → **Class $`\omega_2`$**
- Decision boundary: $`2.8x_1 + x_2 - 8.9 = 0`$ → $`x_2 = -2.8x_1 + 8.9`$ (straight line, slope −2.8, intercept 8.9)। এই line এর **উপর অবস্থিত (on the line)** সব point এ g = 0।

### PM2 *(2 marks)* — φ projection + class normalization (Tutorial data)

> $`\omega_1 = (1,1), (1,-3), (4,5)`$; $`\omega_2 = (2,2.5), (0,2), (2,3)`$। Using $`\varphi(x) = [x_1^2, x_2^2, x_1x_2, x_1, x_2, 1]^T`$, write the normalized high-dimensional vectors.

| Pattern | Class | $`\varphi(x) = [x_1^2,\ x_2^2,\ x_1x_2,\ x_1,\ x_2,\ 1]`$ | Normalized y |
|---|---|---|---|
| (1, 1) | $`\omega_1`$ | [1, 1, 1, 1, 1, 1] | $`y_{11}`$ = [1, 1, 1, 1, 1, 1] |
| (1, −3) | $`\omega_1`$ | [1, **9**, −3, 1, −3, 1] | $`y_{12}`$ = [1, 9, −3, 1, −3, 1] |
| (4, 5) | $`\omega_1`$ | [16, 25, 20, 4, 5, 1] | $`y_{13}`$ = [16, 25, 20, 4, 5, 1] |
| (2, 2.5) | $`\omega_2`$ | [4, 6.25, 5, 2, 2.5, 1] | $`-y_{21}`$ = [−4, −6.25, −5, −2, −2.5, −1] |
| (0, 2) | $`\omega_2`$ | [0, 4, 0, 0, 2, 1] | $`-y_{22}`$ = [0, −4, 0, 0, −2, −1] |
| (2, 3) | $`\omega_2`$ | [4, 9, 6, 2, 3, 1] | $`-y_{23}`$ = [−4, −9, −6, −2, −3, −1] |

> ⚠️ **Tutorial typo:** (1, −3) এর জন্য $`x_2^2 = (-3)^2 = \mathbf{9}`$, কিন্তু Drive এর tutorial এ সব হিসাবে **1** লেখা ($`y_{12} = [1, 1, -3, 1, -3, 1]`$)। তাই tutorial এ step 2 তে $`g = -2`$ আসে; সঠিক 9 দিয়ে $`g = +6`$। Exam এ নিজে হিসাব করলে **সঠিক মান (9)** use করো। Tutorial এর numbers সহ পুরো walk-through: [notes/04-Perceptron Section 4](../notes/04-Perceptron.md)।

### PM3 *(4 marks)* — One-at-a-time, 2-D augmented (no φ), run to convergence

> $`\omega_1 = (2,3), (3,4)`$; $`\omega_2 = (1,0), (2,1)`$। Use augmented vectors $`[x_1, x_2, 1]`$, initial $`w = [1, -1, 0]`$, $`\alpha = 0.5`$, **one-at-a-time**। Find the decision boundary.

**Step 1 — Augment + normalize ($`\omega_2`$ negate):**
$`y_1 = [2, 3, 1]`$, $`\ y_2 = [3, 4, 1]`$, $`\ y_3 = -[1, 0, 1] = [-1, 0, -1]`$, $`\ y_4 = -[2, 1, 1] = [-2, -1, -1]`$

**Step 2 — Iterations** (rule: $`g = w^Ty \le 0`$ → $`w \leftarrow w + 0.5\,y`$):

| Iter | y | w (before) | $`g = w^T y`$ | Update? → new w |
|---|---|---|---|---|
| 1 | $`y_1`$ [2, 3, 1] | [1, −1, 0] | 2 − 3 + 0 = **−1** | Yes → [1, −1, 0] + [1, 1.5, 0.5] = **[2, 0.5, 0.5]** |
| 1 | $`y_2`$ [3, 4, 1] | [2, 0.5, 0.5] | 6 + 2 + 0.5 = **8.5** | No |
| 1 | $`y_3`$ [−1, 0, −1] | [2, 0.5, 0.5] | −2 + 0 − 0.5 = **−2.5** | Yes → [2, 0.5, 0.5] + [−0.5, 0, −0.5] = **[1.5, 0.5, 0]** |
| 1 | $`y_4`$ [−2, −1, −1] | [1.5, 0.5, 0] | −3 − 0.5 + 0 = **−3.5** | Yes → [1.5, 0.5, 0] + [−1, −0.5, −0.5] = **[0.5, 0, −0.5]** |
| 2 | $`y_1`$ | [0.5, 0, −0.5] | 1 + 0 − 0.5 = **0.5** | No |
| 2 | $`y_2`$ | [0.5, 0, −0.5] | 1.5 + 0 − 0.5 = **1** | No |
| 2 | $`y_3`$ | [0.5, 0, −0.5] | −0.5 + 0 + 0.5 = **0** | Yes (g = 0 → misclassified) → **[0, 0, −1]** |
| 2 | $`y_4`$ | [0, 0, −1] | 0 + 0 + 1 = **1** | No |
| 3 | $`y_1`$ | [0, 0, −1] | **−1** | Yes → **[1, 1.5, −0.5]** |
| 3 | $`y_2`$ | [1, 1.5, −0.5] | 3 + 6 − 0.5 = **8.5** | No |
| 3 | $`y_3`$ | [1, 1.5, −0.5] | −1 + 0 + 0.5 = **−0.5** | Yes → **[0.5, 1.5, −1]** |
| 3 | $`y_4`$ | [0.5, 1.5, −1] | −1 − 1.5 + 1 = **−1.5** | Yes → **[−0.5, 1, −1.5]** |
| 4 | $`y_1`$ | [−0.5, 1, −1.5] | −1 + 3 − 1.5 = **0.5** | No |
| 4 | $`y_2`$ | [−0.5, 1, −1.5] | −1.5 + 4 − 1.5 = **1** | No |
| 4 | $`y_3`$ | [−0.5, 1, −1.5] | 0.5 + 0 + 1.5 = **2** | No |
| 4 | $`y_4`$ | [−0.5, 1, −1.5] | 1 − 1 + 1.5 = **1.5** | No |

**Step 3 — Converged** in iteration 4 (কোনো update হয়নি)। Final $`w = [-0.5,\ 1,\ -1.5]`$

**Decision boundary:** $`-0.5x_1 + x_2 - 1.5 = 0`$ → $`\mathbf{x_2 = 0.5x_1 + 1.5}`$

**Check:** $`\omega_1`$: (2,3) → −1 + 3 − 1.5 = 0.5 > 0 ✔, (3,4) → 1 > 0 ✔; $`\omega_2`$: (1,0) → −2 < 0 ✔, (2,1) → −1.5 < 0 ✔

### PM4 *(4 marks)* — Many-at-a-time, same data

> Same data as PM3 ($`w = [1, -1, 0]`$, $`\alpha = 0.5`$), but use **many-at-a-time**।

Rule: পুরো iteration এ **একই w**; iteration শেষে $`w \leftarrow w + 0.5\sum y_{mis}`$

| Iter | w (used for all 4) | $`g(y_1)`$ | $`g(y_2)`$ | $`g(y_3)`$ | $`g(y_4)`$ | Misclassified | $`\sum y_{mis}`$ | New w |
|---|---|---|---|---|---|---|---|---|
| 1 | [1, −1, 0] | −1 | −1 | −1 | −1 | all 4 | [2, 6, 0] | [1, −1, 0] + 0.5[2, 6, 0] = **[2, 2, 0]** |
| 2 | [2, 2, 0] | 10 | 14 | −2 | −6 | $`y_3, y_4`$ | [−3, −1, −2] | [2, 2, 0] + [−1.5, −0.5, −1] = **[0.5, 1.5, −1]** |
| 3 | [0.5, 1.5, −1] | 4.5 | 6.5 | 0.5 | −1.5 | $`y_4`$ | [−2, −1, −1] | [0.5, 1.5, −1] + [−1, −0.5, −0.5] = **[−0.5, 1, −1.5]** |
| 4 | [−0.5, 1, −1.5] | 0.5 | 1 | 2 | 1.5 | none | — | **Converged** |

**Example calculation (Iter 1, $`\sum y_{mis}`$):** $`[2,3,1] + [3,4,1] + [-1,0,-1] + [-2,-1,-1] = [2,\ 6,\ 0]`$

**Result:** Same boundary $`x_2 = 0.5x_1 + 1.5`$, দুইটাই iteration 4 এ converge। কিন্তু One-at-a-time এ weight **7 বার** বদলেছে (3 + 1 + 3), Many-at-a-time এ মাত্র **3 বার** (প্রতি iteration এ একবার), আর একটা single point এর জন্য weight লাফিয়ে যায় না — তাই **more stable**।

### PM5 *(4 marks)* — Tutorial data, Iteration 1 (with correct φ)

> Using PM2's normalized vectors, initial $`w = [1,1,1,1,1,1]`$, $`\alpha = 0.8`$: (a) Show iteration 1 with one-at-a-time. (b) Find the weight after iteration 1 with many-at-a-time.

**(a) One at a time:**

| Step | y | w (before) | $`g = w^T y`$ | Action |
|---|---|---|---|---|
| 1 | $`y_{11}`$ = [1, 1, 1, 1, 1, 1] | [1, 1, 1, 1, 1, 1] | 1+1+1+1+1+1 = **6** | unchanged |
| 2 | $`y_{12}`$ = [1, 9, −3, 1, −3, 1] | [1, 1, 1, 1, 1, 1] | 1+9−3+1−3+1 = **6** | unchanged |
| 3 | $`y_{13}`$ = [16, 25, 20, 4, 5, 1] | [1, 1, 1, 1, 1, 1] | 16+25+20+4+5+1 = **71** | unchanged |
| 4 | $`-y_{21}`$ = [−4, −6.25, −5, −2, −2.5, −1] | [1, 1, 1, 1, 1, 1] | **−20.75** | **update** |
| 5 | $`-y_{22}`$ = [0, −4, 0, 0, −2, −1] | [−2.2, −4, −3, −0.6, −1, 0.2] | 0+16+0+0+2−0.2 = **17.8** | unchanged |
| 6 | $`-y_{23}`$ = [−4, −9, −6, −2, −3, −1] | [−2.2, −4, −3, −0.6, −1, 0.2] | 8.8+36+18+1.2+3−0.2 = **66.8** | unchanged |

Step 4 update: $`w = [1,1,1,1,1,1] + 0.8[-4, -6.25, -5, -2, -2.5, -1] = [1,1,1,1,1,1] + [-3.2, -5, -4, -1.6, -2, -0.8] = \mathbf{[-2.2,\ -4,\ -3,\ -0.6,\ -1,\ 0.2]}`$

After iteration 1: $`w = [-2.2, -4, -3, -0.6, -1, 0.2]`$ (iteration 2 এ $`y_{11}`$ থেকে আবার শুরু)।

**(b) Many at a time:** $`w = [1,1,1,1,1,1]`$ দিয়ে সব g: 6, 6, 71, **−20.75**, **−7**, **−25** → misclassified: $`-y_{21}, -y_{22}, -y_{23}`$
- $`\sum y_{mis} = [-4, -6.25, -5, -2, -2.5, -1] + [0, -4, 0, 0, -2, -1] + [-4, -9, -6, -2, -3, -1] = [-8,\ -19.25,\ -11,\ -4,\ -7.5,\ -3]`$
- $`w = [1,1,1,1,1,1] + 0.8[-8, -19.25, -11, -4, -7.5, -3] = \mathbf{[-5.4,\ -14.4,\ -7.8,\ -2.2,\ -5,\ -1.4]}`$

---

## Part 6 — Answer Key + Exam Tips

### Quick Answer Key (Math)

| Q | Topic | Answer |
|---|---|---|
| AM1 | 15d univariate | p = 0.00396 < 0.1 → **Faulty (anomaly)** |
| AM2 | 15d multivariate | Σ = [[18.65, 23.25], [23.25, 29.85]], p = 8.6 × 10⁻¹⁸ → **Faulty** |
| AM3 | Σ given | p = 0.000625 < 0.01 → **Anomaly** |
| AM4 | Cov matrix only | Σ = [[8, 6.4], [6.4, 6.4]] |
| AM5 | Uni vs multi | Uni p = 0.00599 (normal), Multi p = 4.73 × 10⁻⁷ (**anomaly**) |
| PM1 | Classify | (3,1) → ω₁ (g = 0.5); (1,2) → ω₂ (g = −4.1) |
| PM2 | φ + normalize | table (note $`y_{12}`$ = [1, 9, −3, 1, −3, 1]) |
| PM3 | One-at-a-time | w = [−0.5, 1, −1.5] → $`x_2 = 0.5x_1 + 1.5`$ |
| PM4 | Many-at-a-time | same w, converged in iteration 4 |
| PM5 | Tutorial iter 1 | one: [−2.2, −4, −3, −0.6, −1, 0.2]; many: [−5.4, −14.4, −7.8, −2.2, −5, −1.4] |

### Exam Tips — Anomaly Detection
1. Variance এ **÷ m** (course এর 15d তে 10 দিয়ে ভাগ), ÷ (m−1) না।
2. Univariate এ p(x) = প্রতিটা feature এর **(height factor × exp part)** গুণফল — চারটা সংখ্যা গুণ।
3. Multivariate এ n = 2 হলে denominator এ $`2\pi\sqrt{|\Sigma|}`$ — $`(2\pi)^{n/2}`$ ভুল করো না।
4. 2×2 inverse: **diagonal swap, off-diagonal sign change, ÷ det**।
5. Table বানিয়ে ($`x - \mu`$, square, sum) হিসাব করো — partial marks পাবে।
6. শেষে অবশ্যই **"p < ε → Anomaly"** বা **"p ≥ ε → Normal"** conclusion লিখবে।

### Exam Tips — Perceptron
1. **Normalize ভুলবে না:** $`\omega_2`$ এর সব component negate, **শেষের 1 সহ** (−1 হয়ে যায়)।
2. **φ নিজে হিসাব করো** — $`(-3)^2 = 9`$ (tutorial এ 1 লেখা typo)।
3. **g ≤ 0 → update**, g > 0 → no change।
4. One-at-a-time এ update এর পর **নতুন w** দিয়ে পরের y; Many-at-a-time এ পুরো iteration এ **একই w**।
5. প্রতিটা step এ table: y | w | g | action — পরিষ্কার দেখাবে।
6. Short question এ লিখবে: **limitation = linear only (XOR)**, **many at a time = more stable, better convergence**।
