# 06 — Regression Analysis (Linear, Multiple, Gradient Descent)

> **Exam weight:** teacher বলেছেন **"Regression Analysis ~3 marks"**। সাধারণত **math** আসে: (ক) least squares দিয়ে slope ও intercept বের করে predict করা, অথবা (খ) gradient descent-এর **batch** বা **single** mode-এ ১-২টা epoch হাতে করা।
> এই note-এর সব সংখ্যা Python দিয়ে মিলিয়ে দেখা হয়েছে। Source-এ যেখানে হিসাবের ভুল আছে, সেখানে ⚠️ দিয়ে চিহ্নিত করা আছে।

---

## 1. Regression কী?

- **Regression** হলো একটা **supervised learning** technique, যেটা **continuous value** predict করে। যেমন: price, salary, marks, CO₂ emission।
- **Classification** predict করে **discrete class** (যেমন: malignant/benign)। আর **Regression** predict করে **continuous number** (যেমন: বাড়ির দাম ৳ 1,50,000)।
- **Independent variable (x)**: এটা cause বা input/predictor। যেমন: পড়ার ঘণ্টা।
- **Dependent variable (y)**: এটা effect বা output/target। এর মান x-এর উপর নির্ভর করে। যেমন: class test marks।
- **Linearity**: x আর y-এর সম্পর্ক একটা straight line দিয়ে দেখানো যায়।

**Linear regression** মানে data point-গুলোর ভেতর দিয়ে একটা **best-fit straight line** বের করা, যাতে নতুন x দিলে y predict করা যায়।

```math
y = mx + c \quad\equiv\quad h_\theta(x) = \theta_0 + \theta_1 x
```

| Symbol | Meaning |
|---|---|
| $`m`$ / $`\theta_1`$ / $`B_1`$ / $`a_1`$ | **Slope** (gradient)। x এক unit বাড়লে y কতটা বাড়ে |
| $`c`$ / $`\theta_0`$ / $`B_0`$ / $`a_0`$ | **Intercept** (bias)। x = 0 হলে y-এর মান |
| $`h_\theta(x)`$ | **Hypothesis**, মানে predicted y |
| $`m`$ (GD notation) | Training example-এর সংখ্যা (rows) |
| $`n`$ | Feature-এর সংখ্যা (GD notation-এ)। Statistical formula-তে $`n`$ মানে data point-এর সংখ্যা |

**Positive linear relationship**: x বাড়লে y-ও বাড়ে, তাই slope > 0। যেমন height আর weight।
**Negative linear relationship**: x বাড়লে y কমে, তাই slope < 0। যেমন altitude আর temperature।

**Types:**
1. **Simple Linear Regression**: independent variable একটা (x)।
2. **Multiple Linear Regression**: independent variable ২ বা তার বেশি ($`x_1, x_2, \dots`$)।
3. (Extension) **Polynomial regression**: $`x, x^2, x^3`$-কে আলাদা feature ধরে linear regression চালানো হয়।

---

## 2. Simple Linear Regression — Statistical / Normal Equations (Least Squares) Method

**Idea:** এমন $`m, c`$ খুঁজতে হবে যাতে **sum of squared errors** $`\sum (y_i - \hat y_i)^2`$ সবচেয়ে ছোট হয়। এটাই **least squares**।

### Formulas (দুইটা form, দুটোই একই উত্তর দেয়)

**Form A: Sum form** (02a PDF, Balagurusamy style)

```math
m = \frac{n\sum xy - \sum x \sum y}{n\sum x^2 - (\sum x)^2}, \qquad c = \frac{\sum y \sum x^2 - \sum x \sum xy}{n\sum x^2 - (\sum x)^2}
```

**Form B: Mean form** (02b PDF)

```math
m = \frac{\bar x\,\bar y - \overline{xy}}{(\bar x)^2 - \overline{x^2}}, \qquad c = \bar y - m\,\bar x
```

এখানে $`\overline{xy} = \frac{\sum xy}{n}`$ আর $`\overline{x^2} = \frac{\sum x^2}{n}`$।

> **Shortcut:** আগে $`m`$ বের করো, তারপর $`c = \bar y - m\bar x`$ বসাও। এতে হিসাব কম লাগে, ভুলও কম হয়।

### ✍️ Worked Example 1 (Disney toys: Advertisement → Sales)

| x (Ad, \$) | y (Sales, \$) | $`x^2`$ | $`y^2`$ | $`xy`$ |
|---|---|---|---|---|
| 1 | 1 | 1 | 1 | 1 |
| 2 | 1 | 4 | 1 | 2 |
| 3 | 2 | 9 | 4 | 6 |
| 4 | 2 | 16 | 4 | 8 |
| 5 | 4 | 25 | 16 | 20 |
| **Σ = 15** | **Σ = 10** | **Σ = 55** | **Σ = 26** | **Σ = 37** |

$`n=5,\ \bar x = 3,\ \bar y = 2,\ \overline{xy} = 37/5 = 7.4,\ \overline{x^2} = 55/5 = 11,\ (\bar x)^2 = 9`$

**Slope:**
```math
m = \frac{\bar x\bar y - \overline{xy}}{(\bar x)^2 - \overline{x^2}} = \frac{6 - 7.4}{9 - 11} = \frac{-1.4}{-2} = \mathbf{0.7}
```

(Sum form দিয়ে check: $`m = \frac{5(37) - 15(10)}{5(55) - 15^2} = \frac{185-150}{275-225} = \frac{35}{50} = 0.7`$ ✔)

**Intercept:**
```math
c = \bar y - m\bar x = 2 - 0.7 \times 3 = 2 - 2.1 = \mathbf{-0.1}
```

**Best-fit line:** $`\boxed{y = 0.7x - 0.1}`$

**Prediction:** Ad = \$7 হলে Sales $`= 0.7(7) - 0.1 = \mathbf{4.8}`$

> ⚠️ **Source correction:** 02b PDF-এ লেখা আছে $`c = 0.1`$ আর prediction = 5। সঠিক উত্তর হলো $`c = 2 - 2.1 = -0.1`$ আর prediction = **4.8**। Sum form-ও একই উত্তর দেয়: $`c = \frac{10(55) - 15(37)}{50} = \frac{550 - 555}{50} = -0.1`$।
> **Interpretation:** Advertisement-এ \$1 বাড়ালে sales গড়ে **0.7 unit** বাড়ে।

### ✍️ Worked Example 2 (Hours of study → Class test marks, 02a PDF)

| x | y | xy | $`x^2`$ |
|---|---|---|---|
| 4 | 6 | 24 | 16 |
| 3 | 4 | 12 | 9 |
| 4 | 7 | 28 | 16 |
| 2 | 3 | 6 | 4 |
| 8 | 4 | 32 | 64 |
| **21** | **24** | **102** | **109** |

- Denominator: $`n\sum x^2 - (\sum x)^2 = 5(109) - 441 = 545 - 441 = 104`$
- $`m = \frac{5(102) - 21(24)}{104} = \frac{510 - 504}{104} = \frac{6}{104} = \mathbf{0.0577}`$
- $`c = \frac{24(109) - 21(102)}{104} = \frac{2616 - 2142}{104} = \frac{474}{104} = \mathbf{4.558}`$
- Line: $`y = 0.058x + 4.56`$
- x = 8 হলে: $`y = 0.0577(8) + 4.558 = \mathbf{5.02}`$ ✔ (source-এর উত্তর 5.024)

> এখানে $`R^2 \approx 0.006`$, অর্থাৎ fit খুবই দুর্বল। (8, 4) point-টা outlier-এর মতো আচরণ করছে।

### Worked Example 3 (02b PDF, অন্য dataset): x = 5,3,4,2,3 ; y = 7,4,8,3,4

Σx = 17, Σy = 26, Σxy = 97, Σx² = 63, n = 5, D = 5(63) − 289 = 26
- $`m = \frac{485 - 442}{26} = \frac{43}{26} = 1.654`$
- $`c = \frac{26(63) - 17(97)}{26} = \frac{1638 - 1649}{26} = \frac{-11}{26} = -0.423`$
- x = 8 হলে: $`y = 1.654(8) - 0.423 = \mathbf{12.81}`$ ✔

### Example 4 (CGPA): x = 8,5,9,7,2 ; y = 3.47,2.9,3.56,3.5,2.5
Σx = 31, Σy = 15.93, Σxy = 103.8, Σx² = 223, D = 1115 − 961 = 154
→ $`m = 25.17/154 = 0.163`$, $`c = 334.59/154 = 2.173`$, x = 4 হলে **y ≈ 2.826** ✔

---

## 3. R-Squared ($`R^2`$): model কতটা ভালো fit করছে

- **$`R^2`$** বলে দেয় regression line data-কে কতটা ভালোভাবে explain করে। মান 0 থেকে 1-এর মধ্যে থাকে। মান যত বেশি, fit তত ভালো।
- উদাহরণ: $`R^2 = 0.92`$ মানে y-এর variation-এর 92% model explain করতে পারছে।

**Formula 1 (correlation coefficient r, তারপর square):**
```math
r = \frac{n\sum xy - \sum x\sum y}{\sqrt{[n\sum x^2 - (\sum x)^2]\,[n\sum y^2 - (\sum y)^2]}}, \qquad R^2 = r^2
```

**Formula 2 (SSE/SST):**
```math
R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum (y_i - \hat y_i)^2}{\sum (y_i - \bar y)^2}
```

### ✍️ R² for Disney example ($`y = 0.7x - 0.1`$)

**Method 1:** $`n\sum y^2 - (\sum y)^2 = 5(26) - 100 = 30`$
```math
r = \frac{35}{\sqrt{50 \times 30}} = \frac{35}{\sqrt{1500}} = \frac{35}{38.73} = 0.904 \Rightarrow R^2 = 0.817
```

**Method 2:**

| x | y | $`\hat y = 0.7x - 0.1`$ | $`(y-\hat y)^2`$ | $`(y-\bar y)^2`$ |
|---|---|---|---|---|
| 1 | 1 | 0.6 | 0.16 | 1 |
| 2 | 1 | 1.3 | 0.09 | 1 |
| 3 | 2 | 2.0 | 0 | 0 |
| 4 | 2 | 2.7 | 0.49 | 0 |
| 5 | 4 | 3.4 | 0.36 | 4 |
| | | | **SSE = 1.10** | **SST = 6** |

$`R^2 = 1 - 1.1/6 = \mathbf{0.817}`$, অর্থাৎ ≈ 82% ✔ (দুই method-এ একই উত্তর)

### Pizza example (02b PDF): Size → Price
x = 6, 8, 12, 14, 18 ; y = 350, 775, 1150, 1395, 1675

- $`\bar x = 11.6,\ \bar y = 1069,\ \overline{xy} = 14356,\ (\bar x)^2 = 134.56,\ \overline{x^2} = 152.8`$
- $`a_1 = \frac{11.6 \times 1069 - 14356}{134.56 - 152.8} = \frac{-1955.6}{-18.24} = \mathbf{107.21}`$
- $`a_0 = 1069 - 107.21 \times 11.6 = \mathbf{-174.69}`$ (source-এ round করার কারণে −174.46/−174.6 এসেছে)
- 17-inch pizza: $`107.21(17) - 174.69 \approx`$ **1647.96 Tk** ✔
- ⚠️ সঠিক হিসাবে $`r = 0.984`$, তাই $`R^2 = 0.968 \approx`$ **97%**। Source-এ $`r = 0.96`$, $`R^2 = 0.92`$ লেখা আছে, যেটা ভুল। Method যদিও একই।

---

## 4. Multiple Linear Regression

একাধিক input থেকে একটা output predict করা হয়:
```math
Y = m_0 + m_1X_1 + m_2X_2 + \dots + m_kX_k + \varepsilon \quad\equiv\quad h_\theta(x) = \theta_0 + \theta_1x_1 + \dots + \theta_nx_n = \theta^T x
```
- $`\varepsilon`$ হলো **random error term**। যেমন blood pressure একই মানুষের ক্ষেত্রেও প্রতিবার একটু আলাদা আসে।
- ML notation-এ $`x_0 = 1`$ ধরা হয়, তাহলে $`h_\theta(x) = \theta^T x`$ (এটা $`[1\times(n+1)]\cdot[(n+1)\times 1]`$, ফল একটা scalar)।
- Example: $`h(x) = 80 + 0.1x_1 + 0.01x_2 + 3x_3 - 2x_4`$ (house price: size, bedrooms, floors, age)

### Two-variable normal equations (statistical method)
**Deviation sums** ব্যবহার করতে হবে (source p.11-এর definition অনুযায়ী):
```math
S_{11} = \sum x_1^2 - \frac{(\sum x_1)^2}{n},\quad S_{12} = \sum x_1x_2 - \frac{\sum x_1\sum x_2}{n},\quad S_{1y} = \sum x_1y - \frac{\sum x_1\sum y}{n}\ \ (\text{etc.})
```
```math
m_1 = \frac{S_{22}S_{1y} - S_{12}S_{2y}}{S_{11}S_{22} - S_{12}^2},\qquad m_2 = \frac{S_{11}S_{2y} - S_{12}S_{1y}}{S_{11}S_{22} - S_{12}^2},\qquad m_0 = \bar y - m_1\bar x_1 - m_2\bar x_2
```

### ✍️ CO₂ example (Volume $`x_1`$, Weight $`x_2`$, CO₂ y)
Data: (10,7,9), (12,11,9), (10,9,9), (15,11,10), (15,13,9)
Raw sums: Σx₁ = 62, Σx₂ = 51, Σy = 46, Σx₁² = 794, Σx₂² = 541, Σx₁x₂ = 652, Σx₁y = 573, Σx₂y = 470. Means: 12.4, 10.2, 9.2

- $`S_{11} = 794 - 62^2/5 = 794 - 768.8 = 25.2`$
- $`S_{22} = 541 - 51^2/5 = 541 - 520.2 = 20.8`$
- $`S_{12} = 652 - 62(51)/5 = 652 - 632.4 = 19.6`$
- $`S_{1y} = 573 - 62(46)/5 = 573 - 570.4 = 2.6`$
- $`S_{2y} = 470 - 51(46)/5 = 470 - 469.2 = 0.8`$
- Denominator $`= 25.2(20.8) - 19.6^2 = 524.16 - 384.16 = 140`$
- $`m_1 = \frac{20.8(2.6) - 19.6(0.8)}{140} = \frac{54.08 - 15.68}{140} = \frac{38.4}{140} = \mathbf{0.2743}`$
- $`m_2 = \frac{25.2(0.8) - 19.6(2.6)}{140} = \frac{20.16 - 50.96}{140} = \mathbf{-0.22}`$
- $`m_0 = 9.2 - 0.2743(12.4) + 0.22(10.2) = 9.2 - 3.401 + 2.244 = \mathbf{8.043}`$
- **Model:** $`Y = 8.043 + 0.2743x_1 - 0.22x_2`$; $`x_1 = 17, x_2 = 11`$ হলে **Y ≈ 10.29**

> ⚠️ **Source note:** 02b PDF formula-তে **raw sums** বসিয়েছে ($`m_1 = \frac{541 \cdot 573 - 652 \cdot 470}{794 \cdot 541 - 652^2} = 0.7984`$, $`m_2 = -0.09348`$, $`m_0 = 0.2533`$, Y(17,11) = 12.798)। Raw sum শুধু **intercept ছাড়া** model-এর জন্য খাটে। Intercept রাখলে deviation sums লাগবে (PDF-এর p.11-এও তা-ই বলা আছে)। পরীক্ষায় **method লিখে, deviation form-এ** হিসাব করা নিরাপদ। চাইলে "raw-sum shortcut দিলে 0.7984" কথাটা পাশে লিখে দিতে পারো, যদি teacher-এর উত্তর ওটা হয়।

### Normal Equation (matrix form, ML style)
```math
\boxed{\theta = (X^TX)^{-1}X^Ty}
```
- $`X`$ হলো **design matrix** $`[m \times (n+1)]`$, যার প্রথম column-এ সব 1 ($`x_0`$)। $`y`$ হলো $`[m\times1]`$ column vector।
- Dimension: $`(n+1)\times m \cdot m\times(n+1) \cdot (n+1)\times m \cdot m\times1 \Rightarrow (n+1)\times1`$
- Octave-এ: `pinv(X'*X)*X'*y`
- 3-variable হলে: $`A\cdot B = C \Rightarrow B = A^{-1}C`$, যেখানে A হলো $`S_{ij}`$-এর matrix, B = [m₁ m₂ m₃]ᵀ, C = [S₁y S₂y S₃y]ᵀ। তারপর $`m_0 = \bar Y - m_1\bar X_1 - m_2\bar X_2 - m_3\bar X_3`$।

### Gradient Descent vs Normal Equation

| | Gradient Descent | Normal Equation |
|---|---|---|
| α choose করতে হয়? | হ্যাঁ | না |
| Iteration | অনেক | লাগে না (one shot) |
| Feature scaling | দরকার | দরকার নেই |
| n অনেক বড় হলে (≥10,000) | ভালো কাজ করে | ধীর, কারণ $`(X^TX)^{-1}`$-এর খরচ $`O(n^3)`$ |

**$`X^TX`$ non-invertible হওয়ার কারণ:** (1) **redundant/linearly dependent features** (যেমন size in feet² আর size in m²), (2) **features বেশি, data কম** ($`m \le n`$)। **Fix:** dependent feature delete করা, অথবা **regularization** ব্যবহার করা, অথবা `pinv` দিয়ে inverse নেওয়া।

---

## 5. Cost Function $`J(\theta)`$

**Cost function** মাপে model-এর prediction actual y থেকে গড়ে কতটা দূরে। এটাকে **squared error cost function** বলে:
```math
J(\theta_0,\theta_1) = \frac{1}{2m}\sum_{i=1}^{m}\left(h_\theta(x^{(i)}) - y^{(i)}\right)^2
```
- $`\frac{1}{m}`$ নেওয়া হয় average-এর জন্য। $`\frac{1}{2}`$ রাখা হয় যাতে derivative নিলে 2 কেটে যায়। এতে minimum-এর অবস্থান বদলায় না।
- **Goal:** $`\min_{\theta_0,\theta_1} J(\theta_0,\theta_1)`$
- একটা parameter থাকলে $`J(\theta_1)`$ দেখতে **parabola (bowl)**। দুইটা parameter থাকলে এটা **paraboloid**, যেটা **contour plot**-এ concentric ellipse হিসেবে দেখা যায়। একই contour-এর সব point-এ J-এর মান সমান।
- Linear regression-এর J সবসময় **convex**, তাই এর **local minimum একটাই, আর সেটাই global minimum**। ফলে GD সবসময় global optimum-এ পৌঁছায় (α ঠিক থাকলে)।

**ছোট example** (lecture): data (1,1), (2,2), (3,3), $`\theta_0 = 0`$ ধরে:
- $`\theta_1 = 1`$: $`J = 0`$ (perfect fit)
- $`\theta_1 = 0.5`$: $`J = \frac{1}{6}(0.5^2 + 1^2 + 1.5^2) = \frac{3.5}{6} = 0.58`$
- $`\theta_1 = 0`$: $`J = \frac{1}{6}(1 + 4 + 9) = 2.33`$

অতএব best হলো $`\theta_1 = 1`$।

---

## 6. Gradient Descent (GD)

**Gradient descent** একটা iterative algorithm। Initial θ (যেমন 0, 0) থেকে শুরু করে প্রতি step-এ cost function-এর **slope (gradient)-এর উল্টো দিকে** একটু করে নামে, যতক্ষণ না minimum-এ পৌঁছায় (convergence)।

```math
\theta_j := \theta_j - \alpha\frac{\partial}{\partial\theta_j}J(\theta)\qquad(\text{simultaneous update for all } j)
```

### Equations (01 Equations PDF অনুযায়ী)

**BATCH MODE**: পুরো table (সব m row) দেখার পর θ **একবার** update হয়:

```math
\text{(2a)}\ \ \frac{\partial J}{\partial\theta_0} = \frac{1}{m}\sum_{i=1}^m\left(h_\theta(x^{(i)}) - y^{(i)}\right) = \textbf{Average Error}
```
```math
\text{(2b)}\ \ \frac{\partial J}{\partial\theta_1} = \frac{1}{m}\sum_{i=1}^m\left(h_\theta(x^{(i)}) - y^{(i)}\right)x^{(i)} = \textbf{Average of (Error × x)}
```
```math
\text{(3a)}\ \ temp0 = \theta_0 - \alpha\cdot PD_{\theta_0}\qquad \text{(3b)}\ \ temp1 = \theta_1 - \alpha\cdot PD_{\theta_1}
```
```math
\text{(4a,4b)}\ \ \theta_0 := temp0,\ \ \theta_1 := temp1\ \ (\text{simultaneously})
```

**SINGLE MODE** (Stochastic GD): **প্রতিটা row**-এর পরেই θ update হয়। এখানে **Σ আর 1/m বাদ** দেওয়া হয়:
```math
\theta_0 := \theta_0 - \alpha\cdot Error_i,\qquad \theta_1 := \theta_1 - \alpha\cdot Error_i\cdot x_i
```
যেখানে $`Error_i = h_\theta(x_i) - y_i = \text{Predicted} - \text{Actual}`$

> **Multiple variables-এ:** $`\theta_j := \theta_j - \alpha\frac{1}{m}\sum(h_\theta(x^{(i)}) - y^{(i)})\,x_j^{(i)}`$, যেখানে $`x_0 = 1`$। ফলে $`\theta_0`$-এর rule-ও আসলে একই formula।

**Simultaneous update কেন দরকার:** temp0 আর temp1 দুটোই **পুরোনো θ** দিয়ে হিসাব করতে হবে। নতুন θ₀ বের করে সেটা দিয়ে θ₁ হিসাব করলে সেটা আর সঠিক gradient descent থাকে না।

**Terms:**
- **Epoch** = পুরো training set একবার পার হওয়া।
- **Iteration**: batch mode-এ 1 iteration = 1 epoch (এক update)। Single mode-এ প্রতি row = 1 iteration, তাই 5 row মানে 1 epoch-এ 5টা iteration।

### Batch vs Single (Stochastic) mode

| | Batch Mode | Single (Stochastic) Mode |
|---|---|---|
| Update কখন | প্রতি epoch-এ **১ বার** | প্রতি **row**-এ ১ বার |
| Gradient | সব row-এর **average** error | শুধু ঐ row-এর error (Σ, 1/m নেই) |
| m দিয়ে ভাগ | হ্যাঁ | না |
| Path | Smooth, J ধীরে ধীরে কমে | Noisy, শেষের দিকে "bounce" করে |
| Speed | বড় dataset-এ ধীর | দ্রুত, বড় data-র জন্য ভালো |

---

## 7. ✍️ Tutorial Dataset: Step-by-step (exam pattern ঠিক এরকম)

**Data:** (x, y) = (1,1), (2,3), (4,3), (3,2), (5,5); **θ₀ = 0, θ₁ = 0, α = 0.01, m = 5**

### 7A. BATCH MODE (02 PDF-এর table)

**Epoch 1** (θ₀ = 0, θ₁ = 0):

| x | y | $`h = \theta_0 + \theta_1x`$ | Error = h − y | Error × x |
|---|---|---|---|---|
| 1 | 1 | 0 | −1 | −1 |
| 2 | 3 | 0 | −3 | −6 |
| 4 | 3 | 0 | −3 | −12 |
| 3 | 2 | 0 | −2 | −6 |
| 5 | 5 | 0 | −5 | −25 |
| | | **Sum** | **−14** | **−50** |
| | | **Average (PD)** | **−2.8** | **−10** |

- $`\theta_0 = 0 - 0.01(-2.8) = \mathbf{0.028}`$
- $`\theta_1 = 0 - 0.01(-10) = \mathbf{0.1}`$

**Epoch 2** (θ₀ = 0.028, θ₁ = 0.1):

| x | y | h | Error | Error × x |
|---|---|---|---|---|
| 1 | 1 | 0.128 | −0.872 | −0.872 |
| 2 | 3 | 0.228 | −2.772 | −5.544 |
| 4 | 3 | 0.428 | −2.572 | −10.288 |
| 3 | 2 | 0.328 | −1.672 | −5.016 |
| 5 | 5 | 0.528 | −4.472 | −22.36 |
| | | **Sum** | −12.36 | −44.08 |
| | | **Avg** | **−2.472** | **−8.816** |

- $`\theta_0 = 0.028 + 0.01(2.472) = \mathbf{0.05272}`$
- $`\theta_1 = 0.1 + 0.01(8.816) = \mathbf{0.18816}`$

**Epoch 3** (θ₀ = 0.05272, θ₁ = 0.18816): h = 0.24088, 0.42904, 0.80536, 0.6172, 0.99352
Errors: −0.75912, −2.57096, −2.19464, −1.3828, −4.00648 → **Avg = −2.1828**
Error × x: −0.75912, −5.14192, −8.77856, −4.1484, −20.0324 → **Avg = −7.77208**
→ $`\theta_0 = \mathbf{0.074548}`$, $`\theta_1 = \mathbf{0.265881}`$

**Epoch 4:** Avg error = −1.92781, Avg(err × x) = −6.851667 → θ₀ = **0.093826**, θ₁ = **0.334397**
**Epoch 5:** Avg error = −1.702981, Avg(err × x) = −6.04015 → θ₀ = **0.110856**, θ₁ = **0.394799**

Cost কমছে: J = 4.8 (শুরুতে, = Σy²/2m = 48/10) → 3.785 → 2.997 → 2.384 → 1.907 → 1.536। অর্থাৎ GD ঠিকমতো কাজ করছে ✔

### 7B. SINGLE MODE (03 PDF / Brownlee tutorial)

প্রতি row-এর পর update: $`\theta_0 \leftarrow \theta_0 - \alpha\cdot err`$, $`\theta_1 \leftarrow \theta_1 - \alpha\cdot err\cdot x`$

**Epoch 1:**

| Iter | x | y | h (পুরোনো θ দিয়ে) | Error | Error × x | new θ₀ | new θ₁ |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 0 + 0(1) = 0 | −1 | −1 | 0 + 0.01 = **0.01** | **0.01** |
| 2 | 2 | 3 | 0.01 + 0.01(2) = 0.03 | −2.97 | −5.94 | 0.01 + 0.0297 = **0.0397** | 0.01 + 0.0594 = **0.0694** |
| 3 | 4 | 3 | 0.0397 + 0.0694(4) = 0.3173 | −2.6827 | −10.7308 | **0.066527** | **0.176708** |
| 4 | 3 | 2 | 0.596651 | −1.403349 | −4.210047 | **0.080560** | **0.218808** |
| 5 | 5 | 5 | 1.174603 | −3.825397 | −19.126986 | **0.118814** | **0.410078** |

**Epoch 2:**

| Iter | x | y | h | Error | new θ₀ | new θ₁ |
|---|---|---|---|---|---|---|
| 6 | 1 | 1 | 0.528893 | −0.471107 | 0.123526 | 0.414789 |
| 7 | 2 | 3 | 0.953104 | −2.046896 | 0.143994 | 0.455727 |
| 8 | 4 | 3 | 1.966904 | −1.033096 | 0.154325 | 0.497051 |
| 9 | 3 | 2 | 1.645479 | −0.354521 | 0.157871 | 0.507687 |
| 10 | 5 | 5 | 2.696305 | −2.303695 | 0.180908 | 0.622872 |

**Epoch 3:** iter 11-15 → (0.18287, 0.624834), (0.198544, 0.656183), (0.200312, 0.663252), (0.198411, 0.65755), (0.213549, 0.733242)
**Epoch 4:** iter 16-20 → (0.214081, 0.733774), (0.227265, 0.760141), (0.224587, 0.749428), (0.219858, 0.735242), **(0.230897, 0.790439)**

**Final (20 iterations / 4 epochs):** $`B_0 = \theta_0 \approx 0.2309`$, $`B_1 = \theta_1 \approx 0.7904`$

> **Notice:** iter 14 আর 18-19-এ error positive হয়ে গেছে, তাই θ সামান্য কমেছে। এটাই stochastic mode-এর "bouncing around"।
> **Compare:** Least squares-এর exact answer হলো $`\theta_0 = 0.4`$, $`\theta_1 = 0.8`$ (Σx = 15, Σy = 14, Σxy = 50, Σx² = 55)। ৪ epoch-এই single mode সেই উত্তরের কাছাকাছি চলে এসেছে। Batch mode একই সংখ্যক epoch-এ অনেক পিছিয়ে, কারণ সেখানে ৫ গুণ কম update হয়েছে।
> (Tutorial-এর একটা known quirk: final prediction table-এর মানগুলো (0.955 …) আসলে iter 19-এর θ (0.219858, 0.735242) দিয়ে করা।)

---

## 8. Learning Rate α: effect

- **α খুব ছোট**: ছোট ছোট step, তাই convergence **খুব ধীর**।
- **α খুব বড়**: minimum **overshoot** করে, J **বাড়তে থাকে** বা oscillate করে, ফলে **diverge** করে।
- **α ঠিক মাপের**: প্রতি iteration-এ J কমে। Minimum-এর কাছে গেলে gradient এমনিতেই ছোট হয়ে যায়, তাই α fixed রাখলেও step ছোট হয়ে আসে। α বদলানোর দরকার নেই।
- **Debugging:** J(θ) vs iterations plot করো। J বাড়লে বা ঢেউয়ের মতো উঠানামা করলে **α কমাও**।
- **Try:** 0.001, 0.003, 0.01, 0.03, 0.1, 0.3 (প্রায় 3× করে বাড়িয়ে)।
- **Local minimum-এ** derivative = 0, তাই θ := θ − α·0, অর্থাৎ θ আর বদলায় না।

**Concrete proof (Demo Q01-এর data, নিচে দেখো):** α = 0.2 হলে batch mode-এ J: **2.9 → 7.19 → 27.58 → 122.88**, অর্থাৎ **diverge** করছে। α = 0.01 হলে J: **2.9 → 2.54 → 2.29 → 2.10**, অর্থাৎ ঠিকমতো কমছে।

### Feature Scaling (multiple variables-এ GD-র জন্য)
Feature-গুলোর range খুব আলাদা হলে (যেমন size 0-2000, bedrooms 1-5) contour লম্বা-সরু ellipse হয়, তখন GD অনেক ধীর হয়। সমাধান হলো scale করে মোটামুটি −1 থেকে +1 range-এ আনা:
```math
x_i \leftarrow \frac{x_i - \mu_i}{s_i}\quad (s_i = \text{std dev বা max} - \text{min})\quad\text{(Mean normalization)}
```
Rule of thumb: −3 থেকে +3 পর্যন্ত চলে; −⅓ থেকে +⅓-ও চলে।

---

## 9. Regression Demo Questions: bilingual answers

সব প্রশ্নে একই recipe:
1. প্রতি row-এ $`h = \theta_0 + \theta_1x_1 + \theta_2x_2 + \dots`$ হিসাব করো।
2. Error = h − y বের করো।
3. প্রতিটা j-এর জন্য Error × $`x_j`$ বের করো।
4. **Batch:** column average নাও, তারপর $`\theta_j \leftarrow \theta_j - \alpha \cdot avg`$। **Single:** প্রতি row-এর পরেই $`\theta_j \leftarrow \theta_j - \alpha\cdot Error\cdot x_j`$।

> ⚠️ **গুরুত্বপূর্ণ:** Demo question-গুলোতে **α = 0.2** দেওয়া, যেটা এই data-র জন্য অনেক বড়। তাই প্রায় সব প্রশ্নে θ **diverge** করে। পরীক্ষায় হিসাবের পরে এক লাইন লিখে দাও: *"J increased → α too large → diverging; should reduce α"*। এতে concept দেখানো হয়, বাড়তি নম্বরও পেতে পারো।

### Q01: Linear, 1 variable. θ₀ = 2, θ₁ = 1, α = 0.2; data (1,0), (2,4), (3,5), (5,9), (6,12)

**Batch, Epoch 1:**

| x | y | h = 2 + x | Error | Error × x |
|---|---|---|---|---|
| 1 | 0 | 3 | 3 | 3 |
| 2 | 4 | 4 | 0 | 0 |
| 3 | 5 | 5 | 0 | 0 |
| 5 | 9 | 7 | −2 | −10 |
| 6 | 12 | 8 | −4 | −24 |
| | | Sum | −3 | −31 |
| | | **Avg** | **−0.6** | **−6.2** |

θ₀ = 2 − 0.2(−0.6) = **2.12**, θ₁ = 1 − 0.2(−6.2) = **2.24**

**Batch, Epoch 2** (2.12, 2.24): h = 4.36, 6.6, 8.84, 13.32, 15.56; Error = 4.36, 2.6, 3.84, 4.32, 3.56 (sum 18.68, **avg 3.736**); Error × x = 4.36, 5.2, 11.52, 21.6, 21.36 (sum 64.04, **avg 12.808**)
→ θ₀ = 2.12 − 0.7472 = **1.3728**, θ₁ = 2.24 − 2.5616 = **−0.3216**

**Batch, Epoch 3** (1.3728, −0.3216): Errors 1.0512, −3.2704, −4.592, −9.2352, −12.5568 (**avg −5.72064**); Error × x avg = **−28.15648**
→ θ₀ = **2.516928**, θ₁ = **5.309696**। J = 2.9 → 7.19 → 27.58 → 122.88, অর্থাৎ **diverging** (θ₁ দোল খাচ্ছে: 1 → 2.24 → −0.32 → 5.31)।

**Single mode, Epoch 1:**

| Iter | x | y | h | Error | θ₀ new | θ₁ new |
|---|---|---|---|---|---|---|
| 1 | 1 | 0 | 2 + 1 = 3 | 3 | 2 − 0.6 = **1.4** | 1 − 0.6 = **0.4** |
| 2 | 2 | 4 | 1.4 + 0.8 = 2.2 | −1.8 | 1.4 + 0.36 = **1.76** | 0.4 + 0.72 = **1.12** |
| 3 | 3 | 5 | 1.76 + 3.36 = 5.12 | 0.12 | **1.736** | 1.12 − 0.072 = **1.048** |
| 4 | 5 | 9 | 1.736 + 5.24 = 6.976 | −2.024 | **2.1408** | 1.048 + 2.024 = **3.072** |
| 5 | 6 | 12 | 2.1408 + 18.432 = 20.5728 | 8.5728 | **0.42624** | 3.072 − 10.28736 = **−7.21536** |

Epoch 2 (iter 6-10): (1.784064, −5.857536), (4.570266, −0.285133), (4.827292, 0.485947), (5.175887, 2.22892), (3.866005, −5.63037), অর্থাৎ oscillate করছে।

(α = 0.01 হলে batch-এ: epoch 1 → (2.006, 1.062), epoch 2 → (2.009832, 1.114496), epoch 3 → (2.011841, 1.158987), আর J কমছে ✔। Exact LS answer: θ₀ = −1.512, θ₁ = 2.209, R² = 0.976)

### Q02: Linear, 2 variables. θ = (2, 1, 3), α = 0.2; (x₁, x₂, y) = (1,−1,0), (2,4,3), (3,5,6), (5,9,12), (6,12,20)

$`h = \theta_0 + \theta_1x_1 + \theta_2x_2`$

**Batch, Epoch 1:**

| x₁ | x₂ | y | h = 2 + x₁ + 3x₂ | Err | Err·x₁ | Err·x₂ |
|---|---|---|---|---|---|---|
| 1 | −1 | 0 | 0 | 0 | 0 | 0 |
| 2 | 4 | 3 | 16 | 13 | 26 | 52 |
| 3 | 5 | 6 | 20 | 14 | 42 | 70 |
| 5 | 9 | 12 | 34 | 22 | 110 | 198 |
| 6 | 12 | 20 | 44 | 24 | 144 | 288 |
| | | | Sum | 73 | 322 | 608 |
| | | | **Avg** | **14.6** | **64.4** | **121.6** |

θ₀ = 2 − 0.2(14.6) = **−0.92**, θ₁ = 1 − 0.2(64.4) = **−11.88**, θ₂ = 3 − 0.2(121.6) = **−21.32** (diverge করছে)

**Single, Epoch 1:**
- Row 1: h = 0, err = 0, θ অপরিবর্তিত থাকে: (2, 1, 3)
- Row 2: h = 2 + 2 + 12 = 16, err = 13 → θ₀ = 2 − 2.6 = **−0.6**, θ₁ = 1 − 0.2(26) = **−4.2**, θ₂ = 3 − 0.2(52) = **−7.4**
- Row 3: h = −0.6 − 12.6 − 37 = −50.2, err = −56.2 → (**10.64, 29.52, 48.8**)
- Row 4: h = 597.44, err = 585.44 → (−106.448, −555.92, −1004.992)
- Row 5: err = −15521.872 → (2997.93, 18070.33, 36247.50)। Explode করে গেছে, α অনেক বড়।

### Q03: Linear, 3 variables. θ = (2, 1, 3, 1), α = 0.2; (x₁, x₂, x₃, y) = (1,−1,2,0), (2,4,1,3), (3,5,1,6), (5,9,6,12), (6,12,9,20)

**Batch, Epoch 1:** h = 2 + x₁ + 3x₂ + x₃ → 2, 17, 21, 40, 53; Error = 2, 14, 15, 28, 33

| | Err | Err·x₁ | Err·x₂ | Err·x₃ |
|---|---|---|---|---|
| Sum | 92 | 413 | 777 | 498 |
| **Avg** | **18.4** | **82.6** | **155.4** | **99.6** |

θ = (2 − 3.68, 1 − 16.52, 3 − 31.08, 1 − 19.92) = **(−1.68, −15.52, −28.08, −18.92)**

**Single:** Row 1: h = 2, err = 2 → (1.6, 0.6, 3.4, 0.2); Row 2: h = 16.6, err = 13.6 → (−1.12, −4.84, −7.48, −2.52); Row 3: h = −55.56, err = −61.56 → (11.192, 32.096, 54.08, 9.792); …diverges

### Q04: Polynomial, degree 2. θ = (2, 1, 3), α = 0.2, data (1,0), (2,4), (3,5), (5,9), (6,12)
**Trick:** feature engineering করো: $`x_1 = x`$, $`x_2 = x^2`$। তারপর এটা হুবহু 2-variable linear regression।
$`h = \theta_0 + \theta_1x + \theta_2x^2`$। Rows: (x, x²) = (1,1), (2,4), (3,9), (5,25), (6,36)
**Batch, Epoch 1:** h = 6, 16, 32, 82, 116; Err = 6, 12, 27, 73, 104 → Avg Err = **44.4**, Avg(Err·x) = 1100/5 = **220**, Avg(Err·x²) = 5866/5 = **1173.2**
→ θ = (2 − 8.88, 1 − 44, 3 − 234.64) = **(−6.88, −43, −231.64)**
**Single:** Row 1: h = 6, err = 6 → (0.8, −0.2, 1.8); Row 2: h = 7.6, err = 3.6 → (0.08, −1.64, −1.08); …

> Polynomial-এ **feature scaling আরও জরুরি**, কারণ x² আর x³ খুব দ্রুত বড় হয়ে যায়।

### Q05: Polynomial, degree 3. θ = (2, 1, 3, 1). Features: x, x², x³
Rows: (1,1,1), (2,4,8), (3,9,27), (5,25,125), (6,36,216); h = 2 + x + 3x² + x³ → 7, 24, 59, 207, 332; Err = 7, 20, 54, 198, 320
Sums: 599, 3119, 17043, 95495 → Avg: **119.8, 623.8, 3408.6, 19099**
→ θ = **(−21.96, −123.76, −678.72, −3818.8)**

### Q06-Q08: Logistic Regression
- **Hypothesis:** $`h_\theta(x) = \sigma(z) = \frac{1}{1+e^{-z}}`$, যেখানে $`z = \theta^Tx`$। Output 0 থেকে 1-এর মধ্যে থাকে (probability)।
- **Update rule দেখতে linear-এর মতোই:** $`\theta_j := \theta_j - \alpha\frac{1}{m}\sum(h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}`$। শুধু h বদলায়, আর cost হয় log-loss।
- Logistic হলো **classification**-এর জন্য, তাই y সাধারণত 0/1 হওয়ার কথা। Demo-র data-তে y = 4, 5, 9… দেওয়া আছে, তবুও mechanically করা যায়।
- **Q06 Batch Epoch 1** (θ = 2, 1): z = 3, 4, 5, 7, 8 → h = 0.952574, 0.982014, 0.993307, 0.999089, 0.999665
  Err = 0.952574, −3.017986, −4.006693, −8.000911, −11.000335 (sum −25.073351, avg −5.014670)
  Err·x sum = −123.110044 (avg −24.622009)
  → θ₀ = 2 + 0.2(5.014670) = **3.002934**, θ₁ = 1 + 0.2(24.622009) = **5.924402**
- Q07/Q08: একই নিয়ম। $`z = \theta_0 + \theta_1x_1 + \theta_2x_2 (+\theta_3x_3)`$ বের করে sigmoid নাও, বাকিটা Q02/Q03-এর মতো।

### Q09-Q16: Regularized versions (λ = 0.5)
**Regularization** মানে বড় θ-কে penalize করা, যাতে **overfitting** কমে। এটা অনেক feature কিন্তু কম data থাকলে বা $`X^TX`$ non-invertible হলেও কাজে লাগে।
```math
J(\theta) = \frac{1}{2m}\left[\sum(h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda\sum_{j=1}^{n}\theta_j^2\right]
```
```math
\theta_0 := \theta_0 - \alpha\frac{1}{m}\sum Err\quad(\theta_0 \text{ regularize হয় না})
```
```math
\theta_j := \theta_j - \alpha\left[\frac{1}{m}\sum Err\cdot x_j + \frac{\lambda}{m}\theta_j\right]\quad (j\ge1)
```

**Q09 Batch, Epoch 1** (θ = 2, 1, α = 0.2, λ = 0.5, m = 5): Errors একই (avg −0.6, avg Err·x −6.2)
- θ₀ = 2 − 0.2(−0.6) = **2.12**
- θ₁ = 1 − 0.2[−6.2 + (0.5/5)(1)] = 1 − 0.2(−6.1) = **2.22** (unregularized হলে 2.24 আসত। Regularization θ₁-কে একটু ছোট রাখে)

**Epoch 2** (2.12, 2.22): avg Err = 3.668, avg Err·x = 12.508 → θ₀ = **1.3864**, θ₁ = 2.22 − 0.2(12.508 + 0.222) = **−0.326**

Q10-Q16: একই নিয়ম, শুধু Q02-Q08-এর setup-এ θ₁…θₙ-এর gradient-এ $`+\frac{\lambda}{m}\theta_j`$ যোগ করো।

### Short theory Qs (lecture notes থেকে)
- **Q: GD algorithm for multiple variables লেখো।** Repeat until convergence: $`\theta_j := \theta_j - \alpha\frac1m\sum_{i}(h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}`$, সব j = 0…n-এর জন্য simultaneously, $`x_0 = 1`$।
- **Q: Feature scaling কেন দরকার?** Range আলাদা হলে contour সরু হয়, ফলে GD zig-zag করে ধীরে চলে। Scale করলে contour গোলাকার হয়, তাই GD দ্রুত converge করে।
- **Q: Polynomial regression কীভাবে multiple linear regression দিয়ে করা যায়?** $`x_1 = x, x_2 = x^2, x_3 = x^3`$ ধরে normal linear regression চালাও। আরেক উদাহরণ: frontage × depth = area, এটা একটা নতুন feature।
- **Q: Normal equation method?** $`\theta = (X^TX)^{-1}X^Ty`$। Design matrix-এ $`x_0 = 1`$ column যোগ করতে হয়। α বা iteration লাগে না।
- **Q: GD vs Normal Eq?** Section 4-এর table দেখো।
- **Q: Non-invertibility?** Redundant features বা m ≤ n। সমাধান: feature delete, regularization, বা pinv।

---

## 10. Exam Tips / পরীক্ষার টিপস

1. **Table আগে বানাও:** x, y, xy, x² (R² লাগলে y²-ও)। তারপর Σ row লেখো। অর্ধেক নম্বর এখানেই।
2. **Formula আগে লিখে তারপর বসাও।** Partial marking-এ এটা কাজে দেয়।
3. **m-এর পর c:** $`c = \bar y - m\bar x`$। Sign খেয়াল রাখো, source-ও এখানে ভুল করেছে (Disney: c = −0.1)।
4. **Error = Predicted − Actual** ($`h - y`$)। এই convention মানলে update হয় $`\theta - \alpha\cdot(\dots)`$। উল্টো (y − h) নিলে update-এ + হবে, তাই mix করো না।
5. **Batch:** average (÷ m) নিয়ে **epoch-এ একবার** update। **Single:** কোনো ÷ m নেই, **প্রতি row-এর পর** update, আর পরের row-এর h **নতুন θ দিয়ে** হিসাব করতে হয়।
6. **Simultaneous update:** একটা row/epoch-এর সব θ **পুরোনো** θ দিয়ে হিসাব করো।
7. **θ₀-এর জন্য x₀ = 1**, তাই θ₀ update-এ শুধু Error (× 1)।
8. Calculator-এ অন্তত **4-6 decimal** রাখো। Tutorial table-এর সাথে মিলাতে হবে।
9. J বাড়লে লিখে দাও: **"α too large → overshoot/diverge"**। J কমলে লেখো: **"converging"**।
10. Regularization-এ **θ₀ regularize হয় না**।
11. R² interpretation-এ এক লাইন লেখো: "R² = 0.82, অর্থাৎ y-এর variation-এর 82% model explain করে"।

---

## 11. Practice Problem (Full Solution)

**Problem:** Data (x, y) = (1, 2), (2, 3), (3, 5), (4, 6)
(a) Least squares দিয়ে regression line বের করো এবং x = 5 হলে y predict করো।
(b) R² বের করো।
(c) θ₀ = θ₁ = 0, α = 0.1 নিয়ে **batch mode-এ 2 epoch** চালাও।
(d) একই initial values নিয়ে **single mode-এ 1 epoch** চালাও।

### (a) Least squares

| x | y | xy | x² | y² |
|---|---|---|---|---|
| 1 | 2 | 2 | 1 | 4 |
| 2 | 3 | 6 | 4 | 9 |
| 3 | 5 | 15 | 9 | 25 |
| 4 | 6 | 24 | 16 | 36 |
| **10** | **16** | **47** | **30** | **74** |

- $`n = 4`$, $`D = n\sum x^2 - (\sum x)^2 = 120 - 100 = 20`$
- $`m = \frac{4(47) - 10(16)}{20} = \frac{188 - 160}{20} = \mathbf{1.4}`$
- $`c = \bar y - m\bar x = 4 - 1.4(2.5) = 4 - 3.5 = \mathbf{0.5}`$ (check: $`\frac{16(30) - 10(47)}{20} = \frac{10}{20} = 0.5`$ ✔)
- Line: $`\boxed{y = 1.4x + 0.5}`$; x = 5 হলে **y = 7.5**

### (b) R²
$`\hat y`$ = 1.9, 3.3, 4.7, 6.1; residuals = 0.1, −0.3, 0.3, −0.1
SSE = 0.01 + 0.09 + 0.09 + 0.01 = **0.2**; $`\bar y = 4`$, SST = 4 + 1 + 1 + 4 = **10**
$`R^2 = 1 - 0.2/10 = \mathbf{0.98}`$ (r formula দিয়ে check: $`n\sum y^2 - (\sum y)^2 = 296 - 256 = 40`$, $`r = 28/\sqrt{20 \times 40} = 28/28.284 = 0.98995`$, তাই $`r^2 = 0.98`$ ✔)

### (c) Batch mode, α = 0.1

**Epoch 1** (θ = 0, 0): h = 0 সব row-এ

| x | y | h | Err | Err·x |
|---|---|---|---|---|
| 1 | 2 | 0 | −2 | −2 |
| 2 | 3 | 0 | −3 | −6 |
| 3 | 5 | 0 | −5 | −15 |
| 4 | 6 | 0 | −6 | −24 |
| | | Sum | −16 | −47 |
| | | **Avg** | **−4** | **−11.75** |

θ₀ = 0 − 0.1(−4) = **0.4**, θ₁ = 0 − 0.1(−11.75) = **1.175**

**Epoch 2** (0.4, 1.175):

| x | y | h | Err | Err·x |
|---|---|---|---|---|
| 1 | 2 | 1.575 | −0.425 | −0.425 |
| 2 | 3 | 2.75 | −0.25 | −0.5 |
| 3 | 5 | 3.925 | −1.075 | −3.225 |
| 4 | 6 | 5.1 | −0.9 | −3.6 |
| | | Sum | −2.65 | −7.75 |
| | | **Avg** | **−0.6625** | **−1.9375** |

θ₀ = 0.4 + 0.06625 = **0.46625**, θ₁ = 1.175 + 0.19375 = **1.36875**
J: 9.25 → 0.276 → 0.032। দ্রুত কমছে, θ ধীরে ধীরে exact (0.5, 1.4)-এর দিকে যাচ্ছে ✔

### (d) Single mode, α = 0.1

| Iter | x | y | h | Err | θ₀ = θ₀ − 0.1·Err | θ₁ = θ₁ − 0.1·Err·x |
|---|---|---|---|---|---|---|
| 1 | 1 | 2 | 0 | −2 | **0.2** | 0 + 0.2 = **0.2** |
| 2 | 2 | 3 | 0.2 + 0.4 = 0.6 | −2.4 | **0.44** | 0.2 + 0.48 = **0.68** |
| 3 | 3 | 5 | 0.44 + 2.04 = 2.48 | −2.52 | **0.692** | 0.68 + 0.756 = **1.436** |
| 4 | 4 | 6 | 0.692 + 5.744 = 6.436 | +0.436 | 0.692 − 0.0436 = **0.6484** | 1.436 − 0.1744 = **1.2616** |

1 epoch পরে θ₀ = **0.6484**, θ₁ = **1.2616**। শেষ row-এ error positive হয়েছে (overshoot), তাই θ একটু কমেছে। এটাই single mode-এর noisy/bouncing behaviour।
