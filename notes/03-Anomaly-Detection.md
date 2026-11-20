# 03 — Anomaly Detection (Gaussian Model) — Mid Exam Note

> **Teacher's note:** "Anomaly Detection ~4 marks — MAINLY MATH (covariance + ...). Study 15e Question List and 15d step-by-step."
> **Lecture-07 (14/11/26) note:** "Anomaly + Perceptron ⇒ Mainly **Math + Short Question** (4+4)। Anomaly ⇒ **Cov matrix দেওয়া থাকলে এর পরের part করতে হবে just, or, up to Cov matrix পর্যন্ত করতে হবে**।" → নিচের **Section 3A** এ দুইটা pattern-ই আলাদা করে practice করো।
> তাই এই note এর **core** হলো Section 3–6 এর math (Univariate + Multivariate Gaussian, covariance matrix, inverse)। Theory (Section 8) short answer হিসেবে পড়ো।

---

## 0. Quick Formula Sheet (এক নজরে সব formula)

| কী | Formula |
|---|---|
| Mean of feature $j$ | $\mu_j = \dfrac{1}{m}\sum_{i=1}^{m} x_j^{(i)}$ |
| Variance of feature $j$ | $\sigma_j^2 = \dfrac{1}{m}\sum_{i=1}^{m} (x_j^{(i)}-\mu_j)^2$ |
| Univariate Gaussian | $p(x;\mu,\sigma^2)=\dfrac{1}{\sqrt{2\pi}\,\sigma}\exp\!\left(-\dfrac{(x-\mu)^2}{2\sigma^2}\right)$ |
| Univariate (independent) model | $p(x)=\prod_{j=1}^{n} p(x_j;\mu_j,\sigma_j^2)$ |
| Covariance | $\text{Cov}(X,Y)=\dfrac{1}{m}\sum (X_i-\bar X)(Y_i-\bar Y)$ |
| Covariance matrix | $\Sigma = \dfrac{1}{m}\, a^T a$, যেখানে $a = M-\mu$ (deviation matrix) |
| 2×2 inverse | $\begin{bmatrix}a&b\\c&d\end{bmatrix}^{-1}=\dfrac{1}{ad-bc}\begin{bmatrix}d&-b\\-c&a\end{bmatrix}$ |
| Multivariate Gaussian | $p(x)=\dfrac{1}{(2\pi)^{n/2}\,\lvert\Sigma\rvert^{1/2}}\exp\!\left(-\dfrac12 (x-\mu)^T\Sigma^{-1}(x-\mu)\right)$ |
| Decision | $p(x_{test}) < \varepsilon \Rightarrow$ **Anomaly**; $p(x_{test}) \ge \varepsilon \Rightarrow$ **Normal** |

> **Note:** এই course এ (15d sheet) variance/covariance এ **$m$ দিয়ে ভাগ** করা হয় ($m-1$ না)। Exam এ সেটাই follow করো।
> $(x-\mu)^T\Sigma^{-1}(x-\mu)$ কে বলে **squared Mahalanobis distance**।

---

## 1. Anomaly Detection কী? (Core idea)

- **Anomaly / Outlier** = এমন data point যেটা বাকি normal data থেকে অনেক আলাদা।
- আমাদের কাছে শুধু **normal (non-faulty) data** থাকে → সেটা দিয়ে একটা probability model $p(x)$ বানাই।
- নতুন test point $x_{test}$ আসলে $p(x_{test})$ হিসাব করি:
  - $p(x_{test}) < \varepsilon$ → **Anomaly** (flag করো)
  - $p(x_{test}) \ge \varepsilon$ → **OK / Normal**
- $\varepsilon$ (epsilon) = threshold probability, আমরা নিজেরা ঠিক করি (CV set দিয়ে tune করি)।

এটা মূলত **unsupervised** problem (outliers সাধারণত labeled থাকে না), কিন্তু এর একটা **supervised flavor** আছে — কারণ normal data points গুলো "supervisor" এর মতো কাজ করে, বলে দেয় *কোনটা anomaly না*।

---

## 2. Gaussian (Normal) Distribution — Foundation

$x \sim \mathcal{N}(\mu, \sigma^2)$ মানে "x is distributed as Gaussian with mean $\mu$ and variance $\sigma^2$"।

- **$\mu$ (mean)** → bell curve এর **peak/center** কোথায় সেটা ঠিক করে।
- **$\sigma$ (standard deviation)** → curve কতটা **চওড়া (spread/width)** সেটা ঠিক করে। $\sigma$ বড় → curve চওড়া ও নিচু; $\sigma$ ছোট → সরু ও উঁচু।
- Curve এর নিচের **area সবসময় = 1**।
- $\mu$ থেকে যত দূরে যাবে, probability তত কমবে → তাই দূরের point = anomaly হওয়ার সম্ভাবনা বেশি।

**Parameter estimation:** $\mu$ = average, $\sigma^2$ = average squared deviation। এগুলো **Maximum Likelihood Estimates**। $1/m$ বা $1/(m-1)$ — practically খুব একটা পার্থক্য নাই।

---

## 3. MAIN WORKED PROBLEM (15d) — Aircraft Engine

> **Question:** An aircraft testing facility detects faulty aircraft using two features — engine heat ($x_1$) and vibration ($x_2$). Training set of 10 **non-faulty** aircraft:
> $(2,2), (3,4), (1,3), (4,4), (5,7), (6,9), (7,8), (10,14), (12,15), (15,19)$.
> New aircraft: engine heat = 10, vibration = 5 → $x_{test}=(10,5)$. Threshold $\varepsilon = 0.1$. Faulty or not?
> (Q1) Using **independent univariate Gaussian** model. (Q2) Using **multivariate Gaussian** model.

$n = 2$ features, $m = 10$ examples।

---

### Q1. Univariate (Independent) Gaussian Model — Step by Step

**Step 1: Mean of each feature**

$$\mu_1 = \frac{2+3+1+4+5+6+7+10+12+15}{10} = \frac{65}{10} = 6.5$$

$$\mu_2 = \frac{2+4+3+4+7+9+8+14+15+19}{10} = \frac{85}{10} = 8.5$$

**Step 2: Squared differences from mean**

| $x_1$ | $x_2$ | $d_1=x_1-6.5$ | $d_1^2$ | $d_2=x_2-8.5$ | $d_2^2$ |
|---|---|---|---|---|---|
| 2 | 2 | −4.5 | 20.25 | −6.5 | 42.25 |
| 3 | 4 | −3.5 | 12.25 | −4.5 | 20.25 |
| 1 | 3 | −5.5 | 30.25 | −5.5 | 30.25 |
| 4 | 4 | −2.5 | 6.25 | −4.5 | 20.25 |
| 5 | 7 | −1.5 | 2.25 | −1.5 | 2.25 |
| 6 | 9 | −0.5 | 0.25 | 0.5 | 0.25 |
| 7 | 8 | 0.5 | 0.25 | −0.5 | 0.25 |
| 10 | 14 | 3.5 | 12.25 | 5.5 | 30.25 |
| 12 | 15 | 5.5 | 30.25 | 6.5 | 42.25 |
| 15 | 19 | 8.5 | 72.25 | 10.5 | 110.25 |
| | | **Sum** | **186.5** | **Sum** | **298.5** |

**Step 3: Variance & Standard deviation** (divide by $m=10$)

$$\sigma_1^2 = \frac{186.5}{10} = 18.65,\qquad \sigma_1=\sqrt{18.65}\approx 4.32$$

$$\sigma_2^2 = \frac{298.5}{10} = 29.85,\qquad \sigma_2=\sqrt{29.85}\approx 5.46$$

**Step 4: Probability of each feature of $x_{test}=(10,5)$**

$$p(x_1=10;\ \mu_1=6.5,\ \sigma_1=4.32) = \frac{1}{\sqrt{2\pi}\cdot 4.32}\exp\!\left(-\frac{(10-6.5)^2}{2\times 18.65}\right)$$

- $\sqrt{2\pi} = 2.5066$ → $\dfrac{1}{2.5066\times4.32} = \dfrac{1}{10.829} = 0.09235$
- exponent: $\dfrac{(3.5)^2}{37.3} = \dfrac{12.25}{37.3} = 0.3284$ → $e^{-0.3284} = 0.72022$
- $p(x_1) = 0.09235 \times 0.72022 = 0.06651$

$$p(x_2=5;\ \mu_2=8.5,\ \sigma_2=5.46) = \frac{1}{\sqrt{2\pi}\cdot 5.46}\exp\!\left(-\frac{(5-8.5)^2}{2\times 29.85}\right)$$

- $\dfrac{1}{2.5066\times5.46} = \dfrac{1}{13.686} = 0.07307$
- exponent: $\dfrac{(-3.5)^2}{59.7} = \dfrac{12.25}{59.7} = 0.2052$ → $e^{-0.2052} = 0.81428$
- $p(x_2) = 0.07307\times0.81428 = 0.05950$

**Step 5: Multiply (independence assumption)**

$$p(x_{test}) = 0.09235 \times 0.72022 \times 0.07307 \times 0.81428 = \mathbf{0.00396}$$

**Step 6: Decision**

$0.00396 < \varepsilon = 0.1$ → **$x_{test}=(10,5)$ is an ANOMALY / Outlier → aircraft is FAULTY.** ✅

---

### Q2. Multivariate Gaussian Model — Step by Step (10 steps, exactly as 15d)

Multivariate model এ প্রথমে **variance-covariance matrix $\Sigma$** বানাতে হয়।

**Step (1): Mean vector**

$$\mu = \begin{bmatrix}6.5\\8.5\end{bmatrix}$$

**Step (2): Deviation matrix $a = M - \mu$** (প্রতিটা row থেকে mean বিয়োগ)

$$a = \begin{bmatrix}
-4.5 & -6.5\\ -3.5 & -4.5\\ -5.5 & -5.5\\ -2.5 & -4.5\\ -1.5 & -1.5\\ -0.5 & 0.5\\ 0.5 & -0.5\\ 3.5 & 5.5\\ 5.5 & 6.5\\ 8.5 & 10.5
\end{bmatrix}_{10\times2}$$

**Step (3): Transpose $a'$ ($2\times10$)**

$$a' = \begin{bmatrix}
-4.5 & -3.5 & -5.5 & -2.5 & -1.5 & -0.5 & 0.5 & 3.5 & 5.5 & 8.5\\
-6.5 & -4.5 & -5.5 & -4.5 & -1.5 & 0.5 & -0.5 & 5.5 & 6.5 & 10.5
\end{bmatrix}$$

**Step (4): $a' \cdot a$ ($2\times2$)**

- (1,1) = $\sum d_1^2 = 186.5$ (Q1 এর table থেকে)
- (2,2) = $\sum d_2^2 = 298.5$
- (1,2) = (2,1) = $\sum d_1 d_2$:

| $d_1$ | $d_2$ | $d_1 d_2$ |
|---|---|---|
| −4.5 | −6.5 | 29.25 |
| −3.5 | −4.5 | 15.75 |
| −5.5 | −5.5 | 30.25 |
| −2.5 | −4.5 | 11.25 |
| −1.5 | −1.5 | 2.25 |
| −0.5 | 0.5 | −0.25 |
| 0.5 | −0.5 | −0.25 |
| 3.5 | 5.5 | 19.25 |
| 5.5 | 6.5 | 35.75 |
| 8.5 | 10.5 | 89.25 |
| | **Sum** | **232.5** |

$$a'a = \begin{bmatrix}186.5 & 232.5\\ 232.5 & 298.5\end{bmatrix}$$

**Step (5): Covariance matrix $\Sigma = \dfrac{a'a}{m}$**

$$\Sigma = \frac{1}{10}\begin{bmatrix}186.5 & 232.5\\ 232.5 & 298.5\end{bmatrix} = \begin{bmatrix}18.65 & 23.25\\ 23.25 & 29.85\end{bmatrix}$$

> Diagonal = variances (18.65, 29.85) — Q1 এর সাথে মিলে গেছে ✔. Off-diagonal 23.25 **positive** → heat বাড়লে vibration ও বাড়ে (positive correlation)।

**Step (6): Determinant and its square root**

$$\lvert\Sigma\rvert = (18.65)(29.85) - (23.25)(23.25) = 556.7025 - 540.5625 = 16.14$$

$$\sqrt{\lvert\Sigma\rvert} = \sqrt{16.14} = 4.017$$

**Step (7): Inverse $\Sigma^{-1}$** (2×2 rule: swap $a,d$; negate $b,c$; divide by det)

$$\Sigma^{-1} = \frac{1}{16.14}\begin{bmatrix}29.85 & -23.25\\ -23.25 & 18.65\end{bmatrix} = 0.06196\begin{bmatrix}29.85 & -23.25\\ -23.25 & 18.65\end{bmatrix} \approx \begin{bmatrix}1.85 & -1.44\\ -1.44 & 1.16\end{bmatrix}$$

(Exact: 1.8494, −1.4405, 1.1555)

**Step (8): $x-\mu$**

$$x-\mu = \begin{bmatrix}10-6.5\\5-8.5\end{bmatrix} = \begin{bmatrix}3.5\\-3.5\end{bmatrix},\qquad (x-\mu)^T = [\,3.5\ \ -3.5\,]$$

$$(x-\mu)^T\Sigma^{-1} = [\,3.5\ \ -3.5\,]\begin{bmatrix}1.85 & -1.44\\ -1.44 & 1.16\end{bmatrix}$$

- 1st element: $3.5(1.85) + (-3.5)(-1.44) = 6.475 + 5.04 = 11.515$
- 2nd element: $3.5(-1.44) + (-3.5)(1.16) = -5.04 - 4.06 = -9.1$

$$(x-\mu)^T\Sigma^{-1} = [\,11.515\ \ -9.1\,]$$

**Step (9): $(x-\mu)^T\Sigma^{-1}(x-\mu)$** (squared Mahalanobis distance)

$$[\,11.515\ \ -9.1\,]\begin{bmatrix}3.5\\-3.5\end{bmatrix} = 40.3025 + 31.85 = \mathbf{72.1525}$$

(Exact inverse দিয়ে 72.103 আসে — rounding এর জন্য সামান্য পার্থক্য, কোনো সমস্যা নেই। Mahalanobis distance $=\sqrt{72.1}\approx 8.49$ — অনেক দূরে!)

**Step (10): $p(x)$**

$$p(10,5) = \frac{1}{(2\pi)^{2/2}\,\lvert\Sigma\rvert^{1/2}}\exp\!\left(-\frac12\times72.1525\right) = \frac{1}{2\times3.14\times4.017}\times e^{-36.076}$$

- $\dfrac{1}{2\times3.14\times4.017} = \dfrac{1}{25.227} \approx 0.04$
- $e^{-36.076} \approx 2.149\times10^{-16}$
- $p = 0.04 \times 2.149\times10^{-16} \approx \mathbf{8.6\times10^{-18}}$

(More precise calc দিলে $\approx 8.7\times10^{-18}$ — একই conclusion।)

**Decision:** $8.6\times10^{-18} \ll \varepsilon = 0.1$ → **(10, 5) is an ANOMALY / Outlier → Faulty aircraft.** ✅

> **কেন multivariate এ probability এত ছোট?** Training data তে heat আর vibration একসাথে বাড়ে (positive covariance)। কিন্তু test point এ heat বেশি (10) অথচ vibration কম (5) — এই **combination** টাই অস্বাভাবিক। Multivariate model correlation ধরতে পারে, তাই অনেক strongly flag করে।

---

## 3A. Exam Pattern Drills (Lecture-07) — দুই ধরনের প্রশ্ন

Teacher বলেছেন exam এ পুরো 10 step নাও আসতে পারে। দুইটা pattern:

| Pattern | প্রশ্নে কী দেওয়া থাকবে | তোমাকে কী করতে হবে |
|---|---|---|
| **A — "Cov matrix দেওয়া আছে"** | $\mu$, $\Sigma$, test point $x$, $\varepsilon$ | Step 6–10: $\lvert\Sigma\rvert$ → $\Sigma^{-1}$ → $(x-\mu)^T\Sigma^{-1}(x-\mu)$ → $p(x)$ → decision |
| **B — "Up to Cov matrix"** | শুধু training data | Step 1–5: mean → deviation $a$ → $a'a$ → $\Sigma = a'a/m$ |

### Pattern A Drill — $\Sigma$ given, শুধু পরের part

**Given:** $\mu = \begin{bmatrix}5\\10\end{bmatrix}$, $\Sigma = \begin{bmatrix}4 & 2\\ 2 & 3\end{bmatrix}$, $\varepsilon = 0.01$. Test points: $x_1 = (8, 7)$ আর $x_2 = (6, 11)$।

**Step 6 — Determinant:** $\lvert\Sigma\rvert = 4\times3 - 2\times2 = 12 - 4 = 8$, $\sqrt{8} = 2.828$

**Step 7 — Inverse:** $\Sigma^{-1} = \dfrac{1}{8}\begin{bmatrix}3 & -2\\ -2 & 4\end{bmatrix} = \begin{bmatrix}0.375 & -0.25\\ -0.25 & 0.5\end{bmatrix}$

**Constant part:** $\dfrac{1}{(2\pi)^{2/2}\sqrt{8}} = \dfrac{1}{2\pi \times 2.828} = \dfrac{1}{17.77} = 0.05627$

**Test point $x_1 = (8, 7)$:**
- Step 8: $x-\mu = [3,\ -3]$
- $(x-\mu)^T\Sigma^{-1} = [\,3(0.375) + (-3)(-0.25),\ \ 3(-0.25) + (-3)(0.5)\,] = [\,1.875,\ -2.25\,]$
- Step 9: $[1.875,\ -2.25]\cdot[3,\ -3] = 5.625 + 6.75 = \mathbf{12.375}$
- Step 10: $p = 0.05627 \times e^{-12.375/2} = 0.05627 \times e^{-6.1875} = 0.05627 \times 0.002055 = \mathbf{1.16\times10^{-4}}$
- **Decision:** $1.16\times10^{-4} < 0.01$ → **ANOMALY** ✅

**Test point $x_2 = (6, 11)$:**
- $x-\mu = [1,\ 1]$
- $(x-\mu)^T\Sigma^{-1} = [\,0.375 - 0.25,\ \ -0.25 + 0.5\,] = [\,0.125,\ 0.25\,]$
- $[0.125,\ 0.25]\cdot[1,\ 1] = \mathbf{0.375}$
- $p = 0.05627 \times e^{-0.1875} = 0.05627 \times 0.829 = \mathbf{0.0466}$
- **Decision:** $0.0466 \ge 0.01$ → **NORMAL** ✅

> **খেয়াল করো:** $x_1$ এ feature 1 বেড়েছে কিন্তু feature 2 কমেছে — অথচ covariance positive (2), মানে দুইটা সাধারণত একসাথে বাড়ে। তাই এই combination অস্বাভাবিক → anomaly।

### Pattern B Drill — শুধু Cov matrix পর্যন্ত

**Given (m = 4):** $(2,4),\ (4,6),\ (6,10),\ (8,8)$

**Step 1 — Mean:** $\mu_1 = \dfrac{2+4+6+8}{4} = 5$, $\mu_2 = \dfrac{4+6+10+8}{4} = 7$

**Step 2–4 — Deviation আর products:**

| $x_1$ | $x_2$ | $d_1 = x_1-5$ | $d_2 = x_2-7$ | $d_1^2$ | $d_2^2$ | $d_1 d_2$ |
|---|---|---|---|---|---|---|
| 2 | 4 | −3 | −3 | 9 | 9 | 9 |
| 4 | 6 | −1 | −1 | 1 | 1 | 1 |
| 6 | 10 | 1 | 3 | 1 | 9 | 3 |
| 8 | 8 | 3 | 1 | 9 | 1 | 3 |
| | | | **Sum** | **20** | **20** | **16** |

$$a'a = \begin{bmatrix}20 & 16\\ 16 & 20\end{bmatrix}$$

**Step 5 — Covariance matrix ($\div m = 4$):**

$$\Sigma = \frac{1}{4}\begin{bmatrix}20 & 16\\ 16 & 20\end{bmatrix} = \begin{bmatrix}5 & 4\\ 4 & 5\end{bmatrix}$$

> Check: $\Sigma$ সবসময় **symmetric** হবে, আর diagonal (variance) কখনো negative হবে না। এখানে off-diagonal 4 positive → দুইটা feature একসাথে বাড়ে।

---

## 4. How to Find a Covariance Matrix (15b) — Math/English/Art Example

**Concept:**
- **Variance** = spread of one feature: $\text{Var}(X) = \frac{\sum (X_i-\bar X)^2}{N}$
- **Covariance** = দুইটা feature একসাথে কোন দিকে move করে: $\text{Cov}(X,Y)=\frac{\sum (X_i-\bar X)(Y_i-\bar Y)}{N}$
- **Covariance matrix** ($n\times n$, symmetric): **diagonal = variances**, **off-diagonal $(i,j)$ = Cov(feature i, feature j)**।
  - Cov > 0 → একসাথে বাড়ে; Cov < 0 → উল্টো দিকে; Cov = 0 → কোনো linear relation নেই।

**Procedure (3 steps):** (1) $a = A - \bar{A}$ (deviation scores) → (2) $a'a$ → (3) $V = a'a / n$

**Data** (5 students, 3 tests):

$$A = \begin{bmatrix}90&60&90\\90&90&30\\60&60&60\\60&60&90\\30&30&30\end{bmatrix}$$

**Step 1: Means** — Math $=\frac{90+90+60+60+30}{5}=66$, English $=\frac{300}{5}=60$, Art $=\frac{300}{5}=60$

$$a = A-\bar A = \begin{bmatrix}24&0&30\\24&30&-30\\-6&0&0\\-6&0&30\\-36&-30&-30\end{bmatrix}$$

**Step 2: $a'a$**

- $(1,1) = 24^2+24^2+6^2+6^2+36^2 = 576+576+36+36+1296 = 2520$
- $(2,2) = 0+900+0+0+900 = 1800$
- $(3,3) = 900+900+0+900+900 = 3600$
- $(1,2) = 24\cdot0 + 24\cdot30 + (-6)\cdot0 + (-6)\cdot0 + (-36)(-30) = 720+1080 = 1800$
- $(1,3) = 24\cdot30 + 24(-30) + 0 + (-6)(30) + (-36)(-30) = 720-720-180+1080 = 900$
- $(2,3) = 0 + 30(-30) + 0 + 0 + (-30)(-30) = -900+900 = 0$

$$a'a=\begin{bmatrix}2520&1800&900\\1800&1800&0\\900&0&3600\end{bmatrix}$$

**Step 3: $V = a'a/5$**

$$V = \begin{bmatrix}504&360&180\\360&360&0\\180&0&720\end{bmatrix}$$

**Interpretation:** Art এর variance সবচেয়ে বেশি (720) → সবচেয়ে বেশি spread; English সবচেয়ে কম (360)। Math–English (360), Math–Art (180) positive → একসাথে বাড়ে। English–Art = 0 → কোনো predictable relation নেই।

---

## 5. How to Find the Inverse of a Matrix (15c)

**Definition:** $A\,A^{-1} = A^{-1}A = I$ (Identity matrix — diagonal এ 1, বাকি 0)।
Inverse থাকতে হলে matrix **square** হতে হবে এবং **determinant ≠ 0**। Det = 0 হলে matrix **singular** (non-invertible)।

### 5.1 2×2 Inverse

$$\begin{bmatrix}a&b\\c&d\end{bmatrix}^{-1}=\frac{1}{ad-bc}\begin{bmatrix}d&-b\\-c&a\end{bmatrix}$$

মনে রাখো: **swap $a$ & $d$, negate $b$ & $c$, divide by determinant.**

**Example:** $A=\begin{bmatrix}4&7\\2&6\end{bmatrix}$, det $=4\cdot6-7\cdot2=24-14=10$

$$A^{-1}=\frac{1}{10}\begin{bmatrix}6&-7\\-2&4\end{bmatrix}=\begin{bmatrix}0.6&-0.7\\-0.2&0.4\end{bmatrix}$$

Check: $4(0.6)+7(-0.2)=2.4-1.4=1$ ✔, $4(-0.7)+7(0.4)=0$ ✔ → $AA^{-1}=I$.

**Singular example:** $\begin{bmatrix}3&4\\6&8\end{bmatrix}$ → det $=24-24=0$ → **no inverse** (2nd row = 2 × 1st row, নতুন কোনো information নেই)।

### 5.2 3×3 Inverse (Minors → Cofactors → Adjugate → ×1/Det)

**Example:** $A=\begin{bmatrix}3&0&2\\2&0&-2\\0&1&1\end{bmatrix}$

**Step 1: Matrix of Minors** (প্রতিটা element এর row & column বাদ দিয়ে বাকি 2×2 এর determinant)

- $M_{11}=0\cdot1-(-2)\cdot1=2$, $M_{12}=2\cdot1-(-2)\cdot0=2$, $M_{13}=2\cdot1-0\cdot0=2$
- $M_{21}=0\cdot1-2\cdot1=-2$, $M_{22}=3\cdot1-2\cdot0=3$, $M_{23}=3\cdot1-0\cdot0=3$
- $M_{31}=0\cdot(-2)-2\cdot0=0$, $M_{32}=3(-2)-2\cdot2=-10$, $M_{33}=3\cdot0-0\cdot2=0$

$$\text{Minors}=\begin{bmatrix}2&2&2\\-2&3&3\\0&-10&0\end{bmatrix}$$

**Step 2: Cofactors** — checkerboard sign $\begin{bmatrix}+&-&+\\-&+&-\\+&-&+\end{bmatrix}$ apply:

$$\text{Cofactors}=\begin{bmatrix}2&-2&2\\2&3&-3\\0&10&0\end{bmatrix}$$

**Step 3: Adjugate** = Transpose of cofactors:

$$\text{adj}(A)=\begin{bmatrix}2&2&0\\-2&3&10\\2&-3&0\end{bmatrix}$$

**Step 4: Determinant** (top row × its cofactors): $3(2)+0(-2)+2(2)=10$

$$A^{-1}=\frac{1}{10}\,\text{adj}(A)=\begin{bmatrix}0.2&0.2&0\\-0.2&0.3&1\\0.2&-0.3&0\end{bmatrix}$$

(Verified: $AA^{-1}=I$ ✔)

---

## 6. Practice Problem (3 Features) — Full Solution

> **Question:** A server is monitored with 3 features $(x_1, x_2, x_3)$. Training set of 4 normal observations:
> $(1,2,3),\ (3,4,3),\ (5,4,5),\ (3,6,5)$.
> Test point $x_{test}=(4,5,3)$, $\varepsilon = 0.005$.
> (a) Univariate model দিয়ে decide করো। (b) Multivariate model দিয়ে decide করো।

$n=3$, $m=4$.

**Step 1: Mean vector**

$\mu_1=\frac{1+3+5+3}{4}=3$, $\mu_2=\frac{2+4+4+6}{4}=4$, $\mu_3=\frac{3+3+5+5}{4}=4$ → $\mu=[3,4,4]^T$

**Step 2: Deviation matrix**

$$a=\begin{bmatrix}-2&-2&-1\\0&0&-1\\2&0&1\\0&2&1\end{bmatrix}$$

**Step 3: $a'a$**

- $(1,1)=4+0+4+0=8$, $(2,2)=4+0+0+4=8$, $(3,3)=1+1+1+1=4$
- $(1,2)=(-2)(-2)+0+0+0=4$
- $(1,3)=(-2)(-1)+0+(2)(1)+0=4$
- $(2,3)=(-2)(-1)+0+0+(2)(1)=4$

$$a'a=\begin{bmatrix}8&4&4\\4&8&4\\4&4&4\end{bmatrix}\quad\Rightarrow\quad \Sigma=\frac{a'a}{4}=\begin{bmatrix}2&1&1\\1&2&1\\1&1&1\end{bmatrix}$$

So $\sigma_1^2=2,\ \sigma_2^2=2,\ \sigma_3^2=1$.

**(a) Univariate model** — $x_{test}-\mu=(1,1,-1)$

- $p(x_1)=\frac{1}{\sqrt{2\pi\cdot2}}e^{-1^2/(2\cdot2)} = 0.28209\times e^{-0.25}=0.28209\times0.77880=0.21970$
- $p(x_2)$ = same $=0.21970$
- $p(x_3)=\frac{1}{\sqrt{2\pi\cdot1}}e^{-1/2}=0.39894\times0.60653=0.24197$
- $p(x)=0.21970\times0.21970\times0.24197\approx\mathbf{0.01168}$

$0.01168 \ge 0.005$ → Univariate বলে **Normal** ❌ (miss করলো!)

**(b) Multivariate model**

*Determinant* (1st row expansion):
$|\Sigma| = 2(2\cdot1-1\cdot1) - 1(1\cdot1-1\cdot1) + 1(1\cdot1-2\cdot1) = 2(1) - 1(0) + 1(-1) = 1$

*Minors:*
$M_{11}=2\cdot1-1\cdot1=1,\ M_{12}=1\cdot1-1\cdot1=0,\ M_{13}=1\cdot1-2\cdot1=-1$
$M_{21}=1\cdot1-1\cdot1=0,\ M_{22}=2\cdot1-1\cdot1=1,\ M_{23}=2\cdot1-1\cdot1=1$
$M_{31}=1\cdot1-1\cdot2=-1,\ M_{32}=2\cdot1-1\cdot1=1,\ M_{33}=2\cdot2-1\cdot1=3$

*Cofactors* (checkerboard): $\begin{bmatrix}1&0&-1\\0&1&-1\\-1&-1&3\end{bmatrix}$ — symmetric, তাই adjugate একই।

$$\Sigma^{-1}=\frac{1}{1}\begin{bmatrix}1&0&-1\\0&1&-1\\-1&-1&3\end{bmatrix}$$

*Quadratic form:* $x-\mu=[1,1,-1]^T$

$(x-\mu)^T\Sigma^{-1} = [\,1+0+1,\ \ 0+1+1,\ \ -1-1-3\,] = [\,2,\ 2,\ -5\,]$

$(x-\mu)^T\Sigma^{-1}(x-\mu) = 2(1)+2(1)+(-5)(-1) = 9$

*Probability:*

$$p(x)=\frac{1}{(2\pi)^{3/2}\sqrt{1}}e^{-9/2} = 0.06349\times e^{-4.5} = 0.06349\times0.011109 \approx \mathbf{0.000705}$$

$0.000705 < 0.005$ → Multivariate বলে **ANOMALY** ✅

> **Lesson:** প্রতিটা feature আলাদাভাবে দেখলে মাত্র 1 unit দূরে (normal মনে হয়), কিন্তু data তে $x_3$ বাড়লে $x_1, x_2$ ও বাড়ে — test point এ $x_1,x_2$ বেশি অথচ $x_3$ কম, এই **unusual combination** শুধু multivariate model ধরতে পারলো। (এটাই 15e এর "multivariate detects, univariate fails" প্রশ্নের numerical example।)

---

## 7. Algorithms (Formal) — অবশ্যই লিখতে পারতে হবে

### 7.1 Anomaly Detection Algorithm — Univariate Gaussian (***)

1. **Choose features** $x_j$ that might be indicative of anomalous examples.
2. **Fit parameters** $\mu_1,\dots,\mu_n,\ \sigma_1^2,\dots,\sigma_n^2$ from training set $\{x^{(1)},\dots,x^{(m)}\}$:
   $$\mu_j=\frac1m\sum_{i=1}^m x_j^{(i)},\qquad \sigma_j^2=\frac1m\sum_{i=1}^m\left(x_j^{(i)}-\mu_j\right)^2$$
3. **Given new example $x$, compute**
   $$p(x)=\prod_{j=1}^n p(x_j;\mu_j,\sigma_j^2)=\prod_{j=1}^n\frac{1}{\sqrt{2\pi}\sigma_j}\exp\!\left(-\frac{(x_j-\mu_j)^2}{2\sigma_j^2}\right)$$
4. **Anomaly if** $p(x)<\varepsilon$, else normal.

### 7.2 Anomaly Detection Algorithm — Multivariate Gaussian (***)

1. **Fit model:** training set থেকে
   $$\mu=\frac1m\sum_{i=1}^m x^{(i)}\ (n\times1),\qquad \Sigma=\frac1m\sum_{i=1}^m (x^{(i)}-\mu)(x^{(i)}-\mu)^T\ (n\times n)$$
2. **Given new example $x$, compute**
   $$p(x)=\frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}}\exp\!\left(-\frac12(x-\mu)^T\Sigma^{-1}(x-\mu)\right)$$
   (Dimension check: $(1\times n)(n\times n)(n\times1)=1\times1$ scalar।)
3. **Flag anomaly if** $p(x)<\varepsilon$; else OK.

---

## 8. 15e Question List — Answers (Bangla + English)

### Q1. What do you understand by Anomaly detection? What are its applications?
**Anomaly detection** হলো normal data দিয়ে একটা probability model $p(x)$ বানানো, তারপর নতুন point এর $p(x)<\varepsilon$ হলে সেটাকে anomaly/outlier হিসেবে flag করা। এটা mostly unsupervised (anomaly labeled থাকে না), কিন্তু normal data supervisor এর কাজ করে।
**Applications:**
- **Fraud detection** — login/web access/credit card usage: online time, login location, login frequency, spending frequency দিয়ে normal user model বানাও; weird user কে flag/block করো।
- **Manufacturing / QA** — aircraft engine (heat, vibration) test।
- **Data center monitoring** — memory use, disk access/sec, CPU load, network traffic দিয়ে failure হওয়ার আগে anomalous machine ধরা।

### Q2. Role of Gaussian distribution; role of μ and σ.
Normal data কে আমরা ধরে নেই Gaussian $\mathcal N(\mu,\sigma^2)$ follow করে, তাই প্রতিটা point এর probability $p(x)$ হিসাব করা যায় — mean এর কাছে probability বেশি, দূরে গেলে কম → দূরের point = anomaly। **$\mu$** curve এর **peak/center** ঠিক করে; **$\sigma$** curve এর **width/spread** ঠিক করে (বড় $\sigma$ = চওড়া, নিচু; ছোট $\sigma$ = সরু, উঁচু)। Total area সবসময় 1।

### Q3. *** Formal algorithm using univariate Gaussian model.
→ Section 7.1 দেখো (choose features → fit $\mu_j,\sigma_j^2$ → compute $p(x)=\prod p(x_j)$ → $p(x)<\varepsilon$ হলে anomaly)।

### Q4. * Three considerations with univariate Gaussian algorithm.
1. **How to choose features** — এমন features নাও যেগুলো anomaly তে unusually large/small value দেয়; general properties describe করে।
2. **How to fit parameters** — প্রতিটা feature এর $\mu_j$ ও $\sigma_j^2$ calculate করা (vectorized implementation better)।
3. **How to compute $p(x)$** for test point — Gaussian formula দিয়ে product; খুব ছোট হলে normal হওয়ার chance কম → anomaly।

### Q5. Issues in developing & evaluating an anomaly detection system (labelling, splitting, evaluation, choosing ε).
- **Data labelling:** Evaluate করার জন্য কিছু labeled data লাগে — normal $y=0$, anomalous $y=1$। Training set এ শুধু normal (কয়েকটা anomaly থাকলেও OK)।
- **Splitting (e.g. 10000 good, 20 flawed engines):** Training = 6000 good; **CV** = 2000 good + 10 anomalous; **Test** = 2000 good + 10 anomalous (ratio 3:1:1)। CV আর Test এ **একই** data ব্যবহার করা **bad practice**।
- **Evaluation:** Training set এ $p(x)$ fit করো; CV/Test এ predict $y=1$ if $p(x)<\varepsilon$। Data খুব **skewed** ($y=0$ অনেক বেশি), তাই **accuracy খারাপ metric** (সব সময় $y=0$ বললেও ~99.5% accuracy, 2000/2010)। ব্যবহার করো **TP/FP/FN/TN, Precision/Recall, F1-score**।
- **Choosing ε:** CV set এ বিভিন্ন $\varepsilon$ try করো, যেটা **F1-score maximize** করে সেটা নাও; final evaluation **test set** এ।

### Q6. * When to choose anomaly detection vs supervised learning? Examples.
| Anomaly Detection | Supervised Learning |
|---|---|
| Positive (anomaly) examples **খুব কম** (0–20), negative অনেক বেশি | Positive ও negative দুইটাই **reasonably অনেক** |
| **Many different types** of anomalies; future anomaly আগের গুলোর মতো নাও হতে পারে ("unknown unknowns") | Future positives আগের positives এর **মতোই** দেখতে হবে |
| Examples: **fraud detection**, **manufacturing** (aircraft engine), **data center monitoring** | Examples: **email spam**, **weather prediction**, **cancer classification** |

Spam এর অনেক type থাকলেও common keywords থাকে এবং positive example যথেষ্ট → supervised। Huge online retailer যার কাছে অনেক fraud example আছে, বা huge volume manufacturing → supervised এ shift করতে পারে।

### Q7. * How to deal with non-Gaussian features?
প্রথমে **histogram** plot করে দেখো data Gaussian কিনা (sanity check)। না হলে transformation দিয়ে Gaussian-এর মতো বানাও:
- $\log(x)$
- $\log(x+c)$ — shifted log (বিশেষ করে $x\le 0$ থাকলে); $c$ adjust করো
- $x^{1/2},\ x^{1/3},\ x^{1/k}$ (power/root transform)

(Non-Gaussian হলেও algorithm প্রায়ই কাজ করে, কিন্তু transform করলে ভালো।)

### Q8. * Example of feature engineering; assumptions of univariate model.
**Feature engineering (error analysis):** CV set এ যেগুলো ভুল হয়েছে দেখো, কেন ভুল হলো বুঝে নতুন feature বানাও।
Example — data center: $x_3$ = CPU load, $x_4$ = network traffic; normally দুটো linearly একসাথে বাড়ে। নতুন feature $x_5 = \dfrac{\text{CPU load}}{\text{network traffic}}$:
- High CPU, low network → **infinite loop / deadlock** (hang হবে)
- Low CPU, high network → **BOTNET / malicious host**
এই ratio unusual হলে anomaly ধরা পড়ে। (Feature scaling লাগতে পারে।)

**Assumptions of univariate model:** (1) প্রতিটা feature **Gaussian distributed**, (2) features **independent** (তাই probability গুলো গুণ করা হয়) — contours **axis-aligned**, কোনো correlation capture করে না।

### Q9. *** Example where multivariate detects outlier but univariate fails.
Data center: $x_1$ = CPU load, $x_2$ = memory use — দুটো positively correlated (data একটা তির্যক ellipse এ থাকে)। Test point: CPU load **low** কিন্তু memory **high** (e.g. $x_1=0.4,\ x_2=1.5$)। আলাদাভাবে দেখলে প্রতিটা value acceptable range এ → univariate model এর **concentric circle (axis-aligned)** contour এ এটা normal point এর মতো একই probability পায় → **miss করে**। Multivariate model $\Sigma$ এর off-diagonal (covariance) দিয়ে **tilted ellipse** বানায় → point টা ellipse এর বাইরে → **anomaly** ধরে।
Numerical proof: Section 6 practice — univariate $p=0.0117 \ge 0.005$ (normal) কিন্তু multivariate $p=0.000705<0.005$ (anomaly)। Section 3 এও একই idea: (10,5) এ heat high, vibration low।

### Q10. *** Formal algorithm using multivariate Gaussian model.
→ Section 7.2 দেখো (fit $\mu, \Sigma$ → compute $p(x)$ with $\Sigma^{-1}$ and $|\Sigma|$ → compare with $\varepsilon$)।

### Q11. Relation between univariate and multivariate models.
Univariate model হলো multivariate এর একটা **special case** — যখন $\Sigma$ **diagonal** (off-diagonal সব 0, অর্থাৎ $\text{Cov}(i,j)=0$, no correlation):
$$\Sigma=\begin{bmatrix}\sigma_1^2&0&\cdots\\0&\sigma_2^2&\cdots\\ \vdots&&\ddots\end{bmatrix}\ \Rightarrow\ p_{multi}(x)=\prod_j p(x_j;\mu_j,\sigma_j^2)$$
তখন দুই model **identical**; contour গুলো **axis-aligned**।

### Q12. * Compare univariate vs multivariate; non-invertible matrix issue.
| Univariate (Original) | Multivariate |
|---|---|
| বেশি ব্যবহৃত | কম ব্যবহৃত |
| Correlation ধরতে পারে না → unusual combination ধরতে **manually extra features** বানাতে হয় ($x_1/x_2$) | **Automatically correlation capture** করে, extra feature লাগে না |
| Computationally **cheap** (no covariance, no inverse) | **Expensive** — $n\times n$ $\Sigma^{-1}$ লাগে |
| Huge $n$ (100,000) তেও scale করে | Large $n$ এ খারাপ |
| **Small training set** এও কাজ করে (m = 50, 100) | Must have **$m > n$** (ideally $m \gg n$, e.g. $m\ge10n$), নাহলে $\Sigma$ singular |

**Non-invertible (singular) $\Sigma$ — কারণ ও সমাধান:**
1. **$m < n$** (data কম) → original univariate model ব্যবহার করো, বা বেশি data নাও।
2. **Redundant / linearly dependent features** (e.g. $x_2 = x_1$, $x_3 = x_4 + x_5$) → redundant feature বাদ দাও, বা **PCA** দিয়ে dimension কমাও; irrelevant features sanity check করে remove করো।

---

## 9. Exam Tips / পরীক্ষার টিপস

1. **Variance/covariance এ $m$ দিয়ে ভাগ করো** (15d sheet এ তাই করা)। প্রশ্নে অন্য কিছু বললে সেটা follow করো।
2. **Table বানাও** — $x$, $d=x-\mu$, $d^2$, $d_1d_2$ columns; sum গুলো neat করে লেখো। এতে partial marks নিশ্চিত।
3. **Covariance matrix check:** diagonal = variances (Univariate এর $\sigma^2$ এর সাথে মিলবে), matrix **symmetric** হবে।
4. **2×2 inverse:** swap $a,d$ → negate $b,c$ → ÷ det। শেষে $\Sigma\Sigma^{-1}\approx I$ দ্রুত check করো।
5. **Univariate formula এ $\sigma$ (SD)** denominator এ, exponent এ **$2\sigma^2$ (variance)** — mix up করো না।
6. **Multivariate formula এ $|\Sigma|^{1/2}$** (det এর square root) আর $(2\pi)^{n/2}$ — $n=2$ হলে simply $2\pi$।
7. Quadratic form step-by-step: আগে $(x-\mu)^T\Sigma^{-1}$ (row vector), তারপর $\times(x-\mu)$ → একটা scalar।
8. **Final line সবসময় লেখো:** "$p(x_{test}) = \dots < \varepsilon = \dots$ → Anomaly / Faulty" (বা ≥ → Normal)।
9. Main problem এর key numbers মুখস্থ রাখো: $\mu=(6.5, 8.5)$, $\sigma=(4.32, 5.46)$, $\Sigma=\begin{bmatrix}18.65&23.25\\23.25&29.85\end{bmatrix}$, $|\Sigma|=16.14$, $\Sigma^{-1}\approx\begin{bmatrix}1.85&-1.44\\-1.44&1.16\end{bmatrix}$, quad form = 72.15, $p_{uni}=0.00396$, $p_{multi}\approx8.6\times10^{-18}$।
10. Theory তে *** marked: **univariate algorithm, multivariate algorithm, multivariate-detects-univariate-fails example** — এগুলো অবশ্যই।
11. Calculator এ $e^{-x}$ ঠিক মতো চাপো; খুব ছোট number scientific notation এ লেখো ($8.6\times10^{-18}$)।
