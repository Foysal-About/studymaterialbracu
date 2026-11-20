# 04 — Perceptron (Linear Decision Boundary)

> **Exam focus (teacher's note):** "02 Perceptron ~4 marks — mainly **MATH**." Lecture-07: "Anomaly + Perceptron ⇒ Mainly Math + **Short Que** (4+4)" → Section **6A** এ short questions আছে।
> Teacher board-এ একটা straight line দিয়ে `x` আর `–` points আলাদা করে লিখেছেন: **"limitation – linear"**।
> মানে: (1) step-by-step weight update math পারতে হবে, (2) decision boundary বুঝতে হবে, (3) limitation (linearly separable না হলে কাজ করে না, যেমন XOR) জানতে হবে।

---

## 1. Perceptron কী? (Idea)

**Perceptron** হলো সবচেয়ে simple neural network / linear classifier — এটা একটা straight line (2D-তে), plane (3D-তে) বা **hyperplane** (high dimension-এ) দিয়ে দুইটা class আলাদা করে।

- Input: pattern $x = [x_1 \; x_2]^T$ (দুইটা feature)
- Output: class $\omega_1$ নাকি $\omega_2$
- কাজ: এমন **weight vector** খুঁজে বের করা যাতে line-এর এক পাশে সব class 1, আরেক পাশে সব class 2 থাকে।

Perceptron algorithm-এর কাজ হলো **weights শেখা (learn করা)** — ভুল হলে weight একটু adjust করো, আবার চেক করো, যতক্ষণ না সব ঠিক হয়।

---

## 2. Key Terms & Formulas

### 2.1 Linear Discriminant Function $g(x)$

Straight line-এর general equation: $ax_1 + bx_2 + c = 0$

Perceptron-এর ভাষায়:

$$g(x) = w^T x + w_0 = w_1x_1 + w_2x_2 + w_0$$

| Symbol | মানে |
|---|---|
| $w = [w_1 \; w_2]^T$ | weight vector (line-এর slope/direction ঠিক করে) — line equation-এর $a, b$ |
| $w_0$ | **bias** / constant term — line-এর $c$ (line-টাকে shift করে, y-intercept-এর মতো) |
| $x$ | feature vector |

**Decision rule:**

- $g(x) > 0 \Rightarrow x \in \omega_1$ (Class 1)
- $g(x) < 0 \Rightarrow x \in \omega_2$ (Class 2)
- $g(x) = 0 \Rightarrow$ point টা ঠিক **decision boundary**-র উপর (দুই class equally likely)

> **Decision Boundary** = সেই সব point-এর set যেখানে $g(x) = 0$। এটাই আমাদের line।

### 2.2 Augmented Vector (bias-কে vector-এর ভিতরে ঢোকানো)

$w_0$ আলাদা করে না লিখে, $x$-এর শেষে একটা **1** যোগ করি:

$$g(x) = w^Tx + w_0 = [w_1 \; w_2 \; w_0] \begin{bmatrix} x_1 \\ x_2 \\ 1 \end{bmatrix}$$

- $[x_1 \; x_2 \; 1]^T$ = **Augmented Feature Vector** (শেষের 1 টাকে $x_0 = 1$ বলে)
- $[w_1 \; w_2 \; w_0]$ = **Augmented Weight Vector**
- শেষ component **সবসময় 1** হতে হবে — তাহলে $w_0 \times 1 = w_0$ থাকে, অর্থাৎ constant term কোনো feature-এর উপর depend করে না।

### 2.3 Activation (perceptron-এর output)

$$\text{output} = \begin{cases} \omega_1 & \text{if } g > 0 \\ \omega_2 & \text{if } g < 0 \end{cases}$$

এটা একটা **step / sign activation function** — শুধু sign (positive/negative) দেখা হয়।

---

## 3. Perceptron Algorithm Steps (Tutorial-এর method)

### Step 1: High Dimensional Projection ($\varphi$ function)

Low dimension-এ যেটা linearly separable না, high dimension-এ সেটা প্রায়ই linearly separable হয়ে যায়। তাই tutorial-এ প্রতিটা 2D point-কে **6D**-তে project করা হয়েছে:

$$y = \varphi(x) = [\,x_1^2 \quad x_2^2 \quad x_1x_2 \quad x_1 \quad x_2 \quad 1\,]^T$$

- $y$ = **High Dimensional Augmented Feature Vector** (শেষে 1 আবশ্যক)
- $a^T = [w_5 \; w_4 \; w_3 \; w_2 \; w_1 \; w_0]$ = **High Dimensional Augmented Weight Vector** (y-এর মতোই 6টা component)
- $g(y) = a^T y$

> Note: Weight vector আর feature vector-এর **component সংখ্যা সবসময় সমান** হবে।

### Step 2: Class Normalization (Reflection)

দুইটা condition ($g>0$ আর $g<0$) handle না করে, **Class 2-এর সব pattern-কে negate (–y) করে দিই**। কারণ:

$$g(y) < 0 \iff g(-y) > 0$$

তাহলে শুধু **একটাই condition** থাকে: **সব training pattern-এর জন্য $a^Ty > 0$ হতে হবে।**

Class 2-কে এভাবে reflect করাকে বলে **"Normalizing"** class 2। (Reflected vectors-কে "Reflected Prototypes" বলে।)

### Step 3: Initialization

- **Learning rate** $\alpha$: range $0 < \alpha \le 1$
  - $\alpha = 0$ → কখনো weight update হবে না।
  - $\alpha > 1$ → step অনেক বড়, algorithm unstable হতে পারে।
  - সাধারণত $\alpha \approx 0.5$; আরও ভালো $\alpha = 1/n$ ($n$ = training pattern সংখ্যা, যেমন $1/6 \approx 0.17$)।
- **Initial weights:** arbitrary (tutorial-এ সব 1: $[1\,1\,1\,1\,1\,1]$)। Better: small random values in $[-1, 1]$।
  - **সব 0 দিও না** — training আটকে যেতে পারে।

### Step 4: Weight Update Rule

$$\boxed{w(i+1) = \begin{cases} w(i) & \text{if } w^T(i)\,y > 0 \quad \text{(correct → no change)}\\ w(i) + \alpha\, y & \text{if } w^T(i)\,y \le 0 \quad \text{(misclassified → update)} \end{cases}}$$

মানে: ভুল classify হলে, ঐ misclassified $y$-এর একটা অংশ ($\alpha$ গুণ) weight-এর সাথে **যোগ** করো।

> Tutorial-এ "$< 0$" লেখা; $g = 0$ হলেও point boundary-র উপর, অর্থাৎ ঠিকমতো classified না — তাই নিরাপদ নিয়ম: **$g \le 0$ হলে update**।

### দুইটা Approach

| | **One at a Time** (Single-sample) | **Many at a Time** (Batch) |
|---|---|---|
| কখন update | প্রতিটা misclassified $y$ পাওয়ার **সাথে সাথেই** | পুরো iteration (epoch) শেষে **একবার** |
| Formula | $w = w + \alpha\, y$ | $w = w + \alpha \sum_{\text{misclassified}} y$ |
| Iteration-এর ভিতর $w$ | বদলাতে পারে | **একটাই $w$** পুরো iteration-এ |
| Stability | প্রতিটা point এ weight লাফায় → **noisy / zig-zag** | সব error একসাথে average-এর মতো কাজ করে → **more stable, better convergence** ✅ |

> **Lecture-07 (teacher):** "**Many at a time → more stable, better convergence**" — short question এ এটা লিখবে। কারণ: batch update পুরো dataset এর misclassified points এর sum নেয়, তাই একটা single noisy point weight কে ভুল দিকে টেনে নিতে পারে না।

### Step 5: Convergence (কখন থামবো)

একটা **পুরো iteration (epoch)**-এ যদি **কোনো $y$ misclassified না হয়** (সব $a^Ty > 0$), অর্থাৎ কোনো weight update হয়নি → **Converged, STOP।**
(অথবা নির্দিষ্ট max iteration শেষ হলে থামো।)

Iteration সংখ্যা depend করে: problem কতটা কঠিন, $\alpha$, initial weight, one-at-a-time নাকি many-at-a-time ইত্যাদির উপর।

---

## 4. Full Worked Example (Tutorial-এর example)

### Given

- $\omega_1$: $(1, 1),\; (1, -3),\; (4, 5)$
- $\omega_2$: $(2, 2.5),\; (0, 2),\; (2, 3)$
- $\alpha = 0.8$, initial $w^T = [1 \; 1 \; 1 \; 1 \; 1 \; 1]$
- $\varphi$: $y = [x_1^2,\; x_2^2,\; x_1x_2,\; x_1,\; x_2,\; 1]$

### Step A: High-dim vectors + Normalization (Class 2 negated)

| Point | Class | $y$ (Class 2 → reflected $-y$) |
|---|---|---|
| (1, 1) | $\omega_1$ | $y_{11} = [1,\ 1,\ 1,\ 1,\ 1,\ 1]$ |
| (1, –3) | $\omega_1$ | $y_{12} = [1,\ 1^{*},\ -3,\ 1,\ -3,\ 1]$ |
| (4, 5) | $\omega_1$ | $y_{13} = [16,\ 25,\ 20,\ 4,\ 5,\ 1]$ |
| (2, 2.5) | $\omega_2$ | $-y_{21} = [-4,\ -6.25,\ -5,\ -2,\ -2.5,\ -1]$ |
| (0, 2) | $\omega_2$ | $-y_{22} = [0,\ -4,\ 0,\ 0,\ -2,\ -1]$ |
| (2, 3) | $\omega_2$ | $-y_{23} = [-4,\ -9,\ -6,\ -2,\ -3,\ -1]$ |

> ⚠️ **Tutorial typo note:** $(1,-3)$-এর জন্য সঠিকভাবে $x_2^2 = 9$, অর্থাৎ $y_{12} = [1, 9, -3, 1, -3, 1]$ হওয়ার কথা। কিন্তু tutorial-এর **সব calculation**-এ $y_{12} = [1, 1, -3, 1, -3, 1]$ ব্যবহার করা হয়েছে। নিচের tables tutorial-এর numbers হুবহু follow করে (যাতে তোমার answer key-র সাথে মেলে)। নিজে exam-এ নতুন প্রশ্ন solve করার সময় **সঠিক $\varphi$** হিসাব করবে। (সঠিক 9 দিয়ে হিসাব শেষে Section 4.3-এ দেওয়া আছে।)

---

### 4.1 One-at-a-Time Approach ($\alpha = 0.8$)

Rule: প্রতিটা step-এ $g = w^Ty$ হিসাব করো; $g \le 0$ হলে সাথে সাথে $w \leftarrow w + 0.8\,y$।

#### Iteration 1

**Step 1:** $y_{11} = [1,1,1,1,1,1]$, $w = [1,1,1,1,1,1]$
$g = 1+1+1+1+1+1 = 6 > 0$ ✅ no change.

**Step 2:** $y_{12} = [1,1,-3,1,-3,1]$
$g = 1 + 1 - 3 + 1 - 3 + 1 = -2 < 0$ ❌ update:
$w = [1,1,1,1,1,1] + 0.8[1,1,-3,1,-3,1] = [1,1,1,1,1,1] + [0.8,0.8,-2.4,0.8,-2.4,0.8]$
$\Rightarrow w = [1.8,\ 1.8,\ -1.4,\ 1.8,\ -1.4,\ 1.8]$

**Step 3:** $y_{13} = [16,25,20,4,5,1]$
$g = 1.8(16) + 1.8(25) - 1.4(20) + 1.8(4) - 1.4(5) + 1.8(1)$
$= 28.8 + 45 - 28 + 7.2 - 7 + 1.8 = 47.8 > 0$ ✅

**Step 4:** $-y_{21} = [-4,-6.25,-5,-2,-2.5,-1]$
$g = 1.8(-4) + 1.8(-6.25) - 1.4(-5) + 1.8(-2) - 1.4(-2.5) + 1.8(-1)$
$= -7.2 - 11.25 + 7 - 3.6 + 3.5 - 1.8 = -13.35 < 0$ ❌ update:
$w = [1.8,1.8,-1.4,1.8,-1.4,1.8] + 0.8[-4,-6.25,-5,-2,-2.5,-1]$
$= [1.8,1.8,-1.4,1.8,-1.4,1.8] + [-3.2,-5,-4,-1.6,-2,-0.8]$
$\Rightarrow w = [-1.4,\ -3.2,\ -5.4,\ 0.2,\ -3.4,\ 1.0]$

**Step 5:** $-y_{22} = [0,-4,0,0,-2,-1]$
$g = 0 + (-3.2)(-4) + 0 + 0 + (-3.4)(-2) + 1(-1) = 12.8 + 6.8 - 1 = 18.6 > 0$ ✅

**Step 6:** $-y_{23} = [-4,-9,-6,-2,-3,-1]$
$g = (-1.4)(-4) + (-3.2)(-9) + (-5.4)(-6) + 0.2(-2) + (-3.4)(-3) + 1(-1)$
$= 5.6 + 28.8 + 32.4 - 0.4 + 10.2 - 1 = 75.6 > 0$ ✅

**Iteration 1 Summary:**

| Step | $y$ | $w$ used | $w^Ty$ | Update? |
|---|---|---|---|---|
| 1 | [1, 1, 1, 1, 1, 1] | [1, 1, 1, 1, 1, 1] | **+6** | No |
| 2 | [1, 1, –3, 1, –3, 1] | [1, 1, 1, 1, 1, 1] | **–2** | Yes |
| 3 | [16, 25, 20, 4, 5, 1] | [1.8, 1.8, –1.4, 1.8, –1.4, 1.8] | **47.8** | No |
| 4 | [–4, –6.25, –5, –2, –2.5, –1] | [1.8, 1.8, –1.4, 1.8, –1.4, 1.8] | **–13.35** | Yes |
| 5 | [0, –4, 0, 0, –2, –1] | [–1.4, –3.2, –5.4, 0.2, –3.4, 1] | **18.6** | No |
| 6 | [–4, –9, –6, –2, –3, –1] | [–1.4, –3.2, –5.4, 0.2, –3.4, 1] | **75.6** | No |

$w$ after Iteration 1 $= [-1.4,\ -3.2,\ -5.4,\ 0.2,\ -3.4,\ 1.0]$
(Updates হয়েছে → not converged → Iteration 2।)

> Tip: যদি শেষ step (Step 6)-এর value negative হতো, তাহলে **আরেকটা update অবশ্যই দেখাতে হবে** — ঐ new $w$ পরের iteration-এ যাবে।

#### Iteration 2

**Step 1:** $y_{11}$: $g = -1.4 - 3.2 - 5.4 + 0.2 - 3.4 + 1 = -12.2$ ❌
$w = [-1.4,-3.2,-5.4,0.2,-3.4,1] + 0.8[1,1,1,1,1,1] = [-0.6,\ -2.4,\ -4.6,\ 1.0,\ -2.6,\ 1.8]$

**Step 2:** $y_{12}$: $g = -0.6(1) - 2.4(1) - 4.6(-3) + 1(1) - 2.6(-3) + 1.8(1)$
$= -0.6 - 2.4 + 13.8 + 1 + 7.8 + 1.8 = 21.4$ ✅

**Step 3:** $y_{13}$: $g = -0.6(16) - 2.4(25) - 4.6(20) + 1(4) - 2.6(5) + 1.8(1)$
$= -9.6 - 60 - 92 + 4 - 13 + 1.8 = -168.8$ ❌
$w = [-0.6,-2.4,-4.6,1,-2.6,1.8] + [12.8,20,16,3.2,4,0.8] = [12.2,\ 17.6,\ 11.4,\ 4.2,\ 1.4,\ 2.6]$

**Step 4:** $-y_{21}$: $g = 12.2(-4) + 17.6(-6.25) + 11.4(-5) + 4.2(-2) + 1.4(-2.5) + 2.6(-1)$
$= -48.8 - 110 - 57 - 8.4 - 3.5 - 2.6 = -230.3$ ❌
$w = [12.2,17.6,11.4,4.2,1.4,2.6] + [-3.2,-5,-4,-1.6,-2,-0.8] = [9.0,\ 12.6,\ 7.4,\ 2.6,\ -0.6,\ 1.8]$

**Step 5:** $-y_{22}$: $g = 0 + 12.6(-4) + 0 + 0 + (-0.6)(-2) + 1.8(-1) = -50.4 + 1.2 - 1.8 = -51$ ❌
$w = [9,12.6,7.4,2.6,-0.6,1.8] + [0,-3.2,0,0,-1.6,-0.8] = [9.0,\ 9.4,\ 7.4,\ 2.6,\ -2.2,\ 1.0]$

**Step 6:** $-y_{23}$: $g = 9(-4) + 9.4(-9) + 7.4(-6) + 2.6(-2) + (-2.2)(-3) + 1(-1)$
$= -36 - 84.6 - 44.4 - 5.2 + 6.6 - 1 = -164.6$ ❌
**Last value negative → one more update:**
$w = [9,9.4,7.4,2.6,-2.2,1] + 0.8[-4,-9,-6,-2,-3,-1] = [9,9.4,7.4,2.6,-2.2,1] + [-3.2,-7.2,-4.8,-1.6,-2.4,-0.8]$
$\Rightarrow w = [5.8,\ 2.2,\ 2.6,\ 1.0,\ -4.6,\ 0.2]$

**Iteration 2 Summary:**

| Step | $y$ | $w$ used | $w^Ty$ | Update? |
|---|---|---|---|---|
| 1 | [1, 1, 1, 1, 1, 1] | [–1.4, –3.2, –5.4, 0.2, –3.4, 1] | **–12.2** | Yes |
| 2 | [1, 1, –3, 1, –3, 1] | [–0.6, –2.4, –4.6, 1, –2.6, 1.8] | **21.4** | No |
| 3 | [16, 25, 20, 4, 5, 1] | [–0.6, –2.4, –4.6, 1, –2.6, 1.8] | **–168.8** | Yes |
| 4 | [–4, –6.25, –5, –2, –2.5, –1] | [12.2, 17.6, 11.4, 4.2, 1.4, 2.6] | **–230.3** | Yes |
| 5 | [0, –4, 0, 0, –2, –1] | [9, 12.6, 7.4, 2.6, –0.6, 1.8] | **–51** | Yes |
| 6 | [–4, –9, –6, –2, –3, –1] | [9, 9.4, 7.4, 2.6, –2.2, 1] | **–164.6** | Yes |

$w$ for Iteration 3 $= [5.8,\ 2.2,\ 2.6,\ 1.0,\ -4.6,\ 0.2]$

> (Bonus check — Iteration 3-এ এই $w$ দিয়ে: $g$ = 7.2, 15.2, 181, **–40.65** ❌ … তাই আরও iteration লাগবে। "Repeat until all $w^Ty$ positive or max iterations finished.")

---

### 4.2 Many-at-a-Time Approach ($\alpha = 0.8$)

Rule: পুরো iteration-এ **একই $w$**। সব $g$ বের করো → negative যেগুলো, সেগুলোর $y$ যোগ করো → iteration শেষে $w \leftarrow w + \alpha \sum y$।

#### Iteration 1 ($w = [1,1,1,1,1,1]$)

All-ones $w$ হলে $g$ = $y$-এর component-গুলোর যোগফল:

| Step | $y$ | $w^Ty$ | Misclassified? |
|---|---|---|---|
| 1 | [1, 1, 1, 1, 1, 1] | **6** | No |
| 2 | [1, 1, –3, 1, –3, 1] | **–2** | Yes |
| 3 | [16, 25, 20, 4, 5, 1] | **71** | No |
| 4 | [–4, –6.25, –5, –2, –2.5, –1] | **–20.75** | Yes |
| 5 | [0, –4, 0, 0, –2, –1] | **–7** | Yes |
| 6 | [–4, –9, –6, –2, –3, –1] | **–25** | Yes |

**Sum of misclassified:**
$\sum y = y_{12} + y_{21} + y_{22} + y_{23}$
$= [1,1,-3,1,-3,1] + [-4,-6.25,-5,-2,-2.5,-1] + [0,-4,0,0,-2,-1] + [-4,-9,-6,-2,-3,-1]$
$= [-7,\ -18.25,\ -14,\ -3,\ -10.5,\ -2]$

**Update:**
$w = [1,1,1,1,1,1] + 0.8[-7,-18.25,-14,-3,-10.5,-2] = [1,1,1,1,1,1] + [-5.6,-14.6,-11.2,-2.4,-8.4,-1.6]$
$\Rightarrow w = [-4.6,\ -13.6,\ -10.2,\ -1.4,\ -7.4,\ -0.6]$

#### Iteration 2 ($w = [-4.6, -13.6, -10.2, -1.4, -7.4, -0.6]$)

| Step | Calculation | $w^Ty$ | Misclassified? |
|---|---|---|---|
| 1 | –4.6 –13.6 –10.2 –1.4 –7.4 –0.6 | **–37.8** | Yes |
| 2 | –4.6 –13.6 +30.6 –1.4 +22.2 –0.6 | **32.6** | No |
| 3 | –73.6 –340 –204 –5.6 –37 –0.6 | **–660.8** | Yes |
| 4 | 18.4 +85 +51 +2.8 +18.5 +0.6 | **176.3** | No |
| 5 | 0 +54.4 +0 +0 +14.8 +0.6 | **69.8** | No |
| 6 | 18.4 +122.4 +61.2 +2.8 +22.2 +0.6 | **227.6** | No |

**Sum of misclassified:** $y_{11} + y_{13} = [1,1,1,1,1,1] + [16,25,20,4,5,1] = [17,\ 26,\ 21,\ 5,\ 6,\ 2]$

**Update:**
$w = [-4.6,-13.6,-10.2,-1.4,-7.4,-0.6] + 0.8[17,26,21,5,6,2]$
$= [-4.6,-13.6,-10.2,-1.4,-7.4,-0.6] + [13.6,20.8,16.8,4,4.8,1.6]$
$\Rightarrow w = [9.0,\ 7.2,\ 6.6,\ 2.6,\ -2.6,\ 1.0]$ (Iteration 3-এর জন্য)

Repeat until একটা $w$-এর জন্য সব $w^Ty > 0$। (Python দিয়ে চালিয়ে দেখা গেছে: tutorial-এর data-তে many-at-a-time ~**101 iterations**-এ converge করে — তাই exam-এ সাধারণত শুধু 1–2 iteration দেখাতে বলে।)

---

### 4.3 Correct $\varphi$ ($x_2^2 = 9$) দিয়ে হলে কী হতো? (Reference only)

$y_{12} = [1, 9, -3, 1, -3, 1]$ নিলে One-at-a-Time, Iteration 1:

| Step | $w$ used | $w^Ty$ |
|---|---|---|
| 1 | [1,1,1,1,1,1] | 6 |
| 2 | [1,1,1,1,1,1] | **6** (positive! no update) |
| 3 | [1,1,1,1,1,1] | 71 |
| 4 | [1,1,1,1,1,1] | –20.75 → $w = [-2.2,-4,-3,-0.6,-1,0.2]$ |
| 5 | [–2.2,–4,–3,–0.6,–1,0.2] | 17.8 |
| 6 | [–2.2,–4,–3,–0.6,–1,0.2] | 66.8 |

দেখো — একটা component ভুল হলে পুরো chain বদলে যায়। তাই **$y$ vectors বানানোর সময় সাবধান**, এখানেই সবচেয়ে বেশি mark কাটে।

---

## 5. Decision Boundary — Equation ও কিভাবে আঁকবে

### 5.1 2D case (no $\varphi$)

Final weight $[w_1, w_2, w_0]$ পেলে boundary:

$$w_1x_1 + w_2x_2 + w_0 = 0$$

**আঁকার নিয়ম (intercept method):**
1. $x_1 = 0$ বসাও → $x_2 = -w_0/w_2$ (vertical axis-এ point)
2. $x_2 = 0$ বসাও → $x_1 = -w_0/w_1$ (horizontal axis-এ point)
3. দুই point join করো → straight line।
4. যেকোনো একটা point বসিয়ে দেখো $g > 0$ কোন পাশে → ঐ পাশ $\omega_1$।

**Example (tutorial):** $g(x) = 2.8x_1 + 1x_2 - 8.9 = 0$
- $x_1 = 0 \Rightarrow x_2 = 8.9$
- $x_2 = 0 \Rightarrow x_1 = 8.9/2.8 \approx 3.18$
- Slope form: $x_2 = -2.8x_1 + 8.9$ (slope = $-w_1/w_2$, intercept = $-w_0/w_2$)

```
 x2
  |\
8.9*                    g(x) > 0 side  (ω1)  — line-এর উপরে/ডানে
  |  \
  |    \
  |  –   \     x
  |    –   \       x
  | –        \  x
  |   g(x)<0   \
  |   (ω2)       \
  +---------------*--------- x1
                 3.18
```

### 5.2 6D ($\varphi$) case — interesting point

Tutorial-এর $\varphi$ ব্যবহার করলে $g(y) = 0$ হয়:

$$w_5x_1^2 + w_4x_2^2 + w_3x_1x_2 + w_2x_1 + w_1x_2 + w_0 = 0$$

এটা 6D-তে একটা **hyperplane (linear)**, কিন্তু original 2D-তে এটা একটা **curve (conic: ellipse/parabola/hyperbola)**। যেমন Iteration 2-এর পরের $w$ দিয়ে: $5.8x_1^2 + 2.2x_2^2 + 2.6x_1x_2 + 1.0x_1 - 4.6x_2 + 0.2 = 0$।
👉 এভাবেই high-dimensional projection non-linear data separate করতে সাহায্য করে।

---

## 6. Limitations (Teacher: "limitation – linear")

### 6.1 শুধু Linearly Separable data-তে কাজ করে

- Perceptron শুধু একটা **straight line / hyperplane** শিখতে পারে।
- Data **linearly separable** হলে → **Perceptron Convergence Theorem** অনুযায়ী finite step-এ converge করবেই।
- Linearly separable **না** হলে → কখনো converge করবে না, weights চিরকাল ঘুরতে (oscillate) থাকবে। তাই max iteration limit দিতে হয়।

### 6.2 Classic example: XOR

| $x_1$ | $x_2$ | XOR | Class |
|---|---|---|---|
| 0 | 0 | 0 | – |
| 0 | 1 | 1 | x |
| 1 | 0 | 1 | x |
| 1 | 1 | 0 | – |

```
 x2
  1 |  x           –
    |
    |
  0 |  –           x
    +--------------------- x1
       0           1

  কোনো একটা straight line দিয়ে x গুলোকে
  এক পাশে আর – গুলোকে আরেক পাশে রাখা সম্ভব না!
  (diagonal-এর x দুইটা একসাথে, – দুইটা একসাথে)
```

**কেন অসম্ভব (short proof):** ধরো $g = w_1x_1 + w_2x_2 + w_0$।
- (0,0) → –: $w_0 < 0$
- (1,1) → –: $w_1 + w_2 + w_0 < 0$
- (0,1) → x: $w_2 + w_0 > 0$
- (1,0) → x: $w_1 + w_0 > 0$

শেষ দুইটা যোগ করলে: $w_1 + w_2 + 2w_0 > 0 \Rightarrow w_1 + w_2 + w_0 > -w_0 > 0$ (কারণ $w_0 < 0$)। কিন্তু দ্বিতীয় condition বলে $w_1 + w_2 + w_0 < 0$ — **contradiction!** ✅ তাই single perceptron XOR solve করতে পারে না।

(তুলনায় AND, OR linearly separable → perceptron পারে।)

### 6.3 Solution / Workaround

1. **High-dimensional projection ($\varphi$)** — যেমন $x_1x_2$ feature যোগ করলে XOR separable হয়ে যায় (tutorial-এর Step 1-এর মূল কারণ এটাই)।
2. **Multi-Layer Perceptron (MLP)** — hidden layer দিয়ে non-linear boundary।

### 6.4 অন্যান্য limitation
- Output শুধু hard 0/1 (step activation) — probability দেয় না।
- একাধিক valid solution থাকলে যেকোনো একটা দেয় (best margin না, SVM-এর মতো না)।
- Convergence speed $\alpha$, initial weight, data order-এর উপর depend করে।
- শুধু 2-class (binary) — multi-class-এর জন্য একাধিক perceptron লাগে।

---

## 6A. Short Questions (Lecture-07: "Mainly Math + Short Que")

**Q1. Perceptron কী? / What is a perceptron?**
Perceptron একটা linear binary classifier — weight vector $w$ শিখে $g(x) = w^Tx + w_0$ বানায়; $g(x) > 0$ হলে class 1, $< 0$ হলে class 2। এটা সবচেয়ে simple neural network (single neuron)।

**Q2. Perceptron এর main limitation কী? / Main limitation?**
এটা শুধু **linearly separable** data classify করতে পারে (একটা straight line / hyperplane দিয়ে আলাদা করা যায় এমন)। Non-linearly separable data তে (যেমন **XOR**) কখনো converge করে না।

**Q3. One-at-a-time vs Many-at-a-time — কোনটা ভালো? কেন?**
**Many-at-a-time** বেশি **stable** আর **better convergence** দেয়, কারণ পুরো iteration এর সব misclassified sample এর sum দিয়ে একবার update হয় — একটা single point weight কে এদিক-ওদিক টানতে পারে না। One-at-a-time প্রতিটা ভুলের পরপরই update করে, তাই weight zig-zag করে।

**Q4. Learning rate $\alpha$ এর ভূমিকা কী?**
$\alpha$ ঠিক করে প্রতিটা update এ weight কতটা বদলাবে। খুব ছোট → convergence ধীর; খুব বড় → overshoot / unstable। সাধারণত $0 < \alpha \le 1$।

**Q5. Class 2 এর vector গুলো negate (−1 দিয়ে গুণ) করা হয় কেন?**
তাহলে দুই class এর জন্য আলাদা condition লাগে না — সব $y$ এর জন্য একটাই নিয়ম: $w^Ty > 0$ হলে correct, নাহলে update। Algorithm simple হয়ে যায়।

**Q6. Augmented vector / bias কেন লাগে?**
$x_0 = 1$ যোগ করলে bias $w_0$ টা weight vector এর ভিতরে চলে আসে, ফলে $g(x) = w^Ty$ একটাই dot product। Bias ছাড়া boundary সবসময় origin দিয়ে যেতে বাধ্য হতো।

**Q7. $\varphi$ (high-dimensional projection) কেন করা হয়?**
Original space এ data linearly separable না হলে, $\varphi$ দিয়ে higher dimension এ নিলে ($x_1^2, x_1x_2, x_2^2$ ইত্যাদি) সেখানে linear boundary পাওয়া যেতে পারে — original space এ যেটা curved boundary।

**Q8. কখন algorithm থামে? / When does it stop (converge)?**
যখন একটা পুরো iteration এ কোনো sample misclassified হয় না (সব $w^Ty > 0$), অথবা max iteration শেষ হলে। Linearly separable হলে Perceptron Convergence Theorem অনুযায়ী finite step এ থামবেই।

---

## 7. Exam Tips / পরীক্ষার টিপস

1. **Step order মুখস্থ রাখো:** $\varphi$ projection → augment (শেষে 1) → Class 2 negate → init $w$, $\alpha$ → loop: $g = w^Ty$ → $g \le 0$ হলে $w = w + \alpha y$ → সব positive হলে stop।
2. **$y$ vectors বানানো সবচেয়ে গুরুত্বপূর্ণ।** $x_1^2, x_2^2, x_1x_2$ ঠিকমতো হিসাব করো; negative number-এর square positive (যেমন $(-3)^2 = 9$)।
3. **Class 2 negate করতে ভুলো না** — শেষের 1-ও –1 হয়ে যায়।
4. **প্রতিটা step-এ dot product লিখে দেখাও** (term by term) — partial marks পাবে।
5. **Summary table** দাও: $y$ | $w$ used | $w^Ty$ | update?। Teacher এটাই দেখতে চায়।
6. **One-at-a-time:** update সাথে সাথে, পরের step নতুন $w$ দিয়ে। **Many-at-a-time:** পুরো iteration একই $w$, শেষে sum দিয়ে একবার update।
7. **Last step negative হলে** iteration শেষেও update দেখাতে হবে (ঐ $w$ পরের iteration-এ যায়)।
8. **$g = 0$ → misclassified ধরো** (update করো)।
9. Decision boundary চাইলে: $w_1x_1 + w_2x_2 + w_0 = 0$ লিখে দুই intercept বের করে line আঁকো।
10. "Limitation" প্রশ্নে: **only linearly separable**, XOR diagram + "no single line can separate" — 1 mark নিশ্চিত।
11. Calculator-এ decimal সাবধানে (0.8 × –6.25 = –5)।

---

## 8. Practice Problem (Full Solution)

**Question:** Perceptron (one-at-a-time) দিয়ে নিচের দুই class-এর linear decision boundary বের করো। কোনো $\varphi$ projection লাগবে না, শুধু augmented vector $y = [x_1, x_2, 1]$ ব্যবহার করো।
- $\omega_1$: (2, 2), (3, 1)
- $\omega_2$: (0, 0), (1, 0)
- $\alpha = 0.5$, initial $w = [1, 1, 1]$। $g \le 0$ হলে update।

### Solution

**Augment + Normalize (Class 2 negated):**

| Point | Class | $y$ |
|---|---|---|
| (2, 2) | $\omega_1$ | $y_1 = [2, 2, 1]$ |
| (3, 1) | $\omega_1$ | $y_2 = [3, 1, 1]$ |
| (0, 0) | $\omega_2$ | $y_3 = [0, 0, -1]$ |
| (1, 0) | $\omega_2$ | $y_4 = [-1, 0, -1]$ |

**Iteration 1**

| Step | $y$ | $w$ used | $w^Ty$ | Action → new $w$ |
|---|---|---|---|---|
| 1 | [2, 2, 1] | [1, 1, 1] | 2+2+1 = **5** | ✅ no change |
| 2 | [3, 1, 1] | [1, 1, 1] | 3+1+1 = **5** | ✅ no change |
| 3 | [0, 0, –1] | [1, 1, 1] | 0+0–1 = **–1** | ❌ $[1,1,1]+0.5[0,0,-1] = [1, 1, 0.5]$ |
| 4 | [–1, 0, –1] | [1, 1, 0.5] | –1+0–0.5 = **–1.5** | ❌ $[1,1,0.5]+[-0.5,0,-0.5] = [0.5, 1, 0]$ |

**Iteration 2**

| Step | $y$ | $w$ used | $w^Ty$ | Action → new $w$ |
|---|---|---|---|---|
| 1 | [2, 2, 1] | [0.5, 1, 0] | 1+2+0 = **3** | ✅ |
| 2 | [3, 1, 1] | [0.5, 1, 0] | 1.5+1+0 = **2.5** | ✅ |
| 3 | [0, 0, –1] | [0.5, 1, 0] | 0+0+0 = **0** | ❌ (=0) → $[0.5, 1, -0.5]$ |
| 4 | [–1, 0, –1] | [0.5, 1, –0.5] | –0.5+0+0.5 = **0** | ❌ (=0) → $[0, 1, -1]$ |

**Iteration 3**

| Step | $y$ | $w$ used | $w^Ty$ | Action → new $w$ |
|---|---|---|---|---|
| 1 | [2, 2, 1] | [0, 1, –1] | 0+2–1 = **1** | ✅ |
| 2 | [3, 1, 1] | [0, 1, –1] | 0+1–1 = **0** | ❌ → $[0,1,-1]+[1.5,0.5,0.5] = [1.5, 1.5, -0.5]$ |
| 3 | [0, 0, –1] | [1.5, 1.5, –0.5] | 0+0+0.5 = **0.5** | ✅ |
| 4 | [–1, 0, –1] | [1.5, 1.5, –0.5] | –1.5+0+0.5 = **–1** | ❌ → $[1.5,1.5,-0.5]+[-0.5,0,-0.5] = [1, 1.5, -1]$ |

**Iteration 4**

| Step | $y$ | $w$ used | $w^Ty$ | Action → new $w$ |
|---|---|---|---|---|
| 1 | [2, 2, 1] | [1, 1.5, –1] | 2+3–1 = **4** | ✅ |
| 2 | [3, 1, 1] | [1, 1.5, –1] | 3+1.5–1 = **3.5** | ✅ |
| 3 | [0, 0, –1] | [1, 1.5, –1] | 0+0+1 = **1** | ✅ |
| 4 | [–1, 0, –1] | [1, 1.5, –1] | –1+0+1 = **0** | ❌ (=0) → $[1,1.5,-1]+[-0.5,0,-0.5] = [0.5, 1.5, -1.5]$ |

**Iteration 5**

| Step | $y$ | $w$ used | $w^Ty$ | Action |
|---|---|---|---|---|
| 1 | [2, 2, 1] | [0.5, 1.5, –1.5] | 1+3–1.5 = **2.5** | ✅ |
| 2 | [3, 1, 1] | [0.5, 1.5, –1.5] | 1.5+1.5–1.5 = **1.5** | ✅ |
| 3 | [0, 0, –1] | [0.5, 1.5, –1.5] | 0+0+1.5 = **1.5** | ✅ |
| 4 | [–1, 0, –1] | [0.5, 1.5, –1.5] | –0.5+0+1.5 = **1** | ✅ |

**পুরো Iteration 5-এ কোনো update নেই → CONVERGED ✅**

**Final weights:** $w_1 = 0.5,\; w_2 = 1.5,\; w_0 = -1.5$

**Decision boundary:**
$$0.5x_1 + 1.5x_2 - 1.5 = 0 \quad\Longleftrightarrow\quad x_1 + 3x_2 = 3$$

- $x_1 = 0 \Rightarrow x_2 = 1$ → point (0, 1)
- $x_2 = 0 \Rightarrow x_1 = 3$ → point (3, 0)

**Verification (original, un-negated points দিয়ে):**
- (2,2): $1 + 3 - 1.5 = 2.5 > 0$ → $\omega_1$ ✅
- (3,1): $1.5 + 1.5 - 1.5 = 1.5 > 0$ → $\omega_1$ ✅
- (0,0): $-1.5 < 0$ → $\omega_2$ ✅
- (1,0): $0.5 - 1.5 = -1 < 0$ → $\omega_2$ ✅

```
 x2
  2 |            x (2,2)
    |
  1 *  .                x (3,1)
    |      .    g > 0  (ω1)
    |  g<0     .
  0 –(0,0)  –(1,0)    .
    +--------------------*------ x1
    0      1      2      3
         line: x1 + 3x2 = 3
```

---

### Quick Revision (1-minute)

- $g(y) = w^Ty$, augmented ($x_0 = 1$), Class 2 → $-y$
- $g \le 0$ → $w \leftarrow w + \alpha y$ (one-at-a-time) / $w \leftarrow w + \alpha\sum y_{\text{mis}}$ (many-at-a-time)
- Stop: একটা full epoch-এ কোনো update নেই
- $0 < \alpha \le 1$, $w \ne 0$ initially
- Boundary: $w^Tx + w_0 = 0$ — a straight line/hyperplane
- **Limitation: only linearly separable (XOR fails)** → fix: $\varphi$ projection / MLP
