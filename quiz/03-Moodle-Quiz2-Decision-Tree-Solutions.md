# BracU (Fall-2026) Quiz #2 on Decision Tree — Solved

> নিচে Moodle Quiz #2 এর প্রতিটা প্রশ্ন **হুবহু (full question + সব option + marks)** দেওয়া, তারপর **Answer** আর **কিভাবে solve করবে (step by step)**।
> প্রতিটা math এ আগে **formula + প্রতিটা symbol এর মানে**, তারপর **প্রতিটা ধাপের হিসাব** দেওয়া — যাতে শুধু answer না, পুরো method টা বোঝা যায়।
> সব math answer course এর formula দিয়ে Python এ verify করা ✅
> Formula cheat sheet: [01-Moodle-Quiz Solutions](01-Moodle-Quiz-ZeroR-OneR-CM-DT-Solutions.md#formula-cheat-sheet) — Entropy, Gini ($p^2+q^2$), Chi-square ($\sqrt{(A-E)^2/E}$)।
> Theory: [05-ZeroR-OneR-Confusion-Matrix-Decision-Tree](../notes/05-ZeroR-OneR-Confusion-Matrix-Decision-Tree.md)

> ⚠️ Moodle এ option এর **অক্ষর (a/b/c/d) shuffle হয়** — অক্ষর না, **value** মুখস্থ রাখো।

---

## আগে Formula গুলো ভালো করে বুঝে নাও

এই quiz এর সব math এই ৫টা formula দিয়ে হয়।

### 1. Entropy — একটা set কতটা "mixed"

$$H(S) = -p\log_2 p \;-\; q\log_2 q$$

- **p** = set এ YES এর ratio = (YES সংখ্যা) / (মোট সংখ্যা)
- **q** = set এ NO এর ratio = (NO সংখ্যা) / (মোট সংখ্যা), আর p + q = 1
- **log₂** = base-2 log। Calculator এ: $\log_2 x = \dfrac{\ln x}{\ln 2}$ বা $\dfrac{\log_{10} x}{\log_{10} 2}$
- **মাইনাস চিহ্ন কেন?** 0 আর 1 এর মাঝের সংখ্যার log negative হয়, তাই মাইনাস দিয়ে positive বানানো হয়।
- **মান:** pure set (সব YES বা সব NO) → **0**; 50-50 set → **1** (2-class এ maximum)
- কোনো class এর সংখ্যা 0 হলে ঐ term = 0 ধরো (0 × log 0 = 0)

### 2. Weighted (Combined) Entropy — split এর পরের entropy

$$H_{split} = \frac{|S_1|}{|S|}H(S_1) + \frac{|S_2|}{|S|}H(S_2)$$

- **|S₁|, |S₂|** = child set দুটোর size; **|S|** = মোট size (= |S₁| + |S₂|)
- বড় set এর entropy বেশি weight পায়, কারণ বেশি instance ঐ set এ আছে।

### 3. Information Gain — split করে কতটা uncertainty কমলো

$$IG = H(\text{parent}) - H_{split}$$

- IG **বেশি** = ভালো split। IG = 0 মানে split কোনো কাজে আসেনি।

### 4. Gini (এই course এর version) — set কতটা "pure"

$$Gini(S) = p^2 + q^2 \qquad\qquad Gini_{split} = \frac{|S_1|}{|S|}Gini(S_1) + \frac{|S_2|}{|S|}Gini(S_2)$$

- এই course এ Gini = $p^2+q^2$ → **বড় মান = বেশি pure = ভালো split** (pure set → 1, 50-50 → 0.5)
- (অনেক বইয়ে Gini impurity = $1-p^2-q^2$, সেখানে **ছোট** মান ভালো। Exam এ course এর version use করো।)

### 5. Chi-square (এই course এর version) — split টা "random" থেকে কতটা আলাদা

প্রতিটা child set এর প্রতিটা class এর জন্য (প্রতিটা "cell"):

$$\text{Expected } E = (\text{child set এর size}) \times (\text{parent এ ঐ class এর ratio})$$

$$\chi_{cell} = \sqrt{\frac{(A - E)^2}{E}} \qquad\qquad \text{Total } \chi = \sum \chi_{cell}$$

- **A (Actual)** = child set এ ঐ class এর আসল সংখ্যা
- **E (Expected)** = split টা যদি useless হতো (child এর ratio parent এর মতোই থাকতো), তাহলে কত থাকার কথা ছিল
- **(A − E)²** = আসল আর expected এর পার্থক্য; square করা হয় যাতে + আর − পার্থক্য কাটাকাটি না হয়
- **÷ E** = পার্থক্যকে expected এর তুলনায় মাপা (4 এর জায়গায় 7 পাওয়া বড় ব্যাপার, 400 এর জায়গায় 403 না)
- **√** = এই course এর নিয়ম — প্রতিটা cell এর মানের square root নিয়ে তারপর যোগ। (Standard textbook chi-square এ √ নেই — এ কারণে option এর সাথে মিলাতে √ নিতেই হবে।)
- **মান:** বড় = split ভালো (parent থেকে অনেক আলাদা); 0 = child ratio parent এর সমান, split useless
- **"for the split"** = দুই set এর সব cell যোগ (৪টা); **"for the set S1"** = শুধু S1 এর ২টা cell

> **Calculator shortcut:** 2-class এ একটা set এর YES cell এ (A − E) যত, NO cell এ ঠিক তার উল্টো (−) — কারণ set এর size fixed। তাই (A − E)² দুইটা cell এ একই, শুধু ভাগ করার E আলাদা।

---

## Question 1 *(Marked out of 1.00)*

**Which of the following is an issue with Decision Trees?**

- a. Working with missing values
- b. Working with categorical attributes
- c. Avoiding underfitting
- d. Attributes with too few unique values

### ✅ Answer: a. Working with missing values

### কিভাবে solve করবে

Decision Tree learning এর classic **issues** (Mitchell, *Machine Learning*):
1. **Avoiding overfitting** — tree বেশি deep হয়ে training data মুখস্থ করে ফেলে → solution: pruning
2. **Continuous-valued attributes** handle করা
3. **Missing attribute values** handle করা ← option a
4. **Attributes with many values** (যেমন ID, Date — information gain এগুলোকে ভুল করে বেশি পছন্দ করে) → solution: Gain Ratio
5. Attributes with different costs

**বাকি option কেন ভুল:**
- b — categorical attribute DT খুব সহজেই handle করে (Play Golf এর সব attribute categorical) — এটা issue না।
- c — trap: issue হলো **over**fitting, underfitting না।
- d — trap: issue হলো **too many** unique values, too few না।

> (এই list টা slides এ নেই, standard textbook থেকে।)

---

## Question 2 *(Marked out of 1.25)*

**A Training Data Set has 25 YES instances and 5 NO instances. Now, We Split the Data Set using an attribute X into two new sets S1 and S2. The Set S1 has 15 YES and 3 NO instances, while the other set S2 has 10 YES and 2 NO instances. What is the combined Entropy after the split?**

- a. 0.75
- b. 0.60
- c. 0.65
- d. 0.70

### ✅ Answer: c. 0.65

### কিভাবে solve করবে

**Formula:** Combined entropy = Weighted entropy (Formula 2)
$$H_{split} = \frac{|S_1|}{|S|}H(S_1) + \frac{|S_2|}{|S|}H(S_2), \qquad H = -p\log_2 p - q\log_2 q$$

**Step 1 — Set গুলোর size:** |S1| = 15 + 3 = 18, |S2| = 10 + 2 = 12, |S| = 18 + 12 = 30

**Step 2 — H(S1)** (15 Yes, 3 No):
- p = 15/18 = 0.8333, q = 3/18 = 0.1667
- log₂(0.8333) = −0.2630 → −p·log₂p = 0.8333 × 0.2630 = **0.2192**
- log₂(0.1667) = −2.5850 → −q·log₂q = 0.1667 × 2.5850 = **0.4308**
- H(S1) = 0.2192 + 0.4308 = **0.650**

**Step 3 — H(S2)** (10 Yes, 2 No):
- p = 10/12 = 0.8333, q = 2/12 = 0.1667 → **S1 এর একই ratio**, তাই হিসাবও একই
- H(S2) = **0.650**

**Step 4 — Weighted (combined) entropy:**
$$\frac{18}{30}(0.650) + \frac{12}{30}(0.650) = 0.6 \times 0.650 + 0.4 \times 0.650 = 0.390 + 0.260 = \mathbf{0.65}$$

> Shortcut: দুইটা child এর ratio একই হলে combined entropy = ঐ ratio এর entropy। Parent ও 25:5 = 5:1 → H(parent) = 0.65 → Information Gain = 0.65 − 0.65 = 0।

---

## Question 3 *(Marked out of 1.00)*

**Which of the following is NOT a decision tree induction algorithm?**

- a. BOAT
- b. MOAT
- c. MARS
- d. CHAID

### ✅ Answer: b. MOAT

### কিভাবে solve করবে

Slide (Data Mining Overview) এর **"Tree Induction Algorithms"** list: **ID3, C4.5, CART, CHAID, MARS, BOAT**। MOAT এই list এ নেই — fake option।

| Algorithm | Full form / কাজ |
|---|---|
| ID3 | Iterative Dichotomiser 3 (Entropy / Information Gain) |
| C4.5 | ID3 এর successor (Gain Ratio) |
| CART | Classification And Regression Tree (Gini) |
| CHAID | Chi-square Automatic Interaction Detection |
| MARS | Numeric data এর জন্য ভালো |
| BOAT | Bootstrapped Optimistic Algorithm for Tree construction |

---

## Question 4 *(Marked out of 1.75)*

**A Training Data Set has 10 YES instances and 20 NO instances. By using a feature X, the data set is splitted into Two sets S1 and S2. The set S1 has 8 YES and 5 NO instances, while the set S2 has 2 YES and 15 NO instances. What is the Total CHI-SQUARE value for the split with X?**

- a. 5.64
- b. 3.78
- c. 6.45
- d. 3.97

### ✅ Answer: a. 5.64

### কিভাবে solve করবে

**Formula (Formula 5):**
$$E = (\text{set size}) \times (\text{parent ratio}), \qquad \chi_{cell} = \sqrt{\frac{(A-E)^2}{E}}, \qquad \text{Total} = \text{৪টা cell এর যোগফল}$$

**Step 1 — Parent ratio** (মোট 10 + 20 = 30):
- YES ratio = 10/30 = **1/3**, NO ratio = 20/30 = **2/3**
- মানে: X যদি useless হতো, তাহলে প্রতিটা child set এও 1/3 YES আর 2/3 NO থাকতো।

**Step 2 — Expected values** (E = set size × parent ratio):

| Set | Size | Actual YES | Expected YES | Actual NO | Expected NO |
|---|---|---|---|---|---|
| S1 | 8 + 5 = 13 | 8 | 13 × 1/3 = 4.333 | 5 | 13 × 2/3 = 8.667 |
| S2 | 2 + 15 = 17 | 2 | 17 × 1/3 = 5.667 | 15 | 17 × 2/3 = 11.333 |

> Check: Expected এর যোগ = set size (4.333 + 8.667 = 13 ✔, 5.667 + 11.333 = 17 ✔)

**Step 3 — প্রতিটা cell এর হিসাব** (A − E → square → ÷ E → √):

| Cell | A − E | (A − E)² | ÷ E | √ |
|---|---|---|---|---|
| S1 YES | 8 − 4.333 = 3.667 | 13.444 | 13.444 / 4.333 = 3.103 | **1.762** |
| S1 NO | 5 − 8.667 = −3.667 | 13.444 | 13.444 / 8.667 = 1.551 | **1.245** |
| S2 YES | 2 − 5.667 = −3.667 | 13.444 | 13.444 / 5.667 = 2.373 | **1.540** |
| S2 NO | 15 − 11.333 = 3.667 | 13.444 | 13.444 / 11.333 = 1.186 | **1.089** |

**Step 4 — Total:** 1.762 + 1.245 + 1.540 + 1.089 = 5.636 ≈ **5.64**

> ⚠️ √ না নিলে total = 3.103 + 1.551 + 2.373 + 1.186 = 8.21 — কোনো option এর সাথে মিলে না। তাই এই course এ **√ নিতেই হবে**।

---

## Question 5 *(Marked out of 1.00)*

**If the instances at a node of the decision tree are EQUALLY LIKELY to have labels YES and NO, then the entropy of the node is _______**

- a. 0.00
- b. 0.50
- c. Insufficient Data
- d. 1.00

### ✅ Answer: d. 1.00

### কিভাবে solve করবে

**Formula (Formula 1):** $H = -p\log_2 p - q\log_2 q$

**Step 1 — Ratio:** "Equally likely" মানে YES আর NO সমান → p = 0.5, q = 0.5

**Step 2 — Log:** log₂(0.5) = −1 (কারণ 2⁻¹ = 0.5)

**Step 3 — Entropy:**
$$H = -0.5 \times (-1) - 0.5 \times (-1) = 0.5 + 0.5 = \mathbf{1.00}$$

> মনে রাখো: Pure node → Entropy **0**; 50-50 node → Entropy **1** (2-class এ maximum)। Entropy শুধু **ratio** এর উপর নির্ভর করে, instance সংখ্যার উপর না — তাই "Insufficient Data" ভুল।

---

## Question 6 *(Marked out of 1.75)*

**A Training Data Set has 10 YES instances and 20 NO instances. By using a feature X, the data set is splitted into Two sets S1 and S2. The set S1 has ALL the YES instances, while the set S2 has ALL the NO instances. What is the Total CHI-SQUARE value for the split with X?**

- a. 11.49
- b. 13.25
- c. 10.64
- d. 12.36

### ✅ Answer: c. 10.64

### কিভাবে solve করবে

**Formula (Formula 5):** $E = \text{set size} \times \text{parent ratio}$, $\chi_{cell} = \sqrt{(A-E)^2/E}$, Total = ৪টা cell যোগ

**Step 1 — Parent ratio** (মোট 30): YES = 10/30 = **1/3**, NO = 20/30 = **2/3**

**Step 2 — Set গুলো বের করো:** "S1 has ALL the YES" → S1 = {10 Yes, 0 No}; "S2 has ALL the NO" → S2 = {0 Yes, 20 No}

**Step 3 — Expected values:**

| Set | Size | Actual YES | Expected YES | Actual NO | Expected NO |
|---|---|---|---|---|---|
| S1 | 10 | 10 | 10 × 1/3 = 3.333 | 0 | 10 × 2/3 = 6.667 |
| S2 | 20 | 0 | 20 × 1/3 = 6.667 | 20 | 20 × 2/3 = 13.333 |

**Step 4 — প্রতিটা cell এর হিসাব:**

| Cell | A − E | (A − E)² | ÷ E | √ |
|---|---|---|---|---|
| S1 YES | 10 − 3.333 = 6.667 | 44.444 | 44.444 / 3.333 = 13.333 | **3.651** |
| S1 NO | 0 − 6.667 = −6.667 | 44.444 | 44.444 / 6.667 = 6.667 | **2.582** |
| S2 YES | 0 − 6.667 = −6.667 | 44.444 | 44.444 / 6.667 = 6.667 | **2.582** |
| S2 NO | 20 − 13.333 = 6.667 | 44.444 | 44.444 / 13.333 = 3.333 | **1.826** |

**Step 5 — Total:** 3.651 + 2.582 + 2.582 + 1.826 = 10.641 ≈ **10.64**

> Perfect split (একটা set এ সব YES, অন্যটায় সব NO) → chi-square অনেক বড় → খুব ভালো split।

---

## Question 7 *(Marked out of 1.00)*

**A Training Data Set has two Labels YES and NO and two features X and Y. If we split the Data Set using the feature Y, we get two sets {5YES, 2NO} and {5YES, 3NO}. If we split using the attribute X, we get two sets {8YES, 1NO} and {2YES, 4NO}. Which of the two features will be picked by GINI INDEX to split the tree?**

- a. Y
- b. Insufficient data
- c. either of X or Y (Both equivalent)
- d. X

### ✅ Answer: d. X

### কিভাবে solve করবে

**Formula (Formula 4, course version):**
$$Gini(S) = p^2 + q^2, \qquad Gini_{split} = \frac{|S_1|}{|S|}Gini(S_1) + \frac{|S_2|}{|S|}Gini(S_2)$$
- p = YES ratio, q = NO ratio; **বড় Gini = বেশি pure = ভালো split** → যে feature এর weighted Gini বেশি, সেটা pick হবে।

**Step 1 — Split on Y** (মোট 7 + 8 = 15):
- Set {5Y, 2N}, size 7: p = 5/7 = 0.714, q = 2/7 = 0.286 → Gini = 0.714² + 0.286² = 0.510 + 0.082 = **0.592**
- Set {5Y, 3N}, size 8: p = 5/8 = 0.625, q = 3/8 = 0.375 → Gini = 0.625² + 0.375² = 0.391 + 0.141 = **0.531**
- Weighted = (7/15)(0.592) + (8/15)(0.531) = 0.276 + 0.283 = **0.560**

**Step 2 — Split on X** (মোট 9 + 6 = 15):
- Set {8Y, 1N}, size 9: p = 8/9 = 0.889, q = 1/9 = 0.111 → Gini = 0.889² + 0.111² = 0.790 + 0.012 = **0.802**
- Set {2Y, 4N}, size 6: p = 2/6 = 0.333, q = 4/6 = 0.667 → Gini = 0.333² + 0.667² = 0.111 + 0.444 = **0.556**
- Weighted = (9/15)(0.802) + (6/15)(0.556) = 0.481 + 0.222 = **0.704**

**Step 3 — Compare:** X = 0.704 > Y = 0.560 → **X** বেশি pure split দেয় → **X pick হবে**।

> চোখে দেখেই বোঝা যায়: X এর {8Y,1N} প্রায় pure, Y এর দুইটা set ই mixed।

---

## Question 8 *(Marked out of 1.00)*

**A Training Data Set has 20 YES instances and 25 NO instances. By using a feature X, the data set is splitted into Two sets S1 and S2. The set S1 has ALL the YES instances, while the set S2 has ALL the NO instances. What is the Total CHI-SQUARE value for the set S1?**

- a. 5.25
- b. 7.06
- c. 5.49
- d. 6.07

### ✅ Answer: b. 7.06

### কিভাবে solve করবে

**Formula (Formula 5):** $E = \text{set size} \times \text{parent ratio}$, $\chi_{cell} = \sqrt{(A-E)^2/E}$
- প্রশ্নে **"for the set S1"** → শুধু S1 এর ২টা cell (YES আর NO) যোগ করবে, S2 লাগবে না।

**Step 1 — Parent ratio** (মোট 20 + 25 = 45): YES = 20/45 = **0.4444**, NO = 25/45 = **0.5556**

**Step 2 — S1 বের করো:** "S1 has ALL the YES" → S1 = {20 Yes, 0 No}, size = 20

**Step 3 — Expected values (S1):**
- Expected YES = 20 × 0.4444 = **8.889**
- Expected NO = 20 × 0.5556 = **11.111**

**Step 4 — প্রতিটা cell এর হিসাব:**

| Cell | A − E | (A − E)² | ÷ E | √ |
|---|---|---|---|---|
| S1 YES | 20 − 8.889 = 11.111 | 123.457 | 123.457 / 8.889 = 13.889 | **3.727** |
| S1 NO | 0 − 11.111 = −11.111 | 123.457 | 123.457 / 11.111 = 11.111 | **3.333** |

**Step 5 — Total for S1:** 3.727 + 3.333 = 7.060 ≈ **7.06**

> (আগের quiz এর Question 11 এর মতো একই প্রশ্ন।)

---

## Question 9 *(Marked out of 1.75)*

**A Training Data Set has 20 YES instances and 25 NO instances. By using a feature X, the data set is splitted into Two sets S1 and S2. The set S1 has 8 YES and 10 NO instances, while the set S2 has 12 YES and 15 NO instances. What is the Total CHI-SQUARE value for the split with X?**

- a. 0.25
- b. 0.50
- c. 0.00
- d. 1.00

### ✅ Answer: c. 0.00

### কিভাবে solve করবে

**Formula (Formula 5):** $E = \text{set size} \times \text{parent ratio}$, $\chi_{cell} = \sqrt{(A-E)^2/E}$, Total = ৪টা cell যোগ

**Step 1 — Parent ratio** (মোট 45): YES = 20/45 = 4/9, NO = 25/45 = 5/9 → YES : NO = **4 : 5**

**Step 2 — Child ratio check:** S1 8 : 10 = **4 : 5**; S2 12 : 15 = **4 : 5** → parent এর সমান!

**Step 3 — Expected values:**

| Set | Size | Actual YES | Expected YES | Actual NO | Expected NO |
|---|---|---|---|---|---|
| S1 | 18 | 8 | 18 × 4/9 = **8** | 10 | 18 × 5/9 = **10** |
| S2 | 27 | 12 | 27 × 4/9 = **12** | 15 | 27 × 5/9 = **15** |

**Step 4 — প্রতিটা cell:** প্রতিটা cell এ A = E → A − E = 0 → $\sqrt{0^2/E} = 0$

**Step 5 — Total = 0 + 0 + 0 + 0 = 0.00** (split টা কোনো information দেয় না)

> Shortcut: child ratio = parent ratio হলে হিসাব না করেই answer **0**।

---

## Question 10 *(Marked out of 1.00)*

**A Training Data Set has 15 YES instances and 20 NO instances. By using a feature X, the data set is splitted into Two sets S1 and S2. The set S1 has 8 YES and 5 NO instances, while the set S2 has 7 YES and 15 NO instances. What is the Total CHI-SQUARE value for the set S1?**

- a. 2.97
- b. 1.92
- c. 2.00
- d. 2.29

### ✅ Answer: b. 1.92

### কিভাবে solve করবে

**Formula (Formula 5):** $E = \text{set size} \times \text{parent ratio}$, $\chi_{cell} = \sqrt{(A-E)^2/E}$
- **"for the set S1"** → শুধু S1 এর ২টা cell।

**Step 1 — Parent ratio** (মোট 15 + 20 = 35): YES = 15/35 = **0.4286**, NO = 20/35 = **0.5714**

**Step 2 — S1:** {8 Yes, 5 No}, size = 13

**Step 3 — Expected values (S1):**
- Expected YES = 13 × 0.4286 = **5.571**
- Expected NO = 13 × 0.5714 = **7.429**

**Step 4 — প্রতিটা cell এর হিসাব:**

| Cell | A − E | (A − E)² | ÷ E | √ |
|---|---|---|---|---|
| S1 YES | 8 − 5.571 = 2.429 | 5.898 | 5.898 / 5.571 = 1.059 | **1.029** |
| S1 NO | 5 − 7.429 = −2.429 | 5.898 | 5.898 / 7.429 = 0.794 | **0.891** |

**Step 5 — Total for S1:** 1.029 + 0.891 = 1.920 ≈ **1.92**

---

## Question 11 *(Marked out of 1.25)*

**A Training Data Set has some YES and No instances, with an initial entropy of 0.98. Now, We Split the Data Set using an attribute X into two new sets S1 and S2. The Set S1 has 4 YES and 2 NO instances, while the other set S2 has 3 YES and 3 NO instances. What is the Information Gain after the split?**

- a. 0.00 (No Gain)
- b. 0.07
- c. 0.02
- d. 0.05

### ✅ Answer: c. 0.02

### কিভাবে solve করবে

**Formula (Formula 1, 2, 3):**
$$IG = H(\text{parent}) - \left[\frac{|S_1|}{|S|}H(S_1) + \frac{|S_2|}{|S|}H(S_2)\right], \qquad H = -p\log_2 p - q\log_2 q$$
- H(parent) প্রশ্নেই দেওয়া = **0.98**

**Step 1 — Set গুলোর size:** |S1| = 4 + 2 = 6, |S2| = 3 + 3 = 6, |S| = 12

**Step 2 — H(S1)** (4 Yes, 2 No):
- p = 4/6 = 0.6667, q = 2/6 = 0.3333
- log₂(0.6667) = −0.5850 → −p·log₂p = 0.6667 × 0.5850 = **0.390**
- log₂(0.3333) = −1.5850 → −q·log₂q = 0.3333 × 1.5850 = **0.528**
- H(S1) = 0.390 + 0.528 = **0.918**

**Step 3 — H(S2)** (3 Yes, 3 No): p = q = 0.5 → 50-50 set → H(S2) = **1.000**

**Step 4 — Weighted entropy:**
$$\frac{6}{12}(0.918) + \frac{6}{12}(1.000) = 0.459 + 0.500 = 0.959$$

**Step 5 — Information Gain:** IG = 0.98 − 0.959 = 0.021 ≈ **0.02**

> Check: parent = 4+3 = 7 Yes, 2+3 = 5 No → p = 7/12, q = 5/12 → H = 0.4536 + 0.5263 = 0.980 ✔ (প্রশ্নের দেওয়া মানের সাথে মিলে)।

---

## Question 12 *(Marked out of 1.25)*

**A Training Data Set has two Labels YES and NO and two features X and Y. If we split the Data Set using the feature Y, we get two sets {8YES, 4NO} and {4YES, 2NO}. If we split using the attribute X, we get two sets {6YES, 3NO} and {6YES, 3NO}. Which of the two features will be picked by ENTROPY to split the tree?**

- a. Y
- b. Insufficient data
- c. either of X or Y (Both equivalent)
- d. X

### ✅ Answer: c. either of X or Y (Both equivalent)

### কিভাবে solve করবে

**Formula (Formula 1, 2):** $H = -p\log_2 p - q\log_2 q$; weighted $H_{split} = \sum \frac{|S_i|}{|S|}H(S_i)$
- Entropy দিয়ে pick করা মানে: যে feature এর weighted entropy **কম** (= IG বেশি), সেটা pick হবে।

**Step 1 — Ratio check:** সব set এর YES : NO = **2 : 1** ({8,4}, {4,2}, {6,3}, {6,3}) → p = 2/3, q = 1/3

**Step 2 — 2:1 set এর entropy:**
- log₂(2/3) = −0.5850 → 0.6667 × 0.5850 = 0.390
- log₂(1/3) = −1.5850 → 0.3333 × 1.5850 = 0.528
- H = 0.390 + 0.528 = **0.918** (৪টা set এরই একই)

**Step 3 — Weighted entropy** (দুই split এই মোট 18):
- Y: (12/18)(0.918) + (6/18)(0.918) = 0.612 + 0.306 = **0.918**
- X: (9/18)(0.918) + (9/18)(0.918) = 0.459 + 0.459 = **0.918**

**Step 4 — Compare:** Y = X = 0.918 → **Both equivalent** (আর parent ও 12:6 = 2:1, তাই দুইটারই IG = 0.918 − 0.918 = 0)।

> আগের quiz এর Question 9 (Chi-square দিয়ে) এর একই data — Entropy, Gini, Chi-square **যেটাই use করো**, child ratio = parent ratio হলে সব split সমান।

---

## Answer Summary

| Q | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ans | a | c | b | a | d | c | d | b | c | b | c | c |
| Value | Missing values | 0.65 | MOAT | 5.64 | 1.00 | 10.64 | X | 7.06 | 0.00 | 1.92 | 0.02 | Both |

## এই Quiz থেকে Pattern (Exam এর জন্য)

1. **Same-ratio trick** (Q2, Q9, Q12): child এর ratio parent এর সমান → Chi-square = 0, IG = 0, সব split equivalent। হিসাব করার আগে ratio চেক করো — সময় বাঁচবে।
2. **All-YES / All-NO split** (Q6, Q8): perfect split → chi-square অনেক বড়।
3. **Gini direction**: এই course এ $p^2+q^2$ → **বড় মান = ভালো split**। Entropy তে **ছোট মান = ভালো split**।
4. **"Total for the split"** মানে দুইটা set এর সব cell যোগ; **"for the set S1"** মানে শুধু S1 এর দুইটা cell।
5. **Chi-square এ √ ভুলবে না**: √ ছাড়া যোগ করলে option এর সাথে মিলবে না (যেমন Q4 এ 8.21 আসে, সঠিক 5.64)।
6. **Table বানিয়ে হিসাব করো**: Set | Size | A | E | A − E | (A − E)² | ÷E | √ — ভুল কম হয়, partial marks ও পাবে।
