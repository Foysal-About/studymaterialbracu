# BracU (Fall-2026) Quiz #2 on Decision Tree — Solved

> নিচে Moodle Quiz #2 এর প্রতিটা প্রশ্ন **হুবহু (full question + সব option + marks)** দেওয়া, তারপর **Answer** আর **কিভাবে solve করবে (step by step)**।
> সব math answer course এর formula দিয়ে Python এ verify করা ✅
> Formula cheat sheet: [01-Moodle-Quiz Solutions](01-Moodle-Quiz-ZeroR-OneR-CM-DT-Solutions.md#formula-cheat-sheet) — Entropy, Gini ($p^2+q^2$), Chi-square ($\sqrt{(A-E)^2/E}$)।
> Theory: [05-ZeroR-OneR-Confusion-Matrix-Decision-Tree](../notes/05-ZeroR-OneR-Confusion-Matrix-Decision-Tree.md)

> ⚠️ Moodle এ option এর **অক্ষর (a/b/c/d) shuffle হয়** — অক্ষর না, **value** মুখস্থ রাখো।

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

**Step 1 — H(S1), 18 instances (15 Yes, 3 No):** p = 15/18 = 5/6, q = 1/6
$H(S_1) = -\frac56\log_2\frac56 - \frac16\log_2\frac16 = 0.219 + 0.431 = 0.650$

**Step 2 — H(S2), 12 instances (10 Yes, 2 No):** p = 10/12 = 5/6 → **S1 এর একই ratio**
$H(S_2) = 0.650$

**Step 3 — Combined:**
$\frac{18}{30}(0.650) + \frac{12}{30}(0.650) = 0.390 + 0.260 = $ **0.65**

> Shortcut: দুইটা child এর ratio একই হলে combined entropy = ঐ ratio এর entropy। Parent ও 25:5 = 5:1 → Information Gain = 0।

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

**Step 1 — Parent ratio:** YES = 10/30 = 1/3, NO = 20/30 = 2/3

**Step 2 — S1 (13 instances):** Expected YES = 4.333, Expected NO = 8.667
- YES: $\sqrt{(8-4.333)^2/4.333} = \sqrt{3.103} = 1.762$
- NO: $\sqrt{(5-8.667)^2/8.667} = \sqrt{1.551} = 1.245$

**Step 3 — S2 (17 instances):** Expected YES = 5.667, Expected NO = 11.333
- YES: $\sqrt{(2-5.667)^2/5.667} = \sqrt{2.373} = 1.540$
- NO: $\sqrt{(15-11.333)^2/11.333} = \sqrt{1.186} = 1.089$

**Step 4 — Total:** 1.762 + 1.245 + 1.540 + 1.089 = **5.64**

---

## Question 5 *(Marked out of 1.00)*

**If the instances at a node of the decision tree are EQUALLY LIKELY to have labels YES and NO, then the entropy of the node is _______**

- a. 0.00
- b. 0.50
- c. Insufficient Data
- d. 1.00

### ✅ Answer: d. 1.00

### কিভাবে solve করবে

Equally likely → p = q = 0.5
$H = -0.5\log_2 0.5 - 0.5\log_2 0.5 = 0.5(1) + 0.5(1) = $ **1.00**

> মনে রাখো: Pure node → Entropy **0**; 50-50 node → Entropy **1** (2-class এ maximum)। Instance সংখ্যা জানা লাগে না, শুধু ratio লাগে — তাই "Insufficient Data" ভুল।

---

## Question 6 *(Marked out of 1.75)*

**A Training Data Set has 10 YES instances and 20 NO instances. By using a feature X, the data set is splitted into Two sets S1 and S2. The set S1 has ALL the YES instances, while the set S2 has ALL the NO instances. What is the Total CHI-SQUARE value for the split with X?**

- a. 11.49
- b. 13.25
- c. 10.64
- d. 12.36

### ✅ Answer: c. 10.64

### কিভাবে solve করবে

**Step 1 — Parent ratio:** YES = 1/3, NO = 2/3
S1 = {10 Yes, 0 No}, S2 = {0 Yes, 20 No}

**Step 2 — S1 (10 instances):** Expected YES = 3.333, Expected NO = 6.667
- YES: $\sqrt{(10-3.333)^2/3.333} = \sqrt{44.44/3.333} = \sqrt{13.333} = 3.651$
- NO: $\sqrt{(0-6.667)^2/6.667} = \sqrt{6.667} = 2.582$

**Step 3 — S2 (20 instances):** Expected YES = 6.667, Expected NO = 13.333
- YES: $\sqrt{(0-6.667)^2/6.667} = \sqrt{6.667} = 2.582$
- NO: $\sqrt{(20-13.333)^2/13.333} = \sqrt{44.44/13.333} = \sqrt{3.333} = 1.826$

**Step 4 — Total:** 3.651 + 2.582 + 2.582 + 1.826 = **10.64**

---

## Question 7 *(Marked out of 1.00)*

**A Training Data Set has two Labels YES and NO and two features X and Y. If we split the Data Set using the feature Y, we get two sets {5YES, 2NO} and {5YES, 3NO}. If we split using the attribute X, we get two sets {8YES, 1NO} and {2YES, 4NO}. Which of the two features will be picked by GINI INDEX to split the tree?**

- a. Y
- b. Insufficient data
- c. either of X or Y (Both equivalent)
- d. X

### ✅ Answer: d. X

### কিভাবে solve করবে

Course এর Gini = $p^2 + q^2$ → **বেশি মান = বেশি pure = ভালো split**। প্রতিটা split এর weighted Gini বের করো।

**Step 1 — Split on Y** (7 + 8 = 15):
- {5Y, 2N}: (5/7)² + (2/7)² = 0.510 + 0.082 = 0.592
- {5Y, 3N}: (5/8)² + (3/8)² = 0.391 + 0.141 = 0.531
- Weighted = (7/15)(0.592) + (8/15)(0.531) = 0.276 + 0.283 = **0.560**

**Step 2 — Split on X** (9 + 6 = 15):
- {8Y, 1N}: (8/9)² + (1/9)² = 0.790 + 0.012 = 0.802
- {2Y, 4N}: (2/6)² + (4/6)² = 0.111 + 0.444 = 0.556
- Weighted = (9/15)(0.802) + (6/15)(0.556) = 0.481 + 0.222 = **0.704**

**Step 3:** 0.704 > 0.560 → **X** বেশি pure split দেয়।

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

**Step 1 — Parent ratio:** YES = 20/45 = 0.444, NO = 25/45 = 0.556

**Step 2 — S1 = {20 Yes, 0 No}:** Expected YES = 20 × 0.444 = 8.889, Expected NO = 20 × 0.556 = 11.111

**Step 3:**
- YES: $\sqrt{(20-8.889)^2/8.889} = \sqrt{13.889} = 3.727$
- NO: $\sqrt{(0-11.111)^2/11.111} = \sqrt{11.111} = 3.333$

**Step 4 — Total for S1:** 3.727 + 3.333 = **7.06**

> "for the **set S1**" → শুধু S1 এর দুইটা value। (আগের quiz এর Question 11 এর মতো একই প্রশ্ন।)

---

## Question 9 *(Marked out of 1.75)*

**A Training Data Set has 20 YES instances and 25 NO instances. By using a feature X, the data set is splitted into Two sets S1 and S2. The set S1 has 8 YES and 10 NO instances, while the set S2 has 12 YES and 15 NO instances. What is the Total CHI-SQUARE value for the split with X?**

- a. 0.25
- b. 0.50
- c. 0.00
- d. 1.00

### ✅ Answer: c. 0.00

### কিভাবে solve করবে

**Step 1 — Ratio check:** Parent 20 : 25 = **4 : 5**; S1 8 : 10 = **4 : 5**; S2 12 : 15 = **4 : 5**

**Step 2 — Expected values:**
- S1 (18): Expected YES = 18 × 20/45 = **8**, NO = **10** → Actual এর সাথে হুবহু মিলে → χ = 0 + 0
- S2 (27): Expected YES = 27 × 20/45 = **12**, NO = **15** → χ = 0 + 0

**Step 3 — Total = 0.00** (split টা কোনো information দেয় না)

---

## Question 10 *(Marked out of 1.00)*

**A Training Data Set has 15 YES instances and 20 NO instances. By using a feature X, the data set is splitted into Two sets S1 and S2. The set S1 has 8 YES and 5 NO instances, while the set S2 has 7 YES and 15 NO instances. What is the Total CHI-SQUARE value for the set S1?**

- a. 2.97
- b. 1.92
- c. 2.00
- d. 2.29

### ✅ Answer: b. 1.92

### কিভাবে solve করবে

**Step 1 — Parent ratio:** YES = 15/35 = 0.4286, NO = 20/35 = 0.5714

**Step 2 — S1 (8 + 5 = 13 instances):** Expected YES = 13 × 0.4286 = 5.571, Expected NO = 13 × 0.5714 = 7.429

**Step 3:**
- YES: $\sqrt{(8-5.571)^2/5.571} = \sqrt{5.898/5.571} = \sqrt{1.059} = 1.029$
- NO: $\sqrt{(5-7.429)^2/7.429} = \sqrt{5.898/7.429} = \sqrt{0.794} = 0.891$

**Step 4 — Total for S1:** 1.029 + 0.891 = **1.92**

---

## Question 11 *(Marked out of 1.25)*

**A Training Data Set has some YES and No instances, with an initial entropy of 0.98. Now, We Split the Data Set using an attribute X into two new sets S1 and S2. The Set S1 has 4 YES and 2 NO instances, while the other set S2 has 3 YES and 3 NO instances. What is the Information Gain after the split?**

- a. 0.00 (No Gain)
- b. 0.07
- c. 0.02
- d. 0.05

### ✅ Answer: c. 0.02

### কিভাবে solve করবে

**Step 1 — H(S1), 6 instances (4 Yes, 2 No):**
$H(S_1) = -\frac46\log_2\frac46 - \frac26\log_2\frac26 = 0.390 + 0.528 = 0.918$

**Step 2 — H(S2), 6 instances (3 Yes, 3 No):** $H(S_2) = 1.000$

**Step 3 — Weighted entropy:** $\frac{6}{12}(0.918) + \frac{6}{12}(1.000) = 0.459 + 0.500 = 0.959$

**Step 4 — Information Gain:** IG = Parent entropy − Weighted entropy = 0.98 − 0.959 = **0.021 ≈ 0.02**

> Check: parent = 7Y, 5N → H = 0.980 ✔ (প্রশ্নের দেওয়া মানের সাথে মিলে)।

---

## Question 12 *(Marked out of 1.25)*

**A Training Data Set has two Labels YES and NO and two features X and Y. If we split the Data Set using the feature Y, we get two sets {8YES, 4NO} and {4YES, 2NO}. If we split using the attribute X, we get two sets {6YES, 3NO} and {6YES, 3NO}. Which of the two features will be picked by ENTROPY to split the tree?**

- a. Y
- b. Insufficient data
- c. either of X or Y (Both equivalent)
- d. X

### ✅ Answer: c. either of X or Y (Both equivalent)

### কিভাবে solve করবে

**Step 1:** সব set এর ratio YES : NO = **2 : 1** ({8,4}, {4,2}, {6,3}, {6,3})

**Step 2:** 2:1 এর entropy = $-\frac23\log_2\frac23 - \frac13\log_2\frac13 = 0.390 + 0.528 = 0.918$

**Step 3:**
- Y weighted = (12/18)(0.918) + (6/18)(0.918) = **0.918**
- X weighted = (9/18)(0.918) + (9/18)(0.918) = **0.918**

**Step 4:** সমান → **Both equivalent** (আর parent ও 12:6 = 2:1, তাই দুইটারই IG = 0)।

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
