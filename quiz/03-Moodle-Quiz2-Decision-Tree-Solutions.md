# Moodle Quiz #2 — Decision Tree (Fall-2026) — Solved

> BracU (Fall-2026) Quiz #2 on Decision Tree। সব math answer course এর formula দিয়ে Python এ verify করা ✅
> Formula cheat sheet: [01-Moodle-Quiz Solutions](01-Moodle-Quiz-ZeroR-OneR-CM-DT-Solutions.md) — Entropy, Gini ($p^2+q^2$), Chi-square ($\sqrt{(A-E)^2/E}$)।
> Theory: [05-ZeroR-OneR-Confusion-Matrix-Decision-Tree](../notes/05-ZeroR-OneR-Confusion-Matrix-Decision-Tree.md)

## Answer Summary

| Q | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ans | a | c | b | a | d | c | d | b | c | b | c | c |
| Value | Missing values | 0.65 | MOAT | 5.64 | 1.00 | 10.64 | X | 7.06 | 0.00 | 1.92 | 0.02 | Both |

> ⚠️ Moodle এ option এর **অক্ষর (a/b/c/d) shuffle হয়** — অক্ষর না, **value** মুখস্থ রাখো।

---

## Q1. Which of the following is an issue with Decision Trees?

> a. Working with missing values b. Working with categorical attributes c. Avoiding underfitting d. Attributes with too few unique values

**Answer: a. Working with missing values** ✅

Decision Tree এর classic **issues** (Mitchell, *Machine Learning*):
1. **Avoiding overfitting** (tree বেশি deep হয়ে training data মুখস্থ করে ফেলে) → solution: pruning
2. Continuous-valued attributes handle করা
3. **Missing attribute values** handle করা
4. **Attributes with many values** (যেমন ID, Date — information gain ভুল করে এগুলোকে বেশি পছন্দ করে) → solution: Gain Ratio
5. Attributes with different costs

**Trap options:** (c) "underfitting" না, আসল issue **overfitting**। (d) "too few" না, আসল issue **too many** unique values। (b) categorical attribute DT খুব সহজেই handle করে — এটা issue না।

> (এই list টা slides এ নেই, standard textbook থেকে নেওয়া।)

---

## Q2. Combined Entropy — Parent 25Y/5N; S1 = {15Y, 3N}, S2 = {10Y, 2N}

- S1: p = 15/18 = 5/6 → $H = -\frac56\log_2\frac56 - \frac16\log_2\frac16 = 0.219 + 0.431 = 0.650$
- S2: p = 10/12 = 5/6 → একই ratio → $H = 0.650$
- Combined = $\frac{18}{30}(0.650) + \frac{12}{30}(0.650) = $ **0.65**

**Answer: c. 0.65** ✅

> Shortcut: দুইটা child এর ratio একই হলে combined entropy = ঐ ratio এর entropy। (আর parent ও 25:5 = 5:1 → Information Gain = 0।)

---

## Q3. Which is NOT a decision tree induction algorithm?

> a. BOAT b. MOAT c. MARS d. CHAID

Slide (Data Mining Overview) এর **Tree Induction Algorithms** list: **ID3, C4.5, CART, CHAID, MARS, BOAT**।
MOAT এই list এ নেই (এটা fake/distractor option)।

**Answer: b. MOAT** ✅

| Algorithm | Full form / কাজ |
|---|---|
| ID3 | Iterative Dichotomiser 3 (Entropy / Information Gain) |
| C4.5 | ID3 এর successor (Gain Ratio) |
| CART | Classification And Regression Tree (Gini) |
| CHAID | Chi-square Automatic Interaction Detection |
| MARS | Numeric data এর জন্য ভালো |
| BOAT | Bootstrapped Optimistic Algorithm for Tree construction |

---

## Q4. Total Chi-square — Parent 10Y/20N; S1 = {8Y, 5N}, S2 = {2Y, 15N}

আগের quiz এর Q6 এর মতো একই প্রশ্ন ([solution দেখো](01-Moodle-Quiz-ZeroR-OneR-CM-DT-Solutions.md#q6-total-chi-square-for-split-x))।
1.762 + 1.245 + 1.540 + 1.089 = **5.64**

**Answer: a. 5.64** ✅

---

## Q5. YES আর NO equally likely হলে entropy কত?

p = q = 0.5 → $H = -0.5\log_2 0.5 - 0.5\log_2 0.5 = 0.5 + 0.5 = $ **1.00** (maximum impurity)

**Answer: d. 1.00** ✅

> মনে রাখো: Pure node → Entropy **0**; 50-50 node → Entropy **1** (binary case এ maximum)।

---

## Q6. Total Chi-square — Parent 10Y/20N; S1 = ALL YES {10Y, 0N}, S2 = ALL NO {0Y, 20N}

Parent ratio: YES = 1/3, NO = 2/3

**S1 (10 instances):** Expected Y = 3.333, N = 6.667
- Y: $\sqrt{(10-3.333)^2/3.333} = \sqrt{13.333} = 3.651$
- N: $\sqrt{(0-6.667)^2/6.667} = \sqrt{6.667} = 2.582$

**S2 (20 instances):** Expected Y = 6.667, N = 13.333
- Y: $\sqrt{(0-6.667)^2/6.667} = \sqrt{6.667} = 2.582$
- N: $\sqrt{(20-13.333)^2/13.333} = \sqrt{3.333} = 1.826$

Total = 3.651 + 2.582 + 2.582 + 1.826 = **10.64**

**Answer: c. 10.64** ✅

---

## Q7. GINI INDEX দিয়ে কোনটা pick হবে? Y → {5Y,2N}, {5Y,3N}; X → {8Y,1N}, {2Y,4N}

Course এর Gini = $p^2 + q^2$ → **বেশি মানে বেশি pure → বেশি Gini এর split টা নাও**।

**Split on Y** (total 15):
- {5Y,2N}: (5/7)² + (2/7)² = 0.510 + 0.082 = 0.592
- {5Y,3N}: (5/8)² + (3/8)² = 0.391 + 0.141 = 0.531
- Weighted = (7/15)(0.592) + (8/15)(0.531) = **0.560**

**Split on X** (total 15):
- {8Y,1N}: (8/9)² + (1/9)² = 0.790 + 0.012 = 0.802
- {2Y,4N}: (2/6)² + (4/6)² = 0.111 + 0.444 = 0.556
- Weighted = (9/15)(0.802) + (6/15)(0.556) = **0.704**

0.704 > 0.560 → **X**

**Answer: d. X** ✅

> চোখে দেখেই বোঝা যায়: X এর {8Y,1N} প্রায় pure, Y এর দুইটা set ই mixed।

---

## Q8. Chi-square for set S1 — Parent 20Y/25N; S1 = ALL YES

আগের quiz এর Q11 এর মতো একই ([solution](01-Moodle-Quiz-ZeroR-OneR-CM-DT-Solutions.md#q11-chi-square-for-set-s1-all-yes))। 3.727 + 3.333 = **7.06**

**Answer: b. 7.06** ✅

---

## Q9. Total Chi-square — Parent 20Y/25N; S1 = {8Y, 10N}, S2 = {12Y, 15N}

Parent ratio 20:25 = **4:5**। S1: 8:10 = **4:5**, S2: 12:15 = **4:5** → সব জায়গায় Actual = Expected

S1 expected: 18 × 20/45 = 8 Y, 10 N → একদম মিলে গেছে → 0
S2 expected: 27 × 20/45 = 12 Y, 15 N → 0

Total = **0.00**

**Answer: c. 0.00** ✅ (split টা কোনো information দেয় না)

---

## Q10. Chi-square for set S1 — Parent 15Y/20N; S1 = {8Y, 5N}

Parent ratio: YES = 15/35 = 0.4286, NO = 0.5714
S1 এ 13 instances → Expected Y = 13 × 0.4286 = 5.571, Expected N = 7.429

- Y: $\sqrt{(8-5.571)^2/5.571} = \sqrt{5.898/5.571} = \sqrt{1.059} = 1.029$
- N: $\sqrt{(5-7.429)^2/7.429} = \sqrt{5.898/7.429} = \sqrt{0.794} = 0.891$

Total (S1) = 1.029 + 0.891 = **1.92**

**Answer: b. 1.92** ✅

---

## Q11. Information Gain — initial entropy 0.98; S1 = {4Y, 2N}, S2 = {3Y, 3N}

- H(S1) = $-\frac46\log_2\frac46 - \frac26\log_2\frac26 = 0.390 + 0.528 = 0.918$
- H(S2) = 1.000 (3:3)
- Weighted = (6/12)(0.918) + (6/12)(1.000) = 0.959
- IG = 0.98 − 0.959 = **0.021 ≈ 0.02**

**Answer: c. 0.02** ✅

> Check: parent = 7Y, 5N → H = 0.980 ✔ (প্রশ্নের দেওয়া মানের সাথে মিলে)।

---

## Q12. ENTROPY দিয়ে কোনটা pick হবে? Y → {8Y,4N}, {4Y,2N}; X → {6Y,3N}, {6Y,3N}

সব set এর ratio **2:1** → প্রতিটা set এর entropy = $-\frac23\log_2\frac23 - \frac13\log_2\frac13 = 0.918$

- Y weighted = (12/18)(0.918) + (6/18)(0.918) = 0.918
- X weighted = 0.918

সমান → **Both equivalent** (আর দুইটার IG ই 0, কারণ parent ও 12:6 = 2:1)

**Answer: c. either of X or Y (Both equivalent)** ✅

> আগের quiz এর Q9 (Chi-square দিয়ে) এর একই data — Entropy, Gini, Chi-square **যেটাই use করো**, child ratio = parent ratio হলে সব split সমান।

---

## এই Quiz থেকে Pattern (Exam এর জন্য)

1. **Same-ratio trick** (Q2, Q9, Q12): child এর ratio parent এর সমান → Chi-square = 0, IG = 0, সব split equivalent। হিসাব করার আগে ratio চেক করো — সময় বাঁচবে।
2. **All-YES / All-NO split** (Q6, Q8): perfect split → chi-square অনেক বড়।
3. **Gini direction**: এই course এ $p^2+q^2$ → **বড় মান = ভালো split**। Entropy তে **ছোট মান = ভালো split**।
4. **"Total for the split"** মানে দুইটা set এর সব cell যোগ; **"for the set S1"** মানে শুধু S1 এর দুইটা cell।
