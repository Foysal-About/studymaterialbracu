# 01 — Introductory EDA, Data Types, Analytics & Chi-Square Test

> **Syllabus note (teacher):** "00 Introductory EDA — **Bivariate analysis IMPORTANT**, **Chi-square MATH**. EDA is descriptive."
> Data Mining overview এর সাথে মিলিয়ে প্রায় **~6 marks**। একটা **descriptive (theory) question** + একটা **Chi-square math** আসার chance বেশি।

---

## 0. Quick Roadmap (কী কী পড়বো)

| # | Topic | Exam weight |
|---|-------|-------------|
| 1 | Information System basics (context) | Low |
| 2 | Types of Analytics (Descriptive → Prescriptive) | Medium (descriptive Q) |
| 3 | What is EDA | Medium |
| 4 | Data Types (Nominal, Ordinal, Interval, Ratio; Discrete/Continuous) | Medium |
| 5 | Descriptive Statistics (mean, median, mode, SD, quartiles) | Medium |
| 6 | Box Plot, IQR & Outliers | Medium (small math) |
| 7 | Label Encoding vs One-Hot Encoding | Medium |
| 8 | Univariate vs **Bivariate Analysis** | **HIGH** |
| 9 | **Chi-Square Test (full math)** | **HIGH** |

---

## 1. Context: Information System (IS) — ছোট করে

**Information System (IS)** হলো formal, sociotechnical, organizational system যেটা information **collect, process, store, distribute** করে।

Sociotechnical view তে IS এর ৪টা component: **task, people, structure (roles), technology**।

IS তৈরি করতে যে **six components** লাগে:

| Component | সহজ বাংলায় |
|-----------|-------------|
| **Hardware** | Machine — CPU, input/output, storage, communication devices |
| **Software** | Programs + manuals; hardware কে বলে কী করতে হবে |
| **Data** | Facts — program এগুলো দিয়ে useful information বানায় |
| **Procedures** | Policies/rules — "Procedures are to people what software is to hardware" |
| **People** | Users, operators, maintainers — সবচেয়ে overlooked কিন্তু সবচেয়ে important |
| **Feedback** | Optional component — system কে feedback দেওয়া যায় |

> **Key line:** *Data is the bridge between hardware and people.* Data তখনই **information** হয় যখন মানুষ সেটা interpret করে।

**Classic pyramid (1980s):** Transaction Processing System (নিচে) → Management Information System → Decision Support System → Executive Information System (উপরে)।
Example (Super store): Transactions → Inventory management → নতুন item include/exclude → নতুন store open/close.

---

## 2. Types of Analytics (Four Types)

Data দিয়ে অসংখ্য প্রশ্নের উত্তর দেওয়া যায়, কিন্তু সব প্রশ্নকে **৪টা category** তে ভাগ করা যায় — প্রতিটার জন্য একটা analytics type। এগুলো **increasing order of complexity** তে সাজানো।

| Type of question | Type of analytics | Main techniques |
|------------------|-------------------|-----------------|
| **What happened?** | **Descriptive** | Summary statistics (count, mean, median, min, max, %, SD), visualization |
| **Why did it happen?** | **Diagnostic** | Hypothesis তৈরি, related data আনা, subset এ descriptive analysis ("data mining"), regression |
| **What will happen?** | **Predictive** | Statistical / Machine Learning models (classification, regression, forecasting) |
| **How can we make it happen? / What should we do?** | **Prescriptive** | Multiple prediction models + **optimization** → best action choose |

### Doctor analogy (মনে রাখার সহজ উপায়)
1. **Descriptive** — ডাক্তার রোগীর symptoms দেখে (কী হয়েছে)।
2. **Diagnostic** — রোগটা কী, কেন হলো সেটা diagnose করে।
3. **Predictive** — সামনে অবস্থা ভালো হবে নাকি খারাপ, predict করে।
4. **Prescriptive** — treatment / ওষুধ prescribe করে।

### Sales example (DataCamp source থেকে)
- **Descriptive:** গত মাসে কয়টা sale? Average transaction value কত? Store-wise sales এর standard deviation? Daily total sales এর line plot।
- **Diagnostic:** এক Thursday তে sales হঠাৎ কম (an **outlier**)। Hypothesis 1: website outage → data দেখে বাদ। Hypothesis 2: ঐ region এ storm ছিল → weather data দিয়ে time-series model → **bad weather is correlated with lower sales**।
  - ⚠️ Diagnostic analytics সাধারণত **relationship (correlation)** দেখায়, **causation prove করে না**।
- **Predictive:** Next quarter এর revenue forecast; কোন customer কিনবে তার probability predict।
- **Prescriptive:** Online/in-store mix বা price change এর বিভিন্ন scenario → predictive model দিয়ে প্রতিটা evaluate → **optimization** দিয়ে best scenario।

### Textbook definitions (02a source)
- **Descriptive analytics:** past data analyze করে summarized form এ present করা। E.g., post এর total likes, average monthly rainfall, website এর monthly average visitors। আজকের বেশিরভাগ analytics-ই descriptive।
- **Diagnostic analytics:** past data analyze করে **কেন** কোনো event ঘটলো তার কারণ বের করা (e.g., sensor data pattern থেকে fault এর কারণ)।
- **Predictive analytics:** existing data থেকে pattern শিখে event/outcome predict (**classification**) বা number forecast (**regression**)। Accuracy নির্ভর করে training data এর **quality আর volume** এর উপর।
- **Prescriptive analytics:** **multiple prediction models** দিয়ে বিভিন্ন input/action এর outcome predict করে **best course of action** prescribe করে।

> **Bonus (same source):** Big Data এর **5 V's** — **Volume, Velocity, Variety, Veracity, Value**।

---

## 3. What is EDA? (Exploratory Data Analysis)

**EDA** মানে data নিয়ে কোনো model বানানোর **আগে** data টাকে ভালোভাবে "explore" করা — summary statistics আর visualization দিয়ে data এর pattern, distribution, outlier, relationship বোঝা।

- **EDA is descriptive** — এটা মূলত **Descriptive Analytics** এর কাজ: "data তে কী আছে / কী হয়েছে?"
- EDA এর জন্য প্রথম prerequisite: **data type চেনা** — কারণ কোন statistic আর কোন chart use করবো সেটা data type এর উপর নির্ভর করে।
- EDA তে যা যা করি:
  1. Data types identify করা (numerical / categorical)
  2. **Univariate analysis** — একটা একটা variable এর summary (mean, median, histogram, box plot, bar chart)
  3. **Outlier detection** — box plot / IQR
  4. **Bivariate analysis** — দুইটা variable এর relationship (scatter plot, correlation, chi-square)
  5. Encoding (categorical → numeric) — ML model এর জন্য প্রস্তুত করা

> **English summary:** EDA = understanding data through descriptive statistics and visualizations before modeling. It is descriptive in nature, and depends on knowing data types.

---

## 4. Data Types (Measurement Scales)

```
                     Data
          ┌───────────┴────────────┐
    Categorical               Numerical
   (Qualitative)            (Quantitative)
    ┌────┴────┐              ┌─────┴──────┐
 Nominal   Ordinal       Discrete     Continuous
                                     ┌─────┴─────┐
                                 Interval     Ratio
```

### 4.1 Categorical (Qualitative) Data
Characteristics represent করে — gender, language, marital status, hometown। Number দিয়ে লিখলেও (1 = male, 2 = female) সেই number এর **কোনো mathematical meaning নেই** — এগুলো শুধু **label**। যোগ-বিয়োগ করা যায় না।

- **Nominal:** শুধু label, **order এর কোনো মানে নেই**। {0=Male, 1=Female} আর {0=Female, 1=Male} একই কথা।
  - শুধু ২টা category থাকলে তাকে **dichotomous** বলে (যেমন gender: male/female)।
- **Ordinal:** label + **order matters**। যেমন Education: Elementary < High School < College; star rating 1★ থেকে 5★; happiness/satisfaction scale।
  - **Limitation:** দুই value এর মাঝে **difference জানা নেই/সমান না** (Elementary→High School এর gap আর High School→College এর gap সমান না)।

### 4.2 Numerical (Quantitative) Data
Measurement (height, weight, BP) বা Count (shares, দাঁতের সংখ্যা)।

- **Discrete:** **Count করা যায়**, value গুলো list করা যায়।
  - Finite: 100 coin flip এ heads এর সংখ্যা → 0 থেকে 100।
  - Countably infinite: 100 heads পেতে কয়টা flip লাগবে → 100, 101, 102, …
- **Continuous:** **Measure করা যায়**, count করা যায় না; real number line এর interval এ যেকোনো value (8.4, 8.41, 8.414863 gallon …) — uncountably infinite।
  - **Interval:** ordered, equal differences, কিন্তু **true zero নেই**। Example: **temperature (°C)** — 0°C মানে "no temperature" না। যোগ-বিয়োগ চলে, **গুণ-ভাগ/ratio চলে না** (20°C, 10°C এর "দ্বিগুণ গরম" না)।
  - **Ratio:** interval এর সব property + **absolute (true) zero আছে**। Example: height, weight, length। Ratio meaningful (80 kg = 40 kg এর দ্বিগুণ)।

### 4.3 Which statistic / chart for which type? (Very important table)

| Data type | Allowed statistics | Visualization | Encoding (ML) |
|-----------|-------------------|---------------|---------------|
| **Nominal** | Frequency, proportion, percentage, mode | Pie chart, bar chart | **One-hot encoding** |
| **Ordinal** | Nominal এর সব + **median, mode, quartiles, percentiles, IQR** (কারণ order meaningful) | Pie, bar chart | **Label encoding** |
| **Continuous (Interval/Ratio)** | সব: mean, median, mode, SD, range, quartiles, percentiles, IQR | **Histogram, box plot** | Already numeric |

- **Histogram** দিয়ে central tendency, variability, modality, kurtosis দেখা যায়, কিন্তু **outlier ভালোভাবে দেখায় না** → তাই **box plot** use করি।
- **Why data types matter:** ভুল data type এ ভুল method use করলে **wrong analysis** হবে। যেমন nominal data এর mean নেওয়া meaningless।

> **Frequency** = কতবার ঘটেছে; **Proportion** = frequency / total; **Percentage** = proportion × 100 (sum = 100%)।

---

## 5. Descriptive Statistics (Build Statistics)

| Measure | Type | Formula / meaning |
|---------|------|-------------------|
| **Mean** | Central tendency | `mean = Σx / n` |
| **Median** | Central tendency | Sorted data এর মাঝের value (n even হলে মাঝের দুইটার average) |
| **Mode** | Central tendency | সবচেয়ে বেশি বার আসা value |
| **Range** | Spread | `max − min` |
| **Variance** | Spread | `σ² = Σ(x − mean)² / n` (population); sample: `/ (n − 1)` |
| **Standard Deviation (SD)** | Spread | `σ = √variance` |
| **Quartiles (Q1, Q2, Q3)** | Position | 25%, 50% (= median), 75% point |
| **IQR** | Spread | `IQR = Q3 − Q1` (middle 50% data এর spread) |

### Worked example: Mean, Median, Mode, SD
Data: `2, 4, 4, 4, 5, 5, 7, 9` (n = 8)

- **Mean** = (2+4+4+4+5+5+7+9) / 8 = 40 / 8 = **5**
- **Median** = 4th আর 5th value এর average = (4 + 5)/2 = **4.5**
- **Mode** = **4** (৩ বার এসেছে)
- **Range** = 9 − 2 = **7**
- Deviations (x − 5): −3, −1, −1, −1, 0, 0, 2, 4
- Squared: 9, 1, 1, 1, 0, 0, 4, 16 → Σ = **32**
- Population variance = 32 / 8 = **4** → **SD = √4 = 2**
- (Sample variance = 32 / 7 ≈ 4.571 → sample SD ≈ 2.138)

---

## 6. Box Plot (Box-and-Whisker), IQR & Outliers

**Box plot** হলো **five-number summary** এর ছবি: **Min, Q1, Median (Q2), Q3, Max** (outlier বাদে)।

```
   whisker      box       whisker            outlier
 |-------[  Q1 |  Q2  |  Q3  ]-------|            *
 min                               max (non-outlier)
```

### Outlier rule (1.5 × IQR rule)
```
IQR          = Q3 − Q1
Lower fence  = Q1 − 1.5 × IQR
Upper fence  = Q3 + 1.5 × IQR
Outlier      = যেকোনো value < Lower fence  অথবা  > Upper fence
```

### Quartile বের করার নিয়ম (Khan Academy method)
1. Data sort করো।
2. **Median (Q2)** বের করো।
3. **n odd** হলে median value টা বাদ দিয়ে lower half আর upper half নাও; **n even** হলে ঠিক মাঝখান থেকে দুই ভাগ।
4. Lower half এর median = **Q1**, upper half এর median = **Q3**।

### Worked example (odd n)
Data (commute distance, km): `12, 5, 40, 2, 9, 7, 10, 4, 8`

**Step 1 — Sort:** `2, 4, 5, 7, 8, 9, 10, 12, 40` (n = 9)

**Step 2 — Median:** 5th value = **8**

**Step 3 — Halves:** Lower = `2, 4, 5, 7`; Upper = `9, 10, 12, 40`

**Step 4 — Q1, Q3:**
- Q1 = (4 + 5)/2 = **4.5**
- Q3 = (10 + 12)/2 = **11**

**Step 5 — IQR:** 11 − 4.5 = **6.5**

**Step 6 — Fences:**
- 1.5 × IQR = 1.5 × 6.5 = 9.75
- Lower fence = 4.5 − 9.75 = **−5.25**
- Upper fence = 11 + 9.75 = **20.75**

**Step 7 — Outlier:** 40 > 20.75 → **40 is an outlier** ✅। নিচে কোনো outlier নেই।

**Box plot values:** Min whisker = 2, Q1 = 4.5, Median = 8, Q3 = 11, Max whisker = **12** (largest non-outlier), outlier point = 40।

> **Outlier এর effect:** Mean = 97/9 ≈ **10.78**, কিন্তু Median = **8**। 40 বাদ দিলে mean = 57/8 = 7.125। মানে **mean outlier দ্বারা affected হয়, median robust** — তাই skewed/outlier data তে median ভালো।

---

## 7. Label Encoding vs One-Hot Encoding

ML model শুধু number বোঝে, তাই categorical data কে numeric এ convert করতে হয়।

### 7.1 Label Encoding
প্রতিটা category কে একটা integer দেওয়া (0 থেকে N−1)। sklearn এর `LabelEncoder`।

| Company | Label |
|---------|-------|
| VW | 1 |
| Acura | 2 |
| Honda | 3 |

**Problem:** Model ধরে নেয় **বড় number = "বেশি/ভালো" category**, এবং number গুলোর মধ্যে arithmetic relation আছে। যেমন model average নিলে: (VW + Honda)/2 = (1 + 3)/2 = 2 = **Acura** → "VW আর Honda এর average হলো Acura" — যা absurd! → **Wrong model, lots of errors.**

👉 তাই Label encoding **Ordinal** data এর জন্য ঠিক আছে (যেখানে order আসলেই আছে), কিন্তু **Nominal** এর জন্য না।

### 7.2 One-Hot Encoding
প্রতিটা category এর জন্য আলাদা **binary (0/1) column** — একে "**binarization**" বলে (one-of-K scheme)।

| Company | is_VW | is_Acura | is_Honda |
|---------|-------|----------|----------|
| VW | 1 | 0 | 0 |
| Acura | 0 | 1 | 0 |
| Honda | 0 | 0 | 1 |

Example 2: `flower` ∈ {daffodil, lily, rose} → three binary features: `is_daffodil`, `is_lily`, `is_rose`।

> **English summary:** Label encoding imposes a fake order on nominal data; one-hot encoding avoids this by creating one binary column per category. Use **label encoding for ordinal**, **one-hot for nominal**.

---

## 8. Univariate vs Bivariate Analysis ⭐ (IMPORTANT)

| | **Univariate** | **Bivariate** |
|---|---|---|
| Variables | **একটা** variable | **দুইটা** variable একসাথে |
| Goal | Distribution / summary বোঝা | দুইটার মধ্যে **relationship / association** আছে কিনা |
| Example | Students দের height এর mean, histogram | Height আর weight এর মধ্যে relation আছে কিনা |
| Tools | Mean, median, SD, histogram, box plot, bar/pie chart | Scatter plot, correlation, grouped box plot, contingency table, **chi-square** |

**Bivariate analysis** মানে দুইটা variable একসাথে analyze করা — যেমন height আর weight এর মধ্যে relation আছে কিনা, বা gender এর সাথে ice-cream preference এর সম্পর্ক আছে কিনা।

> (Multivariate = দুইয়ের বেশি variable একসাথে।)

### 8.1 Three combinations of bivariate analysis (মুখস্থ রাখো)

| Variable pair | Example | Visualization | Statistical tool |
|---------------|---------|---------------|------------------|
| **Numerical – Numerical** | Height vs Weight; Study hours vs Marks | **Scatter plot** | **Correlation coefficient (r)** |
| **Categorical – Numerical** | Gender vs Salary; Store region vs Sales | **Side-by-side box plots**, bar chart of group means | Group-wise mean/median compare (formal test: t-test / ANOVA) |
| **Categorical – Categorical** | Gender vs Ice-cream flavor; Eye color vs Hair color | **Contingency table (cross-tab)**, stacked/grouped bar chart | **Chi-Square test of independence** |

### 8.2 Num–Num: Scatter plot & Correlation
- **Scatter plot:** x-axis এ একটা variable, y-axis এ আরেকটা; প্রতিটা observation একটা dot।
  - Dots উপরের দিকে উঠলে → **positive** relation; নিচে নামলে → **negative**; ছড়ানো → **no** relation।
- **Pearson correlation coefficient (r):**

```
          Σ (x − x̄)(y − ȳ)
r = ─────────────────────────────
     √[ Σ(x − x̄)² × Σ(y − ȳ)² ]
```
  - Range: **−1 ≤ r ≤ +1**
  - r ≈ +1 → strong positive, r ≈ −1 → strong negative, r ≈ 0 → no linear relation।
  - ⚠️ **Correlation ≠ Causation** (DataCamp: bad weather is *correlated* with low sales — cause prove হয়নি)।

**Worked example:** Study hours (x) vs Marks (y)

| x | y | x − x̄ | y − ȳ | (x−x̄)(y−ȳ) | (x−x̄)² | (y−ȳ)² |
|---|---|---|---|---|---|---|
| 2 | 3 | −4 | −6 | 24 | 16 | 36 |
| 4 | 7 | −2 | −2 | 4 | 4 | 4 |
| 6 | 8 | 0 | −1 | 0 | 0 | 1 |
| 8 | 12 | 2 | 3 | 6 | 4 | 9 |
| 10 | 15 | 4 | 6 | 24 | 16 | 36 |
| **Σ=30** | **Σ=45** | | | **58** | **40** | **86** |

- x̄ = 30/5 = 6, ȳ = 45/5 = 9
- r = 58 / √(40 × 86) = 58 / √3440 = 58 / 58.65 ≈ **0.989** → **very strong positive correlation**।

### 8.3 Cat–Num
একটা categorical variable দিয়ে data কে group করো, তারপর প্রতিটা group এর numerical variable এর summary (mean/median/box plot) compare করো।
Example: "Online vs In-store" অনুযায়ী sales এর total/mean compare (DataCamp diagnostic example এ এভাবেই in-store sales এর drop ধরা পড়েছিল)।

### 8.4 Cat–Cat → Chi-Square (next section) ⬇️

---

## 9. Chi-Square (χ²) Test ⭐⭐ (MATH — MUST PRACTICE)

### 9.1 What & Why
**Chi-square test** একটা statistical method যা দিয়ে দেখা হয়:
1. **Test of Independence:** দুইটা **categorical variable** এর মধ্যে significant **association** আছে কিনা (bivariate cat–cat)।
2. **Goodness-of-Fit:** observed data একটা expected distribution এর সাথে মেলে কিনা (e.g., একটা die/coin fair কিনা)।

Idea: **Observed frequency (O)** vs **Expected frequency (E)** compare করা। E হলো — *যদি দুই variable independent হতো, তাহলে কত count আশা করতাম*।

### 9.2 Formulas

```
Chi-square statistic:     χ² = Σ (O − E)² / E

Expected frequency:       E_ij = (Row_i Total × Column_j Total) / Grand Total

Degrees of freedom:       df = (r − 1) × (c − 1)       [test of independence]
                          df = k − 1                   [goodness-of-fit, k = categories]
```
- O = Observed frequency, E = Expected frequency, r = rows, c = columns.

### 9.3 Hypotheses
- **H₀ (Null):** Variables are **independent** — কোনো relation নেই (Observed ≈ Expected)।
- **H₁ (Alternative):** Variables are **dependent / associated** — relation আছে।

### 9.4 Decision Rule
- Significance level সাধারণত **α = 0.05**।
- **χ²_calculated > χ²_critical** (⇔ p < 0.05) → **Reject H₀** → association আছে।
- **χ²_calculated ≤ χ²_critical** (⇔ p > 0.05) → **Fail to reject H₀** → association এর পর্যাপ্ত evidence নেই।

> Intuition: (O − E) বড় → χ² বড় → observed data "independent হলে যা হতো" তার থেকে অনেক আলাদা → **H₀ rejected**। (O − E) ছোট → observed প্রায় expected এর সমান → **H₀ accepted (fail to reject)**।

### 9.5 Critical value table (α = 0.05) — মুখস্থ রাখো

| df | 1 | 2 | 3 | 4 | 5 | 6 |
|----|---|---|---|---|---|---|
| χ² critical | **3.841** | **5.991** | **7.815** | **9.488** | 11.070 | 12.592 |

### 9.6 The 7 Steps
1. **Define hypotheses** (H₀: independent, H₁: dependent)
2. **Organize data** into a **contingency table** (row/column totals সহ)
3. **Expected frequency** প্রতিটা cell এর জন্য: `E = (Row Total × Col Total) / Grand Total`
4. **χ²** = Σ (O − E)²/E
5. **df** = (r − 1)(c − 1)
6. **Critical value / p-value** compare (α = 0.05)
7. **Interpret** — reject or fail to reject H₀

---

### 9.7 Worked Example 1: Gender vs Ice-cream flavor (2 × 3)

**Step 1 — Hypotheses**
- H₀: Gender আর flavor preference independent।
- H₁: Gender আর flavor preference dependent।

**Step 2 — Contingency table (Observed, O)**

| | Chocolate | Vanilla | Strawberry | **Row Total** |
|---|---|---|---|---|
| **Male** | 20 | 15 | 10 | **45** |
| **Female** | 25 | 20 | 30 | **75** |
| **Col Total** | **45** | **35** | **40** | **120** |

**Step 3 — Expected frequencies**

| Cell | Calculation | E |
|------|-------------|---|
| Male, Chocolate | 45 × 45 / 120 | **16.875** |
| Male, Vanilla | 45 × 35 / 120 | **13.125** |
| Male, Strawberry | 45 × 40 / 120 | **15.000** |
| Female, Chocolate | 75 × 45 / 120 | **28.125** |
| Female, Vanilla | 75 × 35 / 120 | **21.875** |
| Female, Strawberry | 75 × 40 / 120 | **25.000** |

✔ Check: Expected এর row/column totals observed totals এর সমান হতে হবে (16.875 + 13.125 + 15 = 45 ✓)।

**Step 4 — χ² calculation**

| Cell | O | E | O − E | (O − E)² | (O − E)²/E |
|------|---|---|---|---|---|
| M, Choc | 20 | 16.875 | 3.125 | 9.7656 | 0.5787 |
| M, Van | 15 | 13.125 | 1.875 | 3.5156 | 0.2679 |
| M, Straw | 10 | 15.000 | −5.000 | 25.0000 | 1.6667 |
| F, Choc | 25 | 28.125 | −3.125 | 9.7656 | 0.3472 |
| F, Van | 20 | 21.875 | −1.875 | 3.5156 | 0.1607 |
| F, Straw | 30 | 25.000 | 5.000 | 25.0000 | 1.0000 |
| | | | | **χ² =** | **≈ 4.02** |

> ⚠️ **Note:** GeeksforGeeks source এ এই example এর answer **4.86** লেখা আছে, কিন্তু হিসাব করলে সঠিক মান **≈ 4.02** (উপরের table এ প্রতিটা term verify করা)। Final decision দুইটাতেই একই — কারণ দুইটাই 5.991 এর চেয়ে ছোট।

**Step 5 — df** = (2 − 1)(3 − 1) = 1 × 2 = **2**

**Step 6 — Critical value** (α = 0.05, df = 2) = **5.991**

**Step 7 — Decision:** 4.02 < 5.991 → p > 0.05 → **Fail to reject H₀**.
**Conclusion:** Gender আর ice-cream preference এর মধ্যে significant association এর evidence নেই — "men prefer vanilla, women prefer chocolate" claim টা support পায় না।

---

### 9.8 Worked Example 2: Eye color vs Hair color (3 × 3)

**Observed (O):**

| Eye \ Hair | Blonde | Brunette | Redhead | **Total** |
|---|---|---|---|---|
| **Blue** | 30 | 50 | 20 | **100** |
| **Brown** | 40 | 30 | 10 | **80** |
| **Green** | 20 | 10 | 10 | **40** |
| **Total** | **90** | **90** | **40** | **220** |

**Expected (E = Row × Col / 220):**

| | Blonde | Brunette | Redhead |
|---|---|---|---|
| **Blue** | 100×90/220 = 40.91 | 40.91 | 100×40/220 = 18.18 |
| **Brown** | 80×90/220 = 32.73 | 32.73 | 80×40/220 = 14.55 |
| **Green** | 40×90/220 = 16.36 | 16.36 | 40×40/220 = 7.27 |

**χ² terms:**

| Cell | O | E | (O − E)²/E |
|------|---|---|---|
| Blue-Blonde | 30 | 40.91 | 2.909 |
| Blue-Brunette | 50 | 40.91 | 2.020 |
| Blue-Red | 20 | 18.18 | 0.182 |
| Brown-Blonde | 40 | 32.73 | 1.616 |
| Brown-Brunette | 30 | 32.73 | 0.227 |
| Brown-Red | 10 | 14.55 | 1.420 |
| Green-Blonde | 20 | 16.36 | 0.808 |
| Green-Brunette | 10 | 16.36 | 2.475 |
| Green-Red | 10 | 7.27 | 1.023 |
| | | **χ² ≈** | **12.68** |

(Source এ rounding এর কারণে 12.67 লেখা — দুইটাই ঠিক ধরা হবে।)

- **df** = (3 − 1)(3 − 1) = **4**
- **Critical** (α = 0.05, df = 4) = **9.488**
- **Decision:** 12.68 > 9.488 → **Reject H₀** → Eye color আর hair color এর মধ্যে **significant association** আছে।

---

### 9.9 Worked Example 3: Goodness-of-Fit (Coin)
100 বার coin flip: **55 Heads, 45 Tails**। Coin কি fair?

- H₀: Coin fair (P(H) = P(T) = 0.5); H₁: Coin fair না।
- Expected: Heads = 50, Tails = 50
- χ² = (55 − 50)²/50 + (45 − 50)²/50 = 25/50 + 25/50 = 0.5 + 0.5 = **1.0**
- df = k − 1 = 2 − 1 = **1**
- Critical (α = 0.05, df = 1) = **3.841**
- 1.0 < 3.841 → **Fail to reject H₀** → coin সম্ভবত fair।

---

### 9.10 Assumptions & Considerations
- Observations **independent** হতে হবে (একজন মানুষ একবারই count হবে)।
- প্রতিটা cell এ **expected frequency ≥ 5** হওয়া ভালো; না হলে **Fisher's exact test** use করো।
- Data অবশ্যই **categorical (counts/frequencies)** — percentage বা mean দিয়ে chi-square করা যায় না।
- Chi-square **association** দেখায়, **causation না**।

### 9.11 Applications (Computer Science)
- **A/B testing:** Version A (120 clicks/1000) vs Version B (150 clicks/1000) — difference significant কিনা।
- **ML Feature Selection:** কোন categorical feature (e.g., Browser Type) target (Purchase Decision) এর সাথে related — p < 0.05 হলে feature useful।
- **Database query optimization:** partition গুলোতে row distribution uniform কিনা (sharding strategy check)।
- **NLP:** word frequency distribution expected থেকে deviate করে কিনা (e.g., spam এ overused terms)।

---

## 10. Exam Tips / পরীক্ষার টিপস 📝

1. **Chi-square math এ পুরো 7 step লিখবে** — H₀/H₁, table with totals, E table, (O−E)²/E table, df, critical value, conclusion। Step marks আছে!
2. **Expected formula** সবসময় লিখে দেখাও: `E = (Row Total × Column Total) / Grand Total`।
3. **Check:** প্রতিটা row এর E এর যোগফল = ঐ row এর observed total। ভুল ধরার সহজ উপায়।
4. Decimal অন্তত **2–4 digit** রাখো, শেষে round করো (না হলে final χ² একটু ভুল আসবে)।
5. **df ভুল কোরো না:** (rows − 1)(cols − 1) — total row/column count করবে না!
6. Conclusion এ শুধু "reject" লিখলে হবে না — **context এ** লিখো: "There is a significant association between X and Y."
7. **"Fail to reject H₀"** লেখা safer than "accept H₀"।
8. Critical values মুখস্থ: **df1 = 3.841, df2 = 5.991, df3 = 7.815, df4 = 9.488**। (প্রশ্নে table দিলে সেটাই use করো।)
9. **Descriptive question** এর জন্য: Bivariate analysis এর **3 combinations table** (num-num / cat-num / cat-cat) + example + tool — এটা লিখতে পারলে full marks।
10. "EDA is descriptive" — মানে EDA = **Descriptive analytics** (What happened?), model বানানো না।
11. Types of analytics লিখলে **question + technique + example** তিনটাই দাও; Doctor analogy দিলে বোঝাপড়া দেখায়।
12. Nominal vs Ordinal vs Interval vs Ratio — প্রতিটার **একটা example** আর **key difference** (order? equal gap? true zero?) লিখো।

### Quick Comparison: Data scales

| Scale | Order? | Equal intervals? | True zero? | Example |
|-------|--------|------------------|-----------|---------|
| Nominal | ✗ | ✗ | ✗ | Gender, color |
| Ordinal | ✓ | ✗ | ✗ | Star rating, education level |
| Interval | ✓ | ✓ | ✗ | Temperature (°C) |
| Ratio | ✓ | ✓ | ✓ | Height, weight |

---

## 11. Practice Questions (with Answers)

### Q1. (Descriptive) What is bivariate analysis? Explain its types with suitable tools.
**Answer:** Bivariate analysis দুইটা variable এর relationship/association analyze করে।
- Num–Num → scatter plot, correlation coefficient (e.g., height vs weight)।
- Cat–Num → grouped box plot, group mean comparison (e.g., gender vs salary)।
- Cat–Cat → contingency table, chi-square test (e.g., gender vs ice-cream flavor)।

### Q2. Distinguish the four types of analytics with one example each.
**Answer:** Descriptive (What happened? — last month এর total sales), Diagnostic (Why? — storm এর কারণে Thursday এ sales কম), Predictive (What will happen? — next quarter revenue forecast), Prescriptive (What should we do? — price/sales-mix scenario optimize করা)।

### Q3. Why is one-hot encoding preferred over label encoding for nominal data?
**Answer:** Label encoding fake order আর arithmetic relation তৈরি করে (VW=1, Acura=2, Honda=3 → average(VW, Honda) = Acura — meaningless)। One-hot প্রতিটা category এর জন্য আলাদা binary column বানায়, তাই কোনো false order নেই।

### Q4. Find outliers: `3, 7, 8, 5, 12, 14, 21, 13, 18` *(not from the slides; extra practice)*
**Answer:**
- Sorted: 3, 5, 7, 8, 12, 13, 14, 18, 21 (n = 9); Median = **12**
- Lower half: 3, 5, 7, 8 → Q1 = (5+7)/2 = **6**; Upper half: 13, 14, 18, 21 → Q3 = (14+18)/2 = **16**
- IQR = 16 − 6 = **10**; 1.5 × IQR = 15
- Fences: 6 − 15 = **−9**, 16 + 15 = **31**
- সব value −9 থেকে 31 এর মধ্যে → **No outlier**।

### Q5. (Chi-square math) Gender vs Voting preference — are they independent? (α = 0.05)

| | Male | Female | Total |
|---|---|---|---|
| Candidate A | 80 | 90 | 170 |
| Candidate B | 70 | 60 | 130 |
| Undecided | 50 | 40 | 90 |
| Total | 200 | 190 | 390 |

**Answer:**
- H₀: Gender আর voting preference independent; H₁: dependent।
- Expected:
  - A-Male = 170×200/390 = 87.18; A-Female = 170×190/390 = 82.82
  - B-Male = 130×200/390 = 66.67; B-Female = 130×190/390 = 63.33
  - U-Male = 90×200/390 = 46.15; U-Female = 90×190/390 = 43.85
- (O − E)²/E: 0.591 + 0.622 + 0.167 + 0.175 + 0.321 + 0.337 = **χ² ≈ 2.21**
- df = (3 − 1)(2 − 1) = **2**; critical = **5.991**
- 2.21 < 5.991 → **Fail to reject H₀** → gender আর voting preference এর মধ্যে significant association নেই।

### Q6. (Chi-square math, quick 2×2) Smoking vs Disease (α = 0.05) *(made-up practice data)*

| | Disease | No Disease | Total |
|---|---|---|---|
| Smoker | 30 | 10 | 40 |
| Non-smoker | 20 | 40 | 60 |
| Total | 50 | 50 | 100 |

**Answer:**
- E: Smoker-Disease = 40×50/100 = 20; Smoker-No = 20; Non-Disease = 60×50/100 = 30; Non-No = 30
- χ² = (30−20)²/20 + (10−20)²/20 + (20−30)²/30 + (40−30)²/30 = 5 + 5 + 3.333 + 3.333 = **16.67**
- df = (2−1)(2−1) = **1**; critical = **3.841**
- 16.67 > 3.841 → **Reject H₀** → Smoking আর disease এর মধ্যে significant association আছে।

### Q7. A die is rolled; the chi-square goodness-of-fit test is used. What is df?
**Answer:** k = 6 faces → df = 6 − 1 = **5** (critical at α = 0.05 = 11.070)।

### Q8. Temperature in °C is which data type, and why can't we say 20°C is twice as hot as 10°C?
**Answer:** **Interval** data — ordered, equal gaps, কিন্তু **true zero নেই** (0°C মানে temperature নেই এমন না), তাই ratio/গুণ-ভাগ meaningful না।

---

*Sources: Intro to Information Systems; Four Types of Analytics (textbook + DataCamp); Data Types, Statistics, Box Plot, Outliers, One-Hot Encoding notes; GeeksforGeeks Chi-Square Test. The correlation formula, the cat-num tools and the critical value table are standard additions that explain the bivariate part; they are not from the slides.*
