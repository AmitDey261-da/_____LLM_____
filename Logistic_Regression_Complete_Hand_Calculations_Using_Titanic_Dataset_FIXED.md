# Logistic Regression Complete Hand Calculations Using Titanic Dataset

## A Mathematics-Only Handbook for Predicting Passenger Survival

**Dataset:** `titanic.csv`  \
**Observations:** 891 passengers  \
**Outcome:** `survived`, where 1 = survived and 0 = did not survive  \
**Method:** Binary logistic regression  \
**Programming code in this handbook:** None  \
**Split rule:** 80% training and 20% testing, fixed shuffle state 42  \
**Classification threshold:** 0.50

> This handbook reproduces the logic of a fitted logistic-regression model using mathematical notation, arithmetic substitutions, and passenger-level results. A computer was used only to carry out the large volume of arithmetic accurately while preparing the handbook. No Python code is taught or shown.

---
## Table of Contents
1. [1. The prediction question](#1-the-prediction-question)
2. [2. Understanding the Titanic variables](#2-understanding-the-titanic-variables)
3. [3. Why ordinary linear regression is unsuitable](#3-why-ordinary-linear-regression-is-unsuitable)
4. [4. Probability, odds, and log-odds](#4-probability-odds-and-log-odds)
5. [5. The sigmoid function](#5-the-sigmoid-function)
6. [6. Model specification](#6-model-specification)
7. [7. Dataset audit and target arithmetic](#7-dataset-audit-and-target-arithmetic)
8. [8. Missing-value calculations](#8-missing-value-calculations)
9. [9. Encoding categorical variables](#9-encoding-categorical-variables)
10. [10. Standardizing numerical variables](#10-standardizing-numerical-variables)
11. [11. The design matrix](#11-the-design-matrix)
12. [12. Likelihood of one passenger](#12-likelihood-of-one-passenger)
13. [13. Likelihood of all training passengers](#13-likelihood-of-all-training-passengers)
14. [14. Log-likelihood](#14-log-likelihood)
15. [15. Gradient or score equations](#15-gradient-or-score-equations)
16. [16. Hessian and curvature](#16-hessian-and-curvature)
17. [17. Newton-Raphson coefficient estimation](#17-newton-raphson-coefficient-estimation)
18. [18. Final fitted equation](#18-final-fitted-equation)
19. [19. Complete prediction recipe](#19-complete-prediction-recipe)
20. [20. Detailed passenger calculations](#20-detailed-passenger-calculations)
21. [21. All 179 test predictions](#21-all-179-test-predictions)
22. [22. Confusion matrix by hand](#22-confusion-matrix-by-hand)
23. [23. Accuracy, precision, recall, and F1](#23-accuracy-precision-recall-and-f1)
24. [24. Log loss and ROC-AUC](#24-log-loss-and-roc-auc)
25. [25. Interpreting coefficients and odds ratios](#25-interpreting-coefficients-and-odds-ratios)
26. [26. Decision threshold experiments](#26-decision-threshold-experiments)
27. [27. Errors and residuals](#27-errors-and-residuals)
28. [28. Assumptions and limitations](#28-assumptions-and-limitations)
29. [29. Formula reference sheet](#29-formula-reference-sheet)
30. [30. Practice calculations with answers](#30-practice-calculations-with-answers)

---

## 1. The prediction question

For every passenger, the required prediction is binary:

$$y_i=\begin{cases}1,&	ext{passenger survived}\\0,&	ext{passenger did not survive}\end{cases}$$

The model does not first output 0 or 1. It first calculates a survival probability $\hat p_i$ between 0 and 1. The final prediction is:

$$\hat y_i=\begin{cases}1,&\hat p_i\ge 0.50\\0,&\hat p_i<0.50\end{cases}$$

The uploaded data contains 891 passenger records. The fixed split assigns 712 records to coefficient estimation and 179 records to independent testing.


---

## 2. Understanding the Titanic variables

The model uses seven source predictors:

| Variable | Meaning | Mathematical role |
|---|---|---|
| `pclass` | Ticket class 1, 2, or 3 | Converted to indicator variables |
| `sex` | male or female | Converted to one indicator |
| `age` | Age in years | Median-filled, then standardized |
| `sibsp` | Siblings/spouses aboard | Standardized |
| `parch` | Parents/children aboard | Standardized |
| `fare` | Passenger fare | Standardized |
| `embarked` | C, Q, or S | Converted to indicator variables |

Columns such as `alive` duplicate the target and are excluded. `adult_male`, `who`, and `class` are also excluded because they duplicate or derive from predictors already used. This prevents target leakage and unnecessary duplication.


---

## 3. Why ordinary linear regression is unsuitable

A straight-line model can produce impossible values such as $-0.18$ or $1.24$ for a binary event. Logistic regression instead maps every real-valued score to the interval $(0,1)$.

A linear score is still formed:

$$z_i=\beta_0+\beta_1x_{i1}+\cdots+\beta_kx_{ik}$$

But the score is transformed by the sigmoid:

$$\hat p_i=\frac{1}{1+e^{-z_i}}$$

Therefore, no matter how negative or positive $z_i$ becomes, the resulting probability remains valid.


---

## 4. Probability, odds, and log-odds

If $p$ is survival probability, the odds of survival are

$$	ext{Odds}=\frac{p}{1-p}$$

Examples:

| Probability $p$ | Arithmetic | Odds | Meaning |
|---:|---:|---:|---|
| 0.20 | $0.20/0.80$ | 0.25 | 1 survival for every 4 non-survivals |
| 0.50 | $0.50/0.50$ | 1.00 | equal odds |
| 0.80 | $0.80/0.20$ | 4.00 | 4 survivals for every 1 non-survival |

Taking the natural logarithm produces log-odds:

$$\log\left(\frac{p}{1-p}\right)=z$$

The inverse operation is exactly the sigmoid.


---

## 5. The sigmoid function

The sigmoid conversion is performed in four hand steps:

1. Calculate the linear score $z$.
2. Change its sign to obtain $-z$.
3. Calculate $e^{-z}$.
4. Calculate $1/(1+e^{-z})$.

Example for $z=1.20$:

$$e^{-1.20}=0.3010$$
$$1+0.3010=1.3010$$
$$p=1/1.3010=0.7686$$

Since $0.7686\ge0.50$, the classification is survived.


---

## 6. Model specification

After preparation, the fitted structure is:

$$z=\beta_0+\beta_A A^*+\beta_S S^*+\beta_P P^*+\beta_F F^*+\beta_{C2}I(C=2)+\beta_{C3}I(C=3)+\beta_M I(Male)+\beta_QI(Q)+\beta_SI(S)$$

Here $A^*,S^*,P^*,F^*$ are standardized age, siblings/spouses, parents/children, and fare. Class 1, female, and embarked C are reference categories. Their indicators are therefore zero.


---

## 7. Dataset audit and target arithmetic

Total passengers:

$$N=891$$

Survivors:

$$\sum y_i=342$$

Non-survivors:

$$N-\sum y_i=891-342=549$$

Overall observed survival rate:

$$\bar y=\frac{342}{891}=0.383838=38.38\%$$

The training set contains 268 survivors and 444 non-survivors. The test set contains 74 survivors and 105 non-survivors.


---

## 8. Missing-value calculations

Missing values are not allowed inside the numerical likelihood calculation. Training-set information only is used to fill them.

| Variable | Missing in full data | Training replacement | Rule |
|---|---:|---:|---|
| age | 177 | 28.0000 | training median |
| sibsp | 0 | 0.0000 | training median |
| parch | 0 | 0.0000 | training median |
| fare | 0 | 14.4542 | training median |
| pclass | 0 | 3 | training mode |
| sex | 0 | male | training mode |
| embarked | 2 | S | training mode |


For a sorted training variable with $n$ observations, the median is the central observation when $n$ is odd, or the average of the two central observations when $n$ is even:

$$	ext{Median}=\frac{x_{(n/2)}+x_{(n/2+1)}}{2}$$

A missing age is replaced before standardization. This is important because allowing test-set values to determine the median would leak information into training.


---

## 9. Encoding categorical variables

Dummy variables turn categories into arithmetic inputs.

### Class encoding

| Class | $I(C=2)$ | $I(C=3)$ |
|---:|---:|---:|
| 1 | 0 | 0 |
| 2 | 1 | 0 |
| 3 | 0 | 1 |

### Sex encoding

| Sex | $I(Male)$ |
|---|---:|
| female | 0 |
| male | 1 |

### Embarkation encoding

| Port | $I(Q)$ | $I(S)$ |
|---|---:|---:|
| C | 0 | 0 |
| Q | 1 | 0 |
| S | 0 | 1 |

One category from each group is omitted. Otherwise, the intercept and all indicators become perfectly dependent, creating the dummy-variable trap.


---

## 10. Standardizing numerical variables

For a numerical variable $x$, the training standardization is

$$x^*=\frac{x-\mu_x}{s_x}$$

where $\mu_x$ is the training mean and $s_x$ is the population-form training standard deviation used by the fitted transformation.

| Variable | Training mean $\mu$ | Training scale $s$ | Example raw value | Standardized example |
|---|---:|---:|---:|---:|
| age | 29.204129 | 12.998833 | 22.0000 | $(22.0000-29.204129)/12.998833=-0.554214$ |
| sibsp | 0.553371 | 1.175578 | 1.0000 | $(1.0000-0.553371)/1.175578=0.379923$ |
| parch | 0.379213 | 0.791113 | 0.0000 | $(0.0000-0.379213)/0.791113=-0.479342$ |
| fare | 32.586276 | 51.933021 | 7.2500 | $(7.2500-32.586276)/51.933021=-0.487864$ |


---

## 11. The design matrix

Each training passenger becomes one row of numbers. With an intercept column, the row has this form:

$$\mathbf{x}_i=[1,A_i^*,S_i^*,P_i^*,F_i^*,I(C2),I(C3),I(Male),I(Q),I(S)]$$

Stacking all 712 training rows produces matrix $X$. The target vector is

$$\mathbf y=[y_1,y_2,\ldots,y_{712}]^T$$

The coefficient vector is

$$\boldsymbol\beta=[\beta_0,\beta_1,\ldots,\beta_9]^T$$

All training scores can then be written compactly as

$$\mathbf z=X\boldsymbol\beta$$


---

## 12. Likelihood of one passenger

A passenger with observed $y=1$ contributes probability $p$ to the likelihood. A passenger with observed $y=0$ contributes $1-p$. Both cases combine into one expression:

$$L_i=p_i^{y_i}(1-p_i)^{1-y_i}$$

If $y=1$ and $p=0.80$:

$$L_i=0.80^1(0.20)^0=0.80$$

If $y=0$ and $p=0.20$:

$$L_i=0.20^0(0.80)^1=0.80$$

Thus, assigning high probability to the observed class produces a large likelihood contribution.


---

## 13. Likelihood of all training passengers

Assuming passenger outcomes are conditionally independent given the predictors, multiply all individual contributions:

$$L(\boldsymbol\beta)=\prod_{i=1}^{712}p_i^{y_i}(1-p_i)^{1-y_i}$$

The selected coefficients are the values that maximize this product. Direct multiplication creates extremely small numbers, so the natural logarithm is used.


---

## 14. Log-likelihood

Taking logs converts multiplication to addition:

$$\ell(\boldsymbol\beta)=\sum_{i=1}^{712}\left[y_i\ln(p_i)+(1-y_i)\ln(1-p_i)\right]$$

For three hypothetical observations $(y,p)=(1,0.8),(0,0.3),(1,0.6)$:

$$\ell=\ln(0.8)+\ln(0.7)+\ln(0.6)$$
$$=-0.2231-0.3567-0.5108=-1.0906$$

A value closer to zero is better because probabilities are at most 1, and their logarithms are at most 0.


---

## 15. Gradient or score equations

The derivative of log-likelihood with respect to coefficient $\beta_j$ is

$$\frac{\partial\ell}{\partial\beta_j}=\sum_{i=1}^{n}x_{ij}(y_i-p_i)$$

In vector notation:

$$
abla\ell=X^T(\mathbf y-\mathbf p)$$

At the optimum, every score equation is approximately zero:

$$X^T(\mathbf y-\mathbf p)\approx\mathbf0$$

The residual $y_i-p_i$ controls the direction. A survivor assigned too little probability has a positive residual and pushes relevant coefficients upward. A non-survivor assigned too much probability has a negative residual and pushes them downward.


---

## 16. Hessian and curvature

The second-derivative matrix is

$$H=-X^TWX$$

where $W$ is diagonal and

$$W_{ii}=p_i(1-p_i)$$

At $p_i=0.5$, $W_{ii}=0.25$. At probabilities near 0 or 1, the weight becomes smaller. The Hessian measures curvature and determines how far coefficients should move during each optimization step.


---

## 17. Newton-Raphson coefficient estimation

Start with every coefficient equal to zero. Every score is then $z_i=0$ and every probability is $p_i=0.5$. The first gradient is $X^T(y-0.5)$.

| Parameter | Initial gradient | First Newton estimate |
|---|---:|---:|
| Intercept | -88.000000 | 1.798283 |
| age | -16.760476 | -0.236464 |
| sibsp | -16.420327 | -0.179113 |
| parch | 27.013564 | -0.068007 |
| fare | 85.079279 | 0.105706 |
| Class=2 | -2.500000 | -0.342759 |
| Class=3 | -103.000000 | -1.141782 |
| Sex=male | -146.500000 | -2.045070 |
| Embarked=Q | -8.000000 | -0.096655 |
| Embarked=S | -85.500000 | -0.313748 |


The Newton update is

$$\boldsymbol\beta_{new}=\boldsymbol\beta_{old}+(X^TWX)^{-1}X^T(\mathbf y-\mathbf p)$$

After the first update, the scores and probabilities are recalculated, a new $W$ is formed, and the operation repeats. Iteration stops when the coefficient changes and gradient are negligibly small. This is the matrix arithmetic that numerical software performs; it is not a different statistical method.


---

## 18. Final fitted equation

The converged fitted log-odds equation is

$$z=2.60238295 -0.39049210\,A^* -0.35002473\,Sib^* -0.10805133\,Par^* +0.17052542\,Fare^* -0.54208792\,I(C2) -1.81442140\,I(C3) -2.71661900\,I(Male) -0.15904829\,I(Q) -0.48853644\,I(S)$$

Then

$$\hat p=\frac{1}{1+e^{-z}}$$

| Term | Coefficient | Odds ratio $e^{\beta}$ |
|---|---:|---:|
| Intercept | 2.60238295 | 13.495860 |
| age | -0.39049210 | 0.676724 |
| sibsp | -0.35002473 | 0.704671 |
| parch | -0.10805133 | 0.897582 |
| fare | 0.17052542 | 1.185928 |
| Class=2 | -0.54208792 | 0.581533 |
| Class=3 | -1.81442140 | 0.162932 |
| Sex=male | -2.71661900 | 0.066098 |
| Embarked=Q | -0.15904829 | 0.852955 |
| Embarked=S | -0.48853644 | 0.613524 |

Coefficients are conditional effects, meaning each interpretation holds the other included predictors constant.


---

## 19. Complete prediction recipe

For any passenger:

1. Replace missing age with the training median.
2. Replace missing embarkation with the training mode.
3. Standardize age, siblings/spouses, parents/children, and fare.
4. Create class, sex, and embarkation indicators.
5. Multiply every prepared value by its fitted coefficient.
6. Add the intercept to obtain $z$.
7. Calculate $e^{-z}$.
8. Calculate $p=1/(1+e^{-z})$.
9. Predict survived when $p\ge0.50$; otherwise predict did not survive.
10. Compare the prediction with the actual target only for evaluation, never while predicting.


---

## 20. Detailed passenger calculations

The following examples expose every contribution to the linear score. Values are rounded for display; final probabilities use unrounded arithmetic.

### Passenger 710

Raw values: class 3, sex male, age nan, sibsp 1, parch 1, fare 15.2458, embarked C.

$$z=2.602383+0.036173-0.132983-0.084788-0.056939-0.000000-1.814421-2.716619-0.000000-0.000000=-2.167194$$

$$p=\frac{1}{1+e^{-(-2.167194)}}=0.102735$$

Prediction: **0 (did not survive)**. Actual: **1**. Result: **incorrect**.

### Passenger 440

Raw values: class 2, sex male, age 31.0, sibsp 0, parch 0, fare 10.5, embarked S.

$$z=2.602383-0.053949+0.164764+0.051794-0.072522-0.542088-0.000000-2.716619-0.000000-0.488536=-1.054773$$

$$p=\frac{1}{1+e^{-(-1.054773)}}=0.258310$$

Prediction: **0 (did not survive)**. Actual: **0**. Result: **correct**.

### Passenger 841

Raw values: class 3, sex male, age 20.0, sibsp 0, parch 0, fare 7.925, embarked S.

$$z=2.602383+0.276497+0.164764+0.051794-0.080977-0.000000-1.814421-2.716619-0.000000-0.488536=-2.005116$$

$$p=\frac{1}{1+e^{-(-2.005116)}}=0.118667$$

Prediction: **0 (did not survive)**. Actual: **0**. Result: **correct**.

### Passenger 721

Raw values: class 2, sex female, age 6.0, sibsp 0, parch 1, fare 33.0, embarked S.

$$z=2.602383+0.697065+0.164764-0.084788+0.001358-0.542088-0.000000-0.000000-0.000000-0.488536=2.350158$$

$$p=\frac{1}{1+e^{-(2.350158)}}=0.912947$$

Prediction: **1 (survived)**. Actual: **1**. Result: **correct**.

### Passenger 40

Raw values: class 3, sex female, age 14.0, sibsp 1, parch 0, fare 11.2417, embarked C.

$$z=2.602383+0.456740-0.132983+0.051794-0.070086-0.000000-1.814421-0.000000-0.000000-0.000000=1.093427$$

$$p=\frac{1}{1+e^{-(1.093427)}}=0.749026$$

Prediction: **1 (survived)**. Actual: **1**. Result: **correct**.

### Passenger 291

Raw values: class 1, sex female, age 26.0, sibsp 0, parch 0, fare 78.85, embarked S.

$$z=2.602383+0.096254+0.164764+0.051794+0.151910-0.000000-0.000000-0.000000-0.000000-0.488536=2.578568$$

$$p=\frac{1}{1+e^{-(2.578568)}}=0.929469$$

Prediction: **1 (survived)**. Actual: **1**. Result: **correct**.

### Passenger 301

Raw values: class 3, sex female, age nan, sibsp 0, parch 0, fare 7.75, embarked Q.

$$z=2.602383+0.036173+0.164764+0.051794-0.081552-0.000000-1.814421-0.000000-0.159048-0.000000=0.800092$$

$$p=\frac{1}{1+e^{-(0.800092)}}=0.689994$$

Prediction: **1 (survived)**. Actual: **1**. Result: **correct**.

### Passenger 334

Raw values: class 3, sex male, age 16.0, sibsp 2, parch 0, fare 18.0, embarked S.

$$z=2.602383+0.396659-0.430729+0.051794-0.047895-0.000000-1.814421-2.716619-0.000000-0.488536=-2.447366$$

$$p=\frac{1}{1+e^{-(-2.447366)}}=0.079631$$

Prediction: **0 (did not survive)**. Actual: **0**. Result: **correct**.

### Passenger 209

Raw values: class 3, sex female, age 16.0, sibsp 0, parch 0, fare 7.75, embarked Q.

$$z=2.602383+0.396659+0.164764+0.051794-0.081552-0.000000-1.814421-0.000000-0.159048-0.000000=1.160579$$

$$p=\frac{1}{1+e^{-(1.160579)}}=0.761438$$

Prediction: **1 (survived)**. Actual: **1**. Result: **correct**.

### Passenger 137

Raw values: class 1, sex female, age 19.0, sibsp 0, parch 2, fare 26.2833, embarked S.

$$z=2.602383+0.306538+0.164764-0.221369-0.020696-0.000000-0.000000-0.000000-0.000000-0.488536=2.343083$$

$$p=\frac{1}{1+e^{-(2.343083)}}=0.912383$$

Prediction: **1 (survived)**. Actual: **1**. Result: **correct**.


---

## 21. All 179 test predictions

Each probability below comes from the complete equation in Chapter 18. `NA` indicates a raw missing age that was replaced by the training median before standardization.

| Passenger | Class | Sex | Age | Fare | Actual | Probability | Prediction | Correct? |
|---:|---:|---|---:|---:|---:|---:|---:|---|
| 6 | 3 | male | NA | 8.4583 | 0 | 0.128510 | 0 | Yes |
| 11 | 3 | female | 4.00 | 16.7000 | 1 | 0.687109 | 1 | Yes |
| 24 | 1 | male | 28.00 | 35.5000 | 1 | 0.415694 | 0 | No |
| 26 | 3 | female | 38.00 | 31.3875 | 1 | 0.324572 | 0 | No |
| 31 | 1 | male | 40.00 | 27.7208 | 0 | 0.440787 | 0 | Yes |
| 32 | 1 | female | NA | 146.5208 | 1 | 0.949381 | 1 | Yes |
| 34 | 2 | male | 66.00 | 10.5000 | 0 | 0.108496 | 0 | Yes |
| 40 | 3 | female | 14.00 | 11.2417 | 1 | 0.749026 | 1 | Yes |
| 45 | 3 | female | 19.00 | 7.8792 | 1 | 0.744765 | 1 | Yes |
| 50 | 3 | female | 18.00 | 17.8000 | 0 | 0.623930 | 1 | No |
| 55 | 1 | male | 65.00 | 61.9792 | 0 | 0.266379 | 0 | Yes |
| 64 | 3 | male | 4.00 | 27.9000 | 0 | 0.067529 | 0 | Yes |
| 66 | 3 | male | NA | 15.2458 | 1 | 0.102735 | 0 | No |
| 67 | 2 | female | 29.00 | 10.5000 | 1 | 0.848377 | 1 | Yes |
| 68 | 3 | male | 19.00 | 8.1583 | 0 | 0.121927 | 0 | Yes |
| 71 | 2 | male | 32.00 | 10.5000 | 0 | 0.252596 | 0 | Yes |
| 73 | 2 | male | 21.00 | 73.5000 | 0 | 0.366444 | 0 | Yes |
| 77 | 3 | male | NA | 7.8958 | 0 | 0.095735 | 0 | Yes |
| 78 | 3 | male | NA | 8.0500 | 0 | 0.095779 | 0 | Yes |
| 79 | 2 | male | 0.83 | 29.0000 | 1 | 0.410750 | 0 | No |
| 83 | 3 | female | NA | 7.7875 | 1 | 0.690021 | 1 | Yes |
| 85 | 2 | female | 17.00 | 10.5000 | 1 | 0.889182 | 1 | Yes |
| 87 | 3 | male | 16.00 | 34.3750 | 0 | 0.075467 | 0 | Yes |
| 97 | 1 | male | 71.00 | 34.6542 | 0 | 0.241136 | 0 | Yes |
| 98 | 1 | male | 23.00 | 63.3583 | 1 | 0.562954 | 1 | Yes |
| 110 | 3 | female | NA | 24.1500 | 1 | 0.635572 | 1 | Yes |
| 111 | 1 | male | 47.00 | 52.0000 | 0 | 0.297953 | 0 | Yes |
| 121 | 2 | male | 21.00 | 73.5000 | 0 | 0.241771 | 0 | Yes |
| 137 | 1 | female | 19.00 | 26.2833 | 1 | 0.912383 | 1 | Yes |
| 138 | 1 | male | 37.00 | 53.1000 | 0 | 0.299266 | 0 | Yes |
| 140 | 1 | male | 24.00 | 79.2000 | 0 | 0.601499 | 1 | No |
| 142 | 3 | female | 22.00 | 7.7500 | 1 | 0.657201 | 1 | Yes |
| 166 | 3 | male | 9.00 | 20.5250 | 1 | 0.129379 | 0 | No |
| 169 | 1 | male | NA | 25.9250 | 0 | 0.408078 | 0 | Yes |
| 175 | 1 | male | 56.00 | 30.6958 | 0 | 0.329853 | 0 | Yes |
| 193 | 3 | female | 19.00 | 7.8542 | 1 | 0.609108 | 1 | Yes |
| 197 | 3 | male | NA | 7.7500 | 0 | 0.128249 | 0 | Yes |
| 199 | 3 | female | NA | 7.7500 | 1 | 0.689994 | 1 | Yes |
| 205 | 3 | male | 18.00 | 8.0500 | 1 | 0.125141 | 0 | No |
| 209 | 3 | female | 16.00 | 7.7500 | 1 | 0.761438 | 1 | Yes |
| 210 | 1 | male | 40.00 | 31.0000 | 1 | 0.443443 | 0 | No |
| 211 | 3 | male | 24.00 | 7.0500 | 0 | 0.106391 | 0 | Yes |
| 212 | 2 | female | 35.00 | 21.0000 | 1 | 0.828660 | 1 | Yes |
| 214 | 2 | male | 30.00 | 13.0000 | 0 | 0.265705 | 0 | Yes |
| 216 | 1 | female | 31.00 | 113.2750 | 1 | 0.938900 | 1 | Yes |
| 219 | 1 | female | 32.00 | 76.2917 | 1 | 0.946771 | 1 | Yes |
| 228 | 3 | male | 20.50 | 7.2500 | 0 | 0.116876 | 0 | Yes |
| 232 | 3 | male | 29.00 | 7.7750 | 0 | 0.093132 | 0 | Yes |
| 236 | 3 | female | NA | 7.5500 | 0 | 0.615371 | 1 | No |
| 245 | 3 | male | 30.00 | 7.2250 | 0 | 0.139520 | 0 | Yes |
| 251 | 3 | male | NA | 7.2500 | 0 | 0.095552 | 0 | Yes |
| 255 | 3 | female | 41.00 | 20.2125 | 0 | 0.462032 | 0 | Yes |
| 267 | 3 | male | 16.00 | 39.6875 | 0 | 0.042769 | 0 | Yes |
| 276 | 1 | female | 63.00 | 77.9583 | 1 | 0.762492 | 1 | Yes |
| 281 | 3 | male | 65.00 | 7.7500 | 0 | 0.046176 | 0 | Yes |
| 282 | 3 | male | 28.00 | 7.8542 | 0 | 0.095723 | 0 | Yes |
| 287 | 3 | male | 30.00 | 9.5000 | 1 | 0.091094 | 0 | No |
| 291 | 1 | female | 26.00 | 78.8500 | 1 | 0.929469 | 1 | Yes |
| 292 | 1 | female | 19.00 | 91.0792 | 1 | 0.953461 | 1 | Yes |
| 293 | 2 | male | 36.00 | 12.8750 | 0 | 0.329899 | 0 | Yes |
| 295 | 3 | male | 24.00 | 7.8958 | 0 | 0.106655 | 0 | Yes |
| 297 | 3 | male | 23.50 | 7.2292 | 0 | 0.164653 | 0 | Yes |
| 300 | 1 | female | 50.00 | 247.5208 | 1 | 0.940665 | 1 | Yes |
| 301 | 3 | female | NA | 7.7500 | 1 | 0.689994 | 1 | Yes |
| 307 | 1 | female | NA | 110.8833 | 1 | 0.957393 | 1 | Yes |
| 310 | 1 | female | 30.00 | 56.9292 | 1 | 0.946595 | 1 | Yes |
| 312 | 1 | female | 18.00 | 262.3750 | 1 | 0.954409 | 1 | Yes |
| 313 | 2 | female | 26.00 | 26.0000 | 0 | 0.806686 | 1 | No |
| 320 | 1 | female | 40.00 | 134.5000 | 1 | 0.916441 | 1 | Yes |
| 322 | 3 | male | 27.00 | 7.8958 | 0 | 0.098368 | 0 | Yes |
| 324 | 2 | female | 22.00 | 29.0000 | 1 | 0.826156 | 1 | Yes |
| 328 | 2 | female | 36.00 | 13.0000 | 1 | 0.820517 | 1 | Yes |
| 334 | 3 | male | 16.00 | 18.0000 | 0 | 0.079631 | 0 | Yes |
| 339 | 3 | male | 45.00 | 8.0500 | 1 | 0.059764 | 0 | No |
| 343 | 2 | male | 28.00 | 13.0000 | 0 | 0.277591 | 0 | Yes |
| 345 | 2 | male | 36.00 | 13.0000 | 0 | 0.232051 | 0 | Yes |
| 351 | 3 | male | 23.00 | 9.2250 | 0 | 0.109978 | 0 | Yes |
| 358 | 2 | female | 38.00 | 13.0000 | 0 | 0.811498 | 1 | No |
| 360 | 3 | female | NA | 7.8792 | 1 | 0.690085 | 1 | Yes |
| 361 | 3 | male | 40.00 | 27.9000 | 0 | 0.032786 | 0 | Yes |
| 363 | 3 | female | 45.00 | 14.4542 | 0 | 0.582704 | 1 | No |
| 364 | 3 | male | 35.00 | 7.0500 | 0 | 0.078812 | 0 | Yes |
| 376 | 1 | female | NA | 82.1708 | 1 | 0.938207 | 1 | Yes |
| 389 | 3 | male | NA | 7.7292 | 0 | 0.128242 | 0 | Yes |
| 397 | 3 | female | 31.00 | 7.8542 | 0 | 0.594072 | 1 | No |
| 399 | 2 | male | 23.00 | 10.5000 | 0 | 0.306943 | 0 | Yes |
| 409 | 3 | male | 21.00 | 7.7750 | 0 | 0.115511 | 0 | Yes |
| 417 | 2 | female | 34.00 | 32.5000 | 1 | 0.770234 | 1 | Yes |
| 418 | 2 | female | 18.00 | 13.0000 | 1 | 0.856611 | 1 | Yes |
| 423 | 3 | male | 29.00 | 7.8750 | 0 | 0.093160 | 0 | Yes |
| 427 | 2 | female | 28.00 | 26.0000 | 1 | 0.818336 | 1 | Yes |
| 434 | 3 | male | 17.00 | 7.1250 | 0 | 0.128127 | 0 | Yes |
| 440 | 2 | male | 31.00 | 10.5000 | 0 | 0.258310 | 0 | Yes |
| 443 | 3 | male | 25.00 | 7.7750 | 0 | 0.079179 | 0 | Yes |
| 447 | 2 | female | 13.00 | 19.5000 | 1 | 0.890470 | 1 | Yes |
| 448 | 1 | male | 34.00 | 26.5500 | 1 | 0.365840 | 0 | No |
| 449 | 3 | female | 5.00 | 19.2583 | 1 | 0.722277 | 1 | Yes |
| 458 | 1 | female | NA | 51.8625 | 1 | 0.893987 | 1 | Yes |
| 463 | 1 | male | 47.00 | 38.5000 | 0 | 0.288764 | 0 | Yes |
| 466 | 3 | male | 38.00 | 7.0500 | 0 | 0.072513 | 0 | Yes |
| 478 | 3 | male | 29.00 | 7.0458 | 0 | 0.070692 | 0 | Yes |
| 482 | 2 | male | NA | 0.0000 | 0 | 0.269113 | 0 | Yes |
| 486 | 3 | female | NA | 25.4667 | 0 | 0.377295 | 0 | Yes |
| 494 | 1 | male | 71.00 | 49.5042 | 0 | 0.250171 | 0 | Yes |
| 495 | 3 | male | 21.00 | 8.0500 | 0 | 0.115603 | 0 | Yes |
| 508 | 1 | male | NA | 26.5500 | 1 | 0.408574 | 0 | No |
| 513 | 1 | male | 36.00 | 26.2875 | 1 | 0.351820 | 0 | No |
| 515 | 3 | male | 24.00 | 7.4958 | 0 | 0.106530 | 0 | Yes |
| 519 | 2 | female | 36.00 | 26.0000 | 1 | 0.779850 | 1 | Yes |
| 522 | 3 | male | 22.00 | 7.8958 | 0 | 0.112516 | 0 | Yes |
| 528 | 1 | male | NA | 221.7792 | 0 | 0.567384 | 1 | No |
| 530 | 2 | male | 23.00 | 11.5000 | 0 | 0.176065 | 0 | Yes |
| 533 | 3 | male | 17.00 | 7.2292 | 0 | 0.134346 | 0 | Yes |
| 539 | 3 | male | NA | 14.5000 | 0 | 0.097629 | 0 | Yes |
| 540 | 1 | female | 22.00 | 49.5000 | 1 | 0.943627 | 1 | Yes |
| 542 | 3 | female | 9.00 | 31.2750 | 0 | 0.414467 | 0 | Yes |
| 546 | 1 | male | 64.00 | 26.0000 | 0 | 0.189521 | 0 | Yes |
| 573 | 1 | male | 36.00 | 26.3875 | 1 | 0.351895 | 0 | No |
| 579 | 3 | female | NA | 14.4583 | 0 | 0.664502 | 1 | No |
| 585 | 3 | male | NA | 8.7125 | 0 | 0.147503 | 0 | Yes |
| 586 | 1 | female | 18.00 | 79.6500 | 1 | 0.927463 | 1 | Yes |
| 591 | 3 | male | 35.00 | 7.1250 | 0 | 0.078830 | 0 | Yes |
| 592 | 1 | female | 52.00 | 78.2667 | 1 | 0.879360 | 1 | Yes |
| 595 | 2 | male | 37.00 | 26.0000 | 0 | 0.185145 | 0 | Yes |
| 600 | 1 | male | 49.00 | 56.9292 | 1 | 0.329561 | 0 | No |
| 605 | 1 | male | 35.00 | 26.5500 | 1 | 0.477114 | 0 | No |
| 606 | 3 | male | 36.00 | 15.5500 | 0 | 0.059610 | 0 | Yes |
| 612 | 3 | male | NA | 7.0500 | 0 | 0.095495 | 0 | Yes |
| 621 | 3 | male | 27.00 | 14.4542 | 0 | 0.118870 | 0 | Yes |
| 622 | 1 | male | 42.00 | 52.5542 | 1 | 0.268393 | 0 | No |
| 626 | 1 | male | 61.00 | 32.3208 | 0 | 0.207140 | 0 | Yes |
| 644 | 3 | male | NA | 56.4958 | 1 | 0.110469 | 0 | No |
| 653 | 3 | male | 21.00 | 8.4333 | 0 | 0.115732 | 0 | Yes |
| 654 | 3 | female | NA | 7.8292 | 1 | 0.690050 | 1 | Yes |
| 658 | 3 | female | 32.00 | 15.5000 | 0 | 0.567351 | 1 | No |
| 660 | 1 | male | 58.00 | 113.2750 | 0 | 0.316276 | 0 | Yes |
| 669 | 3 | male | 43.00 | 8.0500 | 0 | 0.063231 | 0 | Yes |
| 670 | 1 | female | NA | 52.0000 | 1 | 0.894029 | 1 | Yes |
| 671 | 2 | female | 40.00 | 39.0000 | 1 | 0.740916 | 1 | Yes |
| 674 | 2 | male | 31.00 | 13.0000 | 1 | 0.259885 | 0 | No |
| 683 | 3 | male | 20.00 | 9.2250 | 0 | 0.119114 | 0 | Yes |
| 689 | 3 | male | 18.00 | 7.7958 | 0 | 0.125049 | 0 | Yes |
| 697 | 3 | male | 44.00 | 8.0500 | 0 | 0.061475 | 0 | Yes |
| 710 | 3 | male | NA | 15.2458 | 1 | 0.102735 | 0 | No |
| 713 | 1 | male | 48.00 | 52.0000 | 1 | 0.234181 | 0 | No |
| 721 | 2 | female | 6.00 | 33.0000 | 1 | 0.912947 | 1 | Yes |
| 737 | 3 | female | 48.00 | 34.3750 | 0 | 0.320762 | 0 | Yes |
| 740 | 3 | male | NA | 7.8958 | 0 | 0.095735 | 0 | Yes |
| 768 | 3 | female | 30.50 | 7.7500 | 0 | 0.673705 | 1 | No |
| 773 | 2 | female | 57.00 | 10.5000 | 0 | 0.706986 | 1 | No |
| 774 | 3 | male | NA | 7.2250 | 0 | 0.146890 | 0 | Yes |
| 778 | 3 | female | 5.00 | 12.4750 | 1 | 0.764417 | 1 | Yes |
| 779 | 3 | male | NA | 7.7375 | 0 | 0.128245 | 0 | Yes |
| 786 | 3 | male | 25.00 | 7.2500 | 0 | 0.103629 | 0 | Yes |
| 791 | 3 | male | NA | 7.7500 | 0 | 0.128249 | 0 | Yes |
| 794 | 1 | male | NA | 30.6958 | 0 | 0.533024 | 1 | No |
| 797 | 1 | female | 49.00 | 25.9292 | 1 | 0.847337 | 1 | Yes |
| 803 | 1 | male | 11.00 | 120.0000 | 1 | 0.469231 | 0 | No |
| 809 | 2 | male | 39.00 | 13.0000 | 0 | 0.216380 | 0 | Yes |
| 815 | 3 | male | 30.50 | 8.0500 | 0 | 0.089469 | 0 | Yes |
| 820 | 3 | male | 10.00 | 27.9000 | 0 | 0.057026 | 0 | Yes |
| 821 | 1 | female | 52.00 | 93.5000 | 1 | 0.803969 | 1 | Yes |
| 823 | 1 | male | 38.00 | 0.0000 | 0 | 0.319201 | 0 | Yes |
| 831 | 3 | female | 15.00 | 14.4542 | 1 | 0.745344 | 1 | Yes |
| 834 | 3 | male | 23.00 | 7.8542 | 0 | 0.109538 | 0 | Yes |
| 836 | 1 | female | 39.00 | 83.1583 | 1 | 0.905201 | 1 | Yes |
| 838 | 3 | male | NA | 8.0500 | 0 | 0.095779 | 0 | Yes |
| 841 | 3 | male | 20.00 | 7.9250 | 0 | 0.118667 | 0 | Yes |
| 843 | 1 | female | 30.00 | 31.0000 | 1 | 0.942123 | 1 | Yes |
| 846 | 3 | male | 42.00 | 7.5500 | 0 | 0.064934 | 0 | Yes |
| 850 | 1 | female | NA | 89.1042 | 1 | 0.939514 | 1 | Yes |
| 853 | 3 | female | 9.00 | 15.2458 | 0 | 0.754025 | 1 | No |
| 854 | 1 | female | 16.00 | 39.4000 | 1 | 0.931684 | 1 | Yes |
| 855 | 2 | female | 44.00 | 26.0000 | 0 | 0.735842 | 1 | No |
| 877 | 3 | male | 20.00 | 9.8458 | 0 | 0.119328 | 0 | Yes |
| 878 | 3 | male | 19.00 | 7.8958 | 0 | 0.121835 | 0 | Yes |
| 886 | 3 | female | 39.00 | 29.1250 | 0 | 0.464292 | 0 | Yes |
| 887 | 2 | male | 27.00 | 13.0000 | 0 | 0.283655 | 0 | Yes |
| 890 | 1 | male | 26.00 | 30.0000 | 1 | 0.547378 | 1 | Yes |


---

## 22. Confusion matrix by hand

The 179 predictions produce:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | TN = 90 | FP = 15 |
| Actual 1 | FN = 21 | TP = 53 |

Check:

$$90+15+21+53=179=179$$

Correct predictions:

$$TN+TP=90+53=143$$

Incorrect predictions:

$$FP+FN=15+21=36$$


---

## 23. Accuracy, precision, recall, and F1

### Accuracy

$$\frac{TP+TN}{N}=\frac{53+90}{179}=\frac{143}{179}=0.798883=79.89\%$$

### Precision

$$\frac{TP}{TP+FP}=\frac{53}{53+15}=0.779412$$

### Recall

$$\frac{TP}{TP+FN}=\frac{53}{53+21}=0.716216$$

### F1 score

$$F1=\frac{2	imes Precision	imes Recall}{Precision+Recall}$$

$$=\frac{2	imes0.779412	imes0.716216}{0.779412+0.716216}=0.746479$$


---

## 24. Log loss and ROC-AUC

For one passenger, binary log loss is

$$-[y\ln(p)+(1-y)\ln(1-p)]$$

Average test log loss is

$$-\frac1{179}\sum_{i=1}^{179}[y_i\ln(p_i)+(1-y_i)\ln(1-p_i)]=0.423058$$

Lower log loss is better. It evaluates probability quality, not only final 0/1 decisions.

The test ROC-AUC is **0.876577**. It can be interpreted as the probability that a randomly selected survivor receives a higher score than a randomly selected non-survivor, with ties handled conventionally.


---

## 25. Interpreting coefficients and odds ratios

An odds ratio is $e^{\beta}$. For a one-unit increase in a standardized numerical feature, multiply the odds by its odds ratio. For an indicator, compare category 1 with its reference category.

- **age:** $e^{-0.390492}=0.676724$. The odds decrease.
- **sibsp:** $e^{-0.350025}=0.704671$. The odds decrease.
- **parch:** $e^{-0.108051}=0.897582$. The odds decrease.
- **fare:** $e^{0.170525}=1.185928$. The odds increase.
- **Class=2:** $e^{-0.542088}=0.581533$. The odds decrease.
- **Class=3:** $e^{-1.814421}=0.162932$. The odds decrease.
- **Sex=male:** $e^{-2.716619}=0.066098$. The odds decrease.
- **Embarked=Q:** $e^{-0.159048}=0.852955$. The odds decrease.
- **Embarked=S:** $e^{-0.488536}=0.613524$. The odds decrease.

A coefficient is not a direct percentage-point change in probability. The same coefficient produces different probability changes depending on the starting score.


---

## 26. Decision threshold experiments

Changing the threshold changes the operational balance between false positives and false negatives.

| Threshold | TN | FP | FN | TP | Accuracy | Precision | Recall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 77 | 28 | 10 | 64 | 0.7877 | 0.6957 | 0.8649 |
| 0.40 | 85 | 20 | 15 | 59 | 0.8045 | 0.7468 | 0.7973 |
| 0.50 | 90 | 15 | 21 | 53 | 0.7989 | 0.7794 | 0.7162 |
| 0.60 | 95 | 10 | 23 | 51 | 0.8156 | 0.8361 | 0.6892 |
| 0.70 | 100 | 5 | 32 | 42 | 0.7933 | 0.8936 | 0.5676 |


---

## 27. Errors and residuals

The raw probability residual is

$$r_i=y_i-p_i$$

For a survivor with $p=0.80$:

$$r=1-0.80=0.20$$

For a non-survivor with $p=0.20$:

$$r=0-0.20=-0.20$$

Large positive residuals represent survivors the model underestimated. Large negative residuals represent non-survivors the model overestimated. These residuals contribute directly to the gradient.


---

## 28. Assumptions and limitations

1. The outcome is binary.
2. Passenger rows are treated as conditionally independent.
3. Continuous predictors have a linear relationship with log-odds, not necessarily with probability.
4. No included predictor is an exact linear combination of the others.
5. Coefficients describe association within this dataset and do not automatically establish causation.
6. Missing-age replacement reduces information and may hide subgroup differences.
7. The train/test split is only one fixed sample division; another split can produce somewhat different coefficients.
8. A 0.50 threshold is conventional, not universally optimal.
9. The model does not know unrecorded circumstances affecting individual survival.
10. “Exact” in this handbook means exact for the documented preprocessing, split, fitted equation, threshold, and floating-point arithmetic. It does not mean perfect prediction of every passenger.


---

## 29. Formula reference sheet

### Linear score
$$z=\beta_0+\sum_{j=1}^k\beta_jx_j$$

### Sigmoid
$$p=\frac1{1+e^{-z}}$$

### Odds
$$	ext{Odds}=\frac{p}{1-p}$$

### Logit
$$\log\left(\frac{p}{1-p}\right)=z$$

### Standardization
$$x^*=\frac{x-\mu}{s}$$

### Likelihood
$$L=\prod_i p_i^{y_i}(1-p_i)^{1-y_i}$$

### Log-likelihood
$$\ell=\sum_i[y_i\ln p_i+(1-y_i)\ln(1-p_i)]$$

### Gradient
$$
abla\ell=X^T(y-p)$$

### Newton update
$$\beta_{new}=\beta_{old}+(X^TWX)^{-1}X^T(y-p)$$

### Accuracy
$$\frac{TP+TN}{TP+TN+FP+FN}$$

### Precision
$$\frac{TP}{TP+FP}$$

### Recall
$$\frac{TP}{TP+FN}$$

### F1
$$\frac{2PR}{P+R}$$


---

## 30. Practice calculations with answers

### Exercise 1
If $z=-1.50$, calculate survival probability.

**Answer:**
$$p=\frac1{1+e^{1.5}}=\frac1{1+4.4817}=0.1824$$
Prediction at 0.50: **0**.

### Exercise 2
If $p=0.75$, calculate odds and log-odds.

**Answer:**
$$Odds=0.75/0.25=3$$
$$Log\ odds=\ln(3)=1.0986$$

### Exercise 3
If age is 22, training mean is 29.50, and scale is 13.00, standardize age.

**Answer:**
$$Age^*=(22-29.50)/13.00=-0.5769$$

### Exercise 4
For $TP=60$, $FP=15$, and $FN=20$, calculate precision and recall.

**Answer:**
$$Precision=60/(60+15)=0.8000$$
$$Recall=60/(60+20)=0.7500$$

### Exercise 5
Explain why a coefficient of 0.7 is not a 70% increase in probability.

**Answer:** The coefficient changes log-odds by 0.7 and multiplies odds by $e^{0.7}=2.0138$. The probability change depends on the original probability.


---

## Appendix A. Reproducibility specification

To reproduce the same numerical results, use exactly these rules:

- Preserve the original 891-row order.
- Use predictors: pclass, sex, age, sibsp, parch, fare, embarked.
- Shuffle and split 80/20 with fixed state 42, producing 712 training and 179 test rows.
- Fit medians, modes, means, and scales from training data only.
- Use reference categories class 1, female, and embarked C.
- Estimate an effectively unregularized binary maximum-likelihood logistic model.
- Use a 0.50 decision threshold.
- Retain unrounded values during calculations and round only for display.

## Appendix B. Final checklist

- [x] Target defined mathematically
- [x] Data preparation shown without code
- [x] Missing values handled
- [x] Categorical variables encoded
- [x] Numerical variables standardized
- [x] Likelihood and log-likelihood derived
- [x] Gradient, Hessian, and Newton update explained
- [x] Final fitted coefficients reported
- [x] Passenger-level predictions listed
- [x] Confusion matrix and performance measures calculated
- [x] Assumptions and limitations documented