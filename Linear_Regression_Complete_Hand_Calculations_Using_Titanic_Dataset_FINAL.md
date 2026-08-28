# Linear Regression Complete Hand Calculations Using Titanic Dataset

## A Mathematics-Only Handbook for Estimating and Classifying Passenger Survival

**Dataset:** `titanic.csv`  
**Observations:** 891 passengers  
**Target:** `survived`, where 1 = survived and 0 = did not survive  
**Primary method:** Multiple linear regression, also called the Linear Probability Model for a binary target  
**Programming code in this handbook:** None  
**Split:** 80% training and 20% testing with fixed shuffle state 42  
**Classification threshold:** 0.50

> This handbook reproduces the complete arithmetic logic of linear regression. A computer was used only to perform the large quantity of arithmetic accurately while preparing the tables. No programming code is shown or taught.

> Important: because the target is binary, logistic regression is normally more appropriate for probability modeling. Linear regression is used here as a learning exercise and as a Linear Probability Model. Its fitted values are scores, not guaranteed valid probabilities.

---
## Table of Contents
1. [1. Prediction objective](#1-prediction-objective)
2. [2. Titanic variables used](#2-titanic-variables-used)
3. [3. What linear regression calculates](#3-what-linear-regression-calculates)
4. [4. Binary target as a linear probability model](#4-binary-target-as-a-linear-probability-model)
5. [5. Data audit and target mean](#5-data-audit-and-target-mean)
6. [6. Train and test split arithmetic](#6-train-and-test-split-arithmetic)
7. [7. Missing-value calculations](#7-missing-value-calculations)
8. [8. Dummy-variable encoding](#8-dummy-variable-encoding)
9. [9. Standardization by hand](#9-standardization-by-hand)
10. [10. Design matrix construction](#10-design-matrix-construction)
11. [11. The linear regression equation](#11-the-linear-regression-equation)
12. [12. Residuals and squared errors](#12-residuals-and-squared-errors)
13. [13. Sum of squared errors](#13-sum-of-squared-errors)
14. [14. Mean squared error cost function](#14-mean-squared-error-cost-function)
15. [15. Ordinary Least Squares principle](#15-ordinary-least-squares-principle)
16. [16. Normal equation derivation](#16-normal-equation-derivation)
17. [17. Matrix calculation of coefficients](#17-matrix-calculation-of-coefficients)
18. [18. Gradient descent alternative](#18-gradient-descent-alternative)
19. [19. Final fitted equation](#19-final-fitted-equation)
20. [20. Complete prediction recipe](#20-complete-prediction-recipe)
21. [21. Detailed passenger calculations](#21-detailed-passenger-calculations)
22. [22. All 179 test predictions](#22-all-179-test-predictions)
23. [23. Converting scores to classes](#23-converting-scores-to-classes)
24. [24. Confusion matrix calculations](#24-confusion-matrix-calculations)
25. [25. Classification evaluation metrics](#25-classification-evaluation-metrics)
26. [26. MAE, MSE, RMSE, and R-squared](#26-mae-mse-rmse-and-r-squared)
27. [27. Coefficient interpretation](#27-coefficient-interpretation)
28. [28. Threshold experiments](#28-threshold-experiments)
29. [29. Assumptions and limitations](#29-assumptions-and-limitations)
30. [30. Formula reference and practice answers](#30-formula-reference-and-practice-answers)

---

## 1. Prediction objective

For passenger \(i\), the observed target is

$$
y_i=\begin{cases}
1, & \text{passenger survived} \\
0, & \text{passenger did not survive}
\end{cases}
$$

Linear regression first calculates a continuous score:

$$
\hat{y}_i=b_0+b_1x_{i1}+b_2x_{i2}+\cdots+b_kx_{ik}
$$

For a secondary survived/not-survived decision:

$$
\hat{c}_i=\begin{cases}
1, & \hat{y}_i \ge 0.50 \\
0, & \hat{y}_i < 0.50
\end{cases}
$$

The uploaded dataset contains 891 records. The fixed split uses 712 records for estimating coefficients and 179 records for testing.

---

## 2. Titanic variables used

| Variable | Meaning | Mathematical treatment |
|---|---|---|
| `pclass` | Ticket class 1, 2, or 3 | Dummy variables |
| `sex` | female or male | Dummy variable |
| `age` | Age in years | Median replacement, then standardization |
| `sibsp` | Siblings and spouses aboard | Standardization |
| `parch` | Parents and children aboard | Standardization |
| `fare` | Fare paid | Standardization |
| `embarked` | Port C, Q, or S | Dummy variables |

The duplicate target column `alive` is excluded. Variables such as `adult_male`, `who`, and `class` are excluded because they repeat information already represented by the selected variables.

---

## 3. What linear regression calculates

A straight line with one predictor is

$$
\hat{y}=b_0+b_1x
$$

A plane or hyperplane with several predictors is

$$
\hat{y}=b_0+b_1x_1+b_2x_2+\cdots+b_kx_k
$$

Each coefficient is a slope. It tells us how much the fitted score changes when that prepared predictor increases by one unit while the other predictors stay fixed.

---

## 4. Binary target as a linear probability model

When the target contains only 0 and 1, the fitted value is often interpreted approximately as a survival probability. This use of ordinary least squares is called a **Linear Probability Model**.

However, the output is not constrained:

$$
- \infty < \hat{y} < +\infty
$$

Therefore, a score such as 1.08 does not mean a literal probability of 108%, and a score such as -0.12 does not mean a negative probability. These remain linear fitted scores. For presentation only, scores may be clipped to 0 and 1, but clipping is not used when calculating regression errors in this handbook.

---

## 5. Data audit and target mean

Total observations:

$$N=891$$

Survivors:

$$\sum_{i=1}^N y_i=342$$

Non-survivors:

$$N-\sum y_i=891-342=549$$

Target mean:

$$\bar{y}=\frac{342}{891}=0.383838$$

Because the target is 0 or 1, its mean equals the observed survival proportion, or 38.38%.

---

## 6. Train and test split arithmetic

The requested test proportion is 20%:

$$891\times0.20=178.2$$

The splitting procedure assigns **179 passengers** to testing and the remaining passengers to training:

$$891-179=712$$

Training target composition:

- Survivors: 268
- Non-survivors: 444

Test target composition:

- Survivors: 74
- Non-survivors: 105

Only training observations determine replacements, scaling quantities, and coefficients.

---

## 7. Missing-value calculations

Ordinary Least Squares requires a complete numerical matrix. Missing values are therefore replaced using training information only.

| Variable | Missing in full dataset | Training replacement | Rule |
|---|---:|---:|---|
| age | 177 | 28.0000 | Training median |
| sibsp | 0 | 0.0000 | Training median |
| parch | 0 | 0.0000 | Training median |
| fare | 0 | 14.4542 | Training median |
| pclass | 0 | 3 | Training mode |
| sex | 0 | male | Training mode |
| embarked | 2 | S | Training mode |


For an even number of sorted valid observations, the median is

$$\text{Median}=\frac{x_{(n/2)}+x_{(n/2+1)}}{2}$$

The mode is the category with the largest frequency. Missing test values do not influence either replacement.

---

## 8. Dummy-variable encoding

### Ticket class

| Class | I(Class 2) | I(Class 3) |
|---:|---:|---:|
| 1 | 0 | 0 |
| 2 | 1 | 0 |
| 3 | 0 | 1 |

### Sex

| Sex | I(Male) |
|---|---:|
| female | 0 |
| male | 1 |

### Embarkation

| Port | I(Q) | I(S) |
|---|---:|---:|
| C | 0 | 0 |
| Q | 1 | 0 |
| S | 0 | 1 |

Class 1, female, and port C are reference categories. One category is intentionally omitted from each group to prevent perfect multicollinearity.

---

## 9. Standardization by hand

For numerical variable \(x\), calculate

$$x^*=\frac{x-\mu_x}{s_x}$$

| Variable | Training mean | Training scale | Example raw value | Hand calculation |
|---|---:|---:|---:|---|
| age | 29.204129 | 12.998833 | 22.0000 | $(22.0000-29.204129)/12.998833=-0.554214$ |
| sibsp | 0.553371 | 1.175578 | 1.0000 | $(1.0000-0.553371)/1.175578=0.379923$ |
| parch | 0.379213 | 0.791113 | 0.0000 | $(0.0000-0.379213)/0.791113=-0.479342$ |
| fare | 32.586276 | 51.933021 | 7.2500 | $(7.2500-32.586276)/51.933021=-0.487864$ |

Standardization changes the measurement scale but does not change the fitted predictions when coefficients are estimated consistently.

---

## 10. Design matrix construction

One prepared passenger row is

$$
\mathbf{x}_i=[1,Age_i^*,SibSp_i^*,Parch_i^*,Fare_i^*,I(C2),I(C3),I(Male),I(Q),I(S)]
$$

The leading 1 multiplies the intercept. Stacking all training rows gives matrix

$$X_{{712\times10}}$$

The target vector is

$$\mathbf{y}=[y_1,y_2,\ldots,y_{{712}}]^T$$

The coefficient vector is

$$\mathbf{b}=[b_0,b_1,\ldots,b_9]^T$$

---

## 11. The linear regression equation

For one passenger:

$$\hat{y}_i=\mathbf{x}_i^T\mathbf{b}$$

For all training passengers simultaneously:

$$\hat{\mathbf{y}}=X\mathbf{b}$$

The model chooses the coefficient vector that makes fitted values as close as possible to the observed zeros and ones according to squared error.

---

## 12. Residuals and squared errors

The residual is actual minus fitted:

$$e_i=y_i-\hat{y}_i$$

If a survivor has fitted score 0.80:

$$e=1-0.80=0.20$$
$$e^2=0.20^2=0.04$$

If a non-survivor has fitted score 0.30:

$$e=0-0.30=-0.30$$
$$e^2=(-0.30)^2=0.09$$

Squaring prevents positive and negative errors from cancelling and penalizes large errors more strongly.

---

## 13. Sum of squared errors

The total fitting error is

$$SSE=\sum_{{i=1}}^n(y_i-\hat{y}_i)^2$$

In matrix form:

$$SSE=(\mathbf{y}-X\mathbf{b})^T(\mathbf{y}-X\mathbf{b})$$

Ordinary Least Squares selects coefficients that minimize this expression.

---

## 14. Mean squared error cost function

A common optimization form is

$$J(\mathbf{b})=\frac{{1}}{{2n}}\sum_{{i=1}}^n(y_i-\hat{y}_i)^2$$

The factor 2 is included only to simplify differentiation. Minimizing SSE, MSE, or this cost produces the same coefficient estimates because each differs only by a positive constant.

---

## 15. Ordinary Least Squares principle

The least-squares objective is

$$\min_{{\mathbf{b}}} \ (\mathbf{y}-X\mathbf{b})^T(\mathbf{y}-X\mathbf{b})$$

At the minimum, the slope in every coefficient direction equals zero. Geometrically, the residual vector is perpendicular to every column of the design matrix:

$$X^T(\mathbf{y}-X\mathbf{b})=\mathbf{0}$$

---

## 16. Normal equation derivation

Start with

$$SSE=(\mathbf{y}-X\mathbf{b})^T(\mathbf{y}-X\mathbf{b})$$

Expand:

$$SSE=\mathbf{y}^T\mathbf{y}-2\mathbf{b}^TX^T\mathbf{y}+\mathbf{b}^TX^T X\mathbf{b}$$

Differentiate with respect to the coefficient vector:

$$\frac{{\partial SSE}}{{\partial\mathbf{b}}}=-2X^T\mathbf{y}+2X^T X\mathbf{b}$$

Set the derivative to zero:

$$X^T X\mathbf{b}=X^T\mathbf{y}$$

Solve:

$$\hat{\mathbf{b}}=(X^T X)^{-1}X^T\mathbf{y}$$

If the inverse is unstable or unavailable, the Moore-Penrose pseudoinverse gives the least-squares solution.

---

## 17. Matrix calculation of coefficients

The complete matrix arithmetic is performed in this order:

1. Build the 712 by 10 matrix X.
2. Transpose X to obtain a 10 by 712 matrix.
3. Multiply X transpose by X to obtain a 10 by 10 cross-product matrix.
4. Multiply X transpose by y to obtain a 10 by 1 vector.
5. Invert or pseudoinvert the cross-product matrix.
6. Multiply the inverse by the target cross-product vector.

The resulting coefficient vector is:

| Parameter | Normal-equation result | Fitted-model result | Difference |
|---|---:|---:|---:|
| Intercept | 0.9495708347 | 0.9495708347 | 2.66e-15 |
| age | -0.0591159908 | -0.0591159908 | 5.90e-16 |
| sibsp | -0.0447783629 | -0.0447783629 | 3.40e-16 |
| parch | -0.0170016689 | -0.0170016689 | 1.28e-16 |
| fare | 0.0264265602 | 0.0264265602 | 1.73e-17 |
| Class=2 | -0.0856896991 | -0.0856896991 | 2.05e-15 |
| Class=3 | -0.2854455173 | -0.2854455173 | 1.89e-15 |
| Sex=male | -0.5112674730 | -0.5112674730 | 4.44e-16 |
| Embarked=Q | -0.0241637181 | -0.0241637181 | 7.28e-15 |
| Embarked=S | -0.0784368898 | -0.0784368898 | 3.16e-15 |

The near-zero differences confirm that the fitted model and direct Ordinary Least Squares matrix calculation produce the same solution.

---

## 18. Gradient descent alternative

Gradient descent reaches the same least-squares minimum through repeated updates. For the cost

$$J(\mathbf{b})=\frac1{2n}\|\mathbf{y}-X\mathbf{b}\|^2$$

the gradient is

$$\nabla J(\mathbf{b})=-\frac1nX^T(\mathbf{y}-X\mathbf{b})$$

Update rule:

$$\mathbf{b}_{\text{new}}=\mathbf{b}_{\text{old}}-\alpha\nabla J(\mathbf{b}_{\text{old}})$$

Starting with all coefficients equal to zero gives these initial gradients:

| Parameter | Initial gradient |
|---|---:|
| Intercept | -0.37640449 |
| age | 0.02353999 |
| sibsp | 0.02306226 |
| parch | -0.03794040 |
| fare | -0.11949337 |
| Class=2 | -0.10252809 |
| Class=3 | -0.13483146 |
| Sex=male | -0.12219101 |
| Embarked=Q | -0.03089888 |
| Embarked=S | -0.25000000 |

The learning rate alpha controls step size. Repeated recalculation continues until the gradient becomes approximately zero.

---

## 19. Final fitted equation

The fitted score equation is

$$\hat{y}=0.9495708347 -0.0591159908\,Age^* -0.0447783629\,SibSp^* -0.0170016689\,Parch^* +0.0264265602\,Fare^* -0.0856896991\,I(C2) -0.2854455173\,I(C3) -0.5112674730\,I(Male) -0.0241637181\,I(Q) -0.0784368898\,I(S)$$

| Term | Coefficient | Meaning of a one-unit prepared increase |
|---|---:|---:|
| Intercept | 0.9495708347 | Baseline fitted score |
| age | -0.0591159908 | Score changes by -0.0591159908 |
| sibsp | -0.0447783629 | Score changes by -0.0447783629 |
| parch | -0.0170016689 | Score changes by -0.0170016689 |
| fare | 0.0264265602 | Score changes by +0.0264265602 |
| Class=2 | -0.0856896991 | Score changes by -0.0856896991 |
| Class=3 | -0.2854455173 | Score changes by -0.2854455173 |
| Sex=male | -0.5112674730 | Score changes by -0.5112674730 |
| Embarked=Q | -0.0241637181 | Score changes by -0.0241637181 |
| Embarked=S | -0.0784368898 | Score changes by -0.0784368898 |

All interpretations hold the other included predictors constant.

---

## 20. Complete prediction recipe

1. Replace missing age with the training median.
2. Replace missing embarkation with the training mode.
3. Standardize age, sibsp, parch, and fare with training means and scales.
4. Create the class, sex, and embarkation indicator variables.
5. Multiply each prepared value by its fitted coefficient.
6. Add all contributions and the intercept.
7. The result is the raw linear survival score.
8. For classification, assign survived when the score is at least 0.50.
9. Compare with the actual target only during evaluation.
10. Keep unrounded values throughout and round only the displayed result.

---

## 21. Detailed passenger calculations

The following passengers show every additive contribution. Final results use unrounded arithmetic.

### Passenger 710

Raw values: class 3, sex male, age missing, sibsp 1, parch 1, fare 15.2458, embarked C.

$$\hat{y}=0.949571+0.005476-0.017012-0.013341-0.008824-0.000000-0.285446-0.511267-0.000000-0.000000=0.119157$$

Threshold decision: **0 (did not survive)**. Actual: **1**. Classification: **incorrect**. Residual: $e=1-(0.119157)=0.880843$. Squared error: **0.775885**.

### Passenger 440

Raw values: class 2, sex male, age 31.00, sibsp 0, parch 0, fare 10.5000, embarked S.

$$\hat{y}=0.949571-0.008167+0.021078+0.008150-0.011239-0.085690-0.000000-0.511267-0.000000-0.078437=0.283999$$

Threshold decision: **0 (did not survive)**. Actual: **0**. Classification: **correct**. Residual: $e=0-(0.283999)=-0.283999$. Squared error: **0.080655**.

### Passenger 841

Raw values: class 3, sex male, age 20.00, sibsp 0, parch 0, fare 7.9250, embarked S.

$$\hat{y}=0.949571+0.041858+0.021078+0.008150-0.012549-0.000000-0.285446-0.511267-0.000000-0.078437=0.132958$$

Threshold decision: **0 (did not survive)**. Actual: **0**. Classification: **correct**. Residual: $e=0-(0.132958)=-0.132958$. Squared error: **0.017678**.

### Passenger 721

Raw values: class 2, sex female, age 6.00, sibsp 0, parch 1, fare 33.0000, embarked S.

$$\hat{y}=0.949571+0.105528+0.021078-0.013341+0.000211-0.085690-0.000000-0.000000-0.000000-0.078437=0.898919$$

Threshold decision: **1 (survived)**. Actual: **1**. Classification: **correct**. Residual: $e=1-(0.898919)=0.101081$. Squared error: **0.010217**.

### Passenger 40

Raw values: class 3, sex female, age 14.00, sibsp 1, parch 0, fare 11.2417, embarked C.

$$\hat{y}=0.949571+0.069145-0.017012+0.008150-0.010861-0.000000-0.285446-0.000000-0.000000-0.000000=0.713546$$

Threshold decision: **1 (survived)**. Actual: **1**. Classification: **correct**. Residual: $e=1-(0.713546)=0.286454$. Squared error: **0.082056**.

### Passenger 291

Raw values: class 1, sex female, age 26.00, sibsp 0, parch 0, fare 78.8500, embarked S.

$$\hat{y}=0.949571+0.014572+0.021078+0.008150+0.023542-0.000000-0.000000-0.000000-0.000000-0.078437=0.938475$$

Threshold decision: **1 (survived)**. Actual: **1**. Classification: **correct**. Residual: $e=1-(0.938475)=0.061525$. Squared error: **0.003785**.

### Passenger 301

Raw values: class 3, sex female, age missing, sibsp 0, parch 0, fare 7.7500, embarked Q.

$$\hat{y}=0.949571+0.005476+0.021078+0.008150-0.012638-0.000000-0.285446-0.000000-0.024164-0.000000=0.662027$$

Threshold decision: **1 (survived)**. Actual: **1**. Classification: **correct**. Residual: $e=1-(0.662027)=0.337973$. Squared error: **0.114226**.

### Passenger 334

Raw values: class 3, sex male, age 16.00, sibsp 2, parch 0, fare 18.0000, embarked S.

$$\hat{y}=0.949571+0.060050-0.055103+0.008150-0.007422-0.000000-0.285446-0.511267-0.000000-0.078437=0.080095$$

Threshold decision: **0 (did not survive)**. Actual: **0**. Classification: **correct**. Residual: $e=0-(0.080095)=-0.080095$. Squared error: **0.006415**.

### Passenger 209

Raw values: class 3, sex female, age 16.00, sibsp 0, parch 0, fare 7.7500, embarked Q.

$$\hat{y}=0.949571+0.060050+0.021078+0.008150-0.012638-0.000000-0.285446-0.000000-0.024164-0.000000=0.716601$$

Threshold decision: **1 (survived)**. Actual: **1**. Classification: **correct**. Residual: $e=1-(0.716601)=0.283399$. Squared error: **0.080315**.

### Passenger 137

Raw values: class 1, sex female, age 19.00, sibsp 0, parch 2, fare 26.2833, embarked S.

$$\hat{y}=0.949571+0.046406+0.021078-0.034832-0.003207-0.000000-0.000000-0.000000-0.000000-0.078437=0.900579$$

Threshold decision: **1 (survived)**. Actual: **1**. Classification: **correct**. Residual: $e=1-(0.900579)=0.099421$. Squared error: **0.009885**.

---

## 22. All 179 test predictions

`NA` means the raw age was missing and the training median was inserted before standardization.

| Passenger | Class | Sex | Age | Fare | Actual | Linear score | Class | Residual | Squared error | Correct? |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 6 | 3 | male | NA | 8.4583 | 0 | 0.151120 | 0 | -0.151120 | 0.022837 | Yes |
| 11 | 3 | female | 4.00 | 16.7000 | 1 | 0.661874 | 1 | 0.338126 | 0.114329 | Yes |
| 24 | 1 | male | 28.00 | 35.5000 | 1 | 0.396053 | 0 | 0.603947 | 0.364752 | No |
| 26 | 3 | female | 38.00 | 31.3875 | 1 | 0.428760 | 0 | 0.571240 | 0.326315 | No |
| 31 | 1 | male | 40.00 | 27.7208 | 0 | 0.415958 | 0 | -0.415958 | 0.173021 | Yes |
| 32 | 1 | female | NA | 146.5208 | 1 | 1.004161 | 1 | -0.004161 | 0.000017 | Yes |
| 34 | 2 | male | 66.00 | 10.5000 | 0 | 0.124826 | 0 | -0.124826 | 0.015581 | Yes |
| 40 | 3 | female | 14.00 | 11.2417 | 1 | 0.713546 | 1 | 0.286454 | 0.082056 | Yes |
| 45 | 3 | female | 19.00 | 7.8792 | 1 | 0.703023 | 1 | 0.296977 | 0.088195 | Yes |
| 50 | 3 | female | 18.00 | 17.8000 | 0 | 0.620256 | 1 | -0.620256 | 0.384717 | No |
| 55 | 1 | male | 65.00 | 61.9792 | 0 | 0.298205 | 0 | -0.298205 | 0.088926 | Yes |
| 64 | 3 | male | 4.00 | 27.9000 | 0 | 0.058634 | 0 | -0.058634 | 0.003438 | Yes |
| 66 | 3 | male | NA | 15.2458 | 1 | 0.119157 | 0 | 0.880843 | 0.775885 | No |
| 67 | 2 | female | 29.00 | 10.5000 | 1 | 0.804362 | 1 | 0.195638 | 0.038274 | Yes |
| 68 | 3 | male | 19.00 | 8.1583 | 0 | 0.137625 | 0 | -0.137625 | 0.018941 | Yes |
| 71 | 2 | male | 32.00 | 10.5000 | 0 | 0.279451 | 0 | -0.279451 | 0.078093 | Yes |
| 73 | 2 | male | 21.00 | 73.5000 | 0 | 0.361535 | 0 | -0.361535 | 0.130707 | Yes |
| 77 | 3 | male | NA | 7.8958 | 0 | 0.096561 | 0 | -0.096561 | 0.009324 | Yes |
| 78 | 3 | male | NA | 8.0500 | 0 | 0.096639 | 0 | -0.096639 | 0.009339 | Yes |
| 79 | 2 | male | 0.83 | 29.0000 | 1 | 0.387638 | 0 | 0.612362 | 0.374988 | No |
| 83 | 3 | female | NA | 7.7875 | 1 | 0.662046 | 1 | 0.337954 | 0.114213 | Yes |
| 85 | 2 | female | 17.00 | 10.5000 | 1 | 0.858935 | 1 | 0.141065 | 0.019899 | Yes |
| 87 | 3 | male | 16.00 | 34.3750 | 0 | 0.062046 | 0 | -0.062046 | 0.003850 | Yes |
| 97 | 1 | male | 71.00 | 34.6542 | 0 | 0.278504 | 0 | -0.278504 | 0.077565 | Yes |
| 98 | 1 | male | 23.00 | 63.3583 | 1 | 0.489914 | 0 | 0.510086 | 0.260188 | No |
| 110 | 3 | female | NA | 24.1500 | 1 | 0.632282 | 1 | 0.367718 | 0.135216 | Yes |
| 111 | 1 | male | 47.00 | 52.0000 | 0 | 0.318041 | 0 | -0.318041 | 0.101150 | Yes |
| 121 | 2 | male | 21.00 | 73.5000 | 0 | 0.285353 | 0 | -0.285353 | 0.081427 | Yes |
| 137 | 1 | female | 19.00 | 26.2833 | 1 | 0.900579 | 1 | 0.099421 | 0.009885 | Yes |
| 138 | 1 | male | 37.00 | 53.1000 | 0 | 0.325988 | 0 | -0.325988 | 0.106268 | Yes |
| 140 | 1 | male | 24.00 | 79.2000 | 0 | 0.514918 | 1 | -0.514918 | 0.265141 | No |
| 142 | 3 | female | 22.00 | 7.7500 | 1 | 0.635041 | 1 | 0.364959 | 0.133195 | Yes |
| 166 | 3 | male | 9.00 | 20.5250 | 1 | 0.146414 | 0 | 0.853586 | 0.728609 | No |
| 169 | 1 | male | NA | 25.9250 | 0 | 0.391181 | 0 | -0.391181 | 0.153022 | Yes |
| 175 | 1 | male | 56.00 | 30.6958 | 0 | 0.344707 | 0 | -0.344707 | 0.118823 | Yes |
| 193 | 3 | female | 19.00 | 7.8542 | 1 | 0.610647 | 1 | 0.389353 | 0.151596 | Yes |
| 197 | 3 | male | NA | 7.7500 | 0 | 0.150760 | 0 | -0.150760 | 0.022729 | Yes |
| 199 | 3 | female | NA | 7.7500 | 1 | 0.662027 | 1 | 0.337973 | 0.114226 | Yes |
| 205 | 3 | male | 18.00 | 8.0500 | 1 | 0.142117 | 0 | 0.857883 | 0.735963 | No |
| 209 | 3 | female | 16.00 | 7.7500 | 1 | 0.716601 | 1 | 0.283399 | 0.080315 | Yes |
| 210 | 1 | male | 40.00 | 31.0000 | 1 | 0.417627 | 0 | 0.582373 | 0.339159 | No |
| 211 | 3 | male | 24.00 | 7.0500 | 0 | 0.114322 | 0 | -0.114322 | 0.013069 | Yes |
| 212 | 2 | female | 35.00 | 21.0000 | 1 | 0.782418 | 1 | 0.217582 | 0.047342 | Yes |
| 214 | 2 | male | 30.00 | 13.0000 | 0 | 0.289818 | 0 | -0.289818 | 0.083995 | Yes |
| 216 | 1 | female | 31.00 | 113.2750 | 1 | 0.973600 | 1 | 0.026400 | 0.000697 | Yes |
| 219 | 1 | female | 32.00 | 76.2917 | 1 | 0.988323 | 1 | 0.011677 | 0.000136 | Yes |
| 228 | 3 | male | 20.50 | 7.2500 | 0 | 0.130341 | 0 | -0.130341 | 0.016989 | Yes |
| 232 | 3 | male | 29.00 | 7.7750 | 0 | 0.091952 | 0 | -0.091952 | 0.008455 | Yes |
| 236 | 3 | female | NA | 7.5500 | 0 | 0.607652 | 1 | -0.607652 | 0.369241 | No |
| 245 | 3 | male | 30.00 | 7.2250 | 0 | 0.165561 | 0 | -0.165561 | 0.027410 | Yes |
| 251 | 3 | male | NA | 7.2500 | 0 | 0.096232 | 0 | -0.096232 | 0.009261 | Yes |
| 255 | 3 | female | 41.00 | 20.2125 | 0 | 0.511993 | 1 | -0.511993 | 0.262137 | No |
| 267 | 3 | male | 16.00 | 39.6875 | 0 | -0.006541 | 0 | 0.006541 | 0.000043 | Yes |
| 276 | 1 | female | 63.00 | 77.9583 | 1 | 0.731663 | 1 | 0.268337 | 0.072005 | Yes |
| 281 | 3 | male | 65.00 | 7.7500 | 0 | -0.017508 | 0 | 0.017508 | 0.000307 | Yes |
| 282 | 3 | male | 28.00 | 7.8542 | 0 | 0.096540 | 0 | -0.096540 | 0.009320 | Yes |
| 287 | 3 | male | 30.00 | 9.5000 | 1 | 0.088282 | 0 | 0.911718 | 0.831230 | No |
| 291 | 1 | female | 26.00 | 78.8500 | 1 | 0.938475 | 1 | 0.061525 | 0.003785 | Yes |
| 292 | 1 | female | 19.00 | 91.0792 | 1 | 1.016879 | 1 | -0.016879 | 0.000285 | Yes |
| 293 | 2 | male | 36.00 | 12.8750 | 0 | 0.340905 | 0 | -0.340905 | 0.116216 | Yes |
| 295 | 3 | male | 24.00 | 7.8958 | 0 | 0.114752 | 0 | -0.114752 | 0.013168 | Yes |
| 297 | 3 | male | 23.50 | 7.2292 | 0 | 0.195124 | 0 | -0.195124 | 0.038073 | Yes |
| 300 | 1 | female | 50.00 | 247.5208 | 1 | 0.972104 | 1 | 0.027896 | 0.000778 | Yes |
| 301 | 3 | female | NA | 7.7500 | 1 | 0.662027 | 1 | 0.337973 | 0.114226 | Yes |
| 307 | 1 | female | NA | 110.8833 | 1 | 1.024117 | 1 | -0.024117 | 0.000582 | Yes |
| 310 | 1 | female | 30.00 | 56.9292 | 1 | 0.987566 | 1 | 0.012434 | 0.000155 | Yes |
| 312 | 1 | female | 18.00 | 262.3750 | 1 | 1.027520 | 1 | -0.027520 | 0.000757 | Yes |
| 313 | 2 | female | 26.00 | 26.0000 | 0 | 0.766311 | 1 | -0.766311 | 0.587232 | No |
| 320 | 1 | female | 40.00 | 134.5000 | 1 | 0.921980 | 1 | 0.078020 | 0.006087 | Yes |
| 322 | 3 | male | 27.00 | 7.8958 | 0 | 0.101109 | 0 | -0.101109 | 0.010223 | Yes |
| 324 | 2 | female | 22.00 | 29.0000 | 1 | 0.786029 | 1 | 0.213971 | 0.045784 | Yes |
| 328 | 2 | female | 36.00 | 13.0000 | 1 | 0.773799 | 1 | 0.226201 | 0.051167 | Yes |
| 334 | 3 | male | 16.00 | 18.0000 | 0 | 0.080095 | 0 | -0.080095 | 0.006415 | Yes |
| 339 | 3 | male | 45.00 | 8.0500 | 1 | 0.019327 | 0 | 0.980673 | 0.961720 | No |
| 343 | 2 | male | 28.00 | 13.0000 | 0 | 0.298914 | 0 | -0.298914 | 0.089350 | Yes |
| 345 | 2 | male | 36.00 | 13.0000 | 0 | 0.262532 | 0 | -0.262532 | 0.068923 | Yes |
| 351 | 3 | male | 23.00 | 9.2250 | 0 | 0.119976 | 0 | -0.119976 | 0.014394 | Yes |
| 358 | 2 | female | 38.00 | 13.0000 | 0 | 0.764704 | 1 | -0.764704 | 0.584772 | No |
| 360 | 3 | female | NA | 7.8792 | 1 | 0.662093 | 1 | 0.337907 | 0.114181 | Yes |
| 361 | 3 | male | 40.00 | 27.9000 | 0 | -0.071887 | 0 | 0.071887 | 0.005168 | Yes |
| 363 | 3 | female | 45.00 | 14.4542 | 0 | 0.590799 | 1 | -0.590799 | 0.349044 | No |
| 364 | 3 | male | 35.00 | 7.0500 | 0 | 0.064296 | 0 | -0.064296 | 0.004134 | Yes |
| 376 | 1 | female | NA | 82.1708 | 1 | 0.971416 | 1 | 0.028584 | 0.000817 | Yes |
| 389 | 3 | male | NA | 7.7292 | 0 | 0.150749 | 0 | -0.150749 | 0.022725 | Yes |
| 397 | 3 | female | 31.00 | 7.8542 | 0 | 0.594164 | 1 | -0.594164 | 0.353031 | No |
| 399 | 2 | male | 23.00 | 10.5000 | 0 | 0.320381 | 0 | -0.320381 | 0.102644 | Yes |
| 409 | 3 | male | 21.00 | 7.7750 | 0 | 0.128334 | 0 | -0.128334 | 0.016470 | Yes |
| 417 | 2 | female | 34.00 | 32.5000 | 1 | 0.733236 | 1 | 0.266764 | 0.071163 | Yes |
| 418 | 2 | female | 18.00 | 13.0000 | 1 | 0.812678 | 1 | 0.187322 | 0.035090 | Yes |
| 423 | 3 | male | 29.00 | 7.8750 | 0 | 0.092003 | 0 | -0.092003 | 0.008464 | Yes |
| 427 | 2 | female | 28.00 | 26.0000 | 1 | 0.778706 | 1 | 0.221294 | 0.048971 | Yes |
| 434 | 3 | male | 17.00 | 7.1250 | 0 | 0.146194 | 0 | -0.146194 | 0.021373 | Yes |
| 440 | 2 | male | 31.00 | 10.5000 | 0 | 0.283999 | 0 | -0.283999 | 0.080655 | Yes |
| 443 | 3 | male | 25.00 | 7.7750 | 0 | 0.072052 | 0 | -0.072052 | 0.005192 | Yes |
| 447 | 2 | female | 13.00 | 19.5000 | 1 | 0.860215 | 1 | 0.139785 | 0.019540 | Yes |
| 448 | 1 | male | 34.00 | 26.5500 | 1 | 0.364212 | 0 | 0.635788 | 0.404226 | No |
| 449 | 3 | female | 5.00 | 19.2583 | 1 | 0.698975 | 1 | 0.301025 | 0.090616 | Yes |
| 458 | 1 | female | NA | 51.8625 | 1 | 0.877556 | 1 | 0.122444 | 0.014992 | Yes |
| 463 | 1 | male | 47.00 | 38.5000 | 0 | 0.311172 | 0 | -0.311172 | 0.096828 | Yes |
| 466 | 3 | male | 38.00 | 7.0500 | 0 | 0.050653 | 0 | -0.050653 | 0.002566 | Yes |
| 478 | 3 | male | 29.00 | 7.0458 | 0 | 0.053490 | 0 | -0.053490 | 0.002861 | Yes |
| 482 | 2 | male | NA | 0.0000 | 0 | 0.292299 | 0 | -0.292299 | 0.085439 | Yes |
| 486 | 3 | female | NA | 25.4667 | 0 | 0.481007 | 0 | -0.481007 | 0.231368 | Yes |
| 494 | 1 | male | 71.00 | 49.5042 | 0 | 0.286061 | 0 | -0.286061 | 0.081831 | Yes |
| 495 | 3 | male | 21.00 | 8.0500 | 0 | 0.128474 | 0 | -0.128474 | 0.016506 | Yes |
| 508 | 1 | male | NA | 26.5500 | 1 | 0.391499 | 0 | 0.608501 | 0.370274 | No |
| 513 | 1 | male | 36.00 | 26.2875 | 1 | 0.354983 | 0 | 0.645017 | 0.416047 | No |
| 515 | 3 | male | 24.00 | 7.4958 | 0 | 0.114549 | 0 | -0.114549 | 0.013121 | Yes |
| 519 | 2 | female | 36.00 | 26.0000 | 1 | 0.742324 | 1 | 0.257676 | 0.066397 | Yes |
| 522 | 3 | male | 22.00 | 7.8958 | 0 | 0.123848 | 0 | -0.123848 | 0.015338 | Yes |
| 528 | 1 | male | NA | 221.7792 | 0 | 0.490843 | 0 | -0.490843 | 0.240927 | Yes |
| 530 | 2 | male | 23.00 | 11.5000 | 0 | 0.223218 | 0 | -0.223218 | 0.049826 | Yes |
| 533 | 3 | male | 17.00 | 7.2292 | 0 | 0.165103 | 0 | -0.165103 | 0.027259 | Yes |
| 539 | 3 | male | NA | 14.5000 | 0 | 0.099922 | 0 | -0.099922 | 0.009984 | Yes |
| 540 | 1 | female | 22.00 | 49.5000 | 1 | 0.977187 | 1 | 0.022813 | 0.000520 | Yes |
| 542 | 3 | female | 9.00 | 31.2750 | 0 | 0.510789 | 1 | -0.510789 | 0.260906 | No |
| 546 | 1 | male | 64.00 | 26.0000 | 0 | 0.227498 | 0 | -0.227498 | 0.051756 | Yes |
| 573 | 1 | male | 36.00 | 26.3875 | 1 | 0.355034 | 0 | 0.644966 | 0.415981 | No |
| 579 | 3 | female | NA | 14.4583 | 0 | 0.651514 | 1 | -0.651514 | 0.424471 | No |
| 585 | 3 | male | NA | 8.7125 | 0 | 0.175413 | 0 | -0.175413 | 0.030770 | Yes |
| 586 | 1 | female | 18.00 | 79.6500 | 1 | 0.932283 | 1 | 0.067717 | 0.004586 | Yes |
| 591 | 3 | male | 35.00 | 7.1250 | 0 | 0.064334 | 0 | -0.064334 | 0.004139 | Yes |
| 592 | 1 | female | 52.00 | 78.2667 | 1 | 0.860282 | 1 | 0.139718 | 0.019521 | Yes |
| 595 | 2 | male | 37.00 | 26.0000 | 0 | 0.226509 | 0 | -0.226509 | 0.051306 | Yes |
| 600 | 1 | male | 49.00 | 56.9292 | 1 | 0.351800 | 0 | 0.648200 | 0.420163 | No |
| 605 | 1 | male | 35.00 | 26.5500 | 1 | 0.438101 | 0 | 0.561899 | 0.315730 | No |
| 606 | 3 | male | 36.00 | 15.5500 | 0 | 0.025983 | 0 | -0.025983 | 0.000675 | Yes |
| 612 | 3 | male | NA | 7.0500 | 0 | 0.096131 | 0 | -0.096131 | 0.009241 | Yes |
| 621 | 3 | male | 27.00 | 14.4542 | 0 | 0.144792 | 0 | -0.144792 | 0.020965 | Yes |
| 622 | 1 | male | 42.00 | 52.5542 | 1 | 0.302972 | 0 | 0.697028 | 0.485849 | No |
| 626 | 1 | male | 61.00 | 32.3208 | 0 | 0.244358 | 0 | -0.244358 | 0.059711 | Yes |
| 644 | 3 | male | NA | 56.4958 | 1 | 0.121291 | 0 | 0.878709 | 0.772129 | No |
| 653 | 3 | male | 21.00 | 8.4333 | 0 | 0.128669 | 0 | -0.128669 | 0.016556 | Yes |
| 654 | 3 | female | NA | 7.8292 | 1 | 0.662068 | 1 | 0.337932 | 0.114198 | Yes |
| 658 | 3 | female | 32.00 | 15.5000 | 0 | 0.588199 | 1 | -0.588199 | 0.345977 | No |
| 660 | 1 | male | 58.00 | 113.2750 | 0 | 0.334651 | 0 | -0.334651 | 0.111991 | Yes |
| 669 | 3 | male | 43.00 | 8.0500 | 0 | 0.028422 | 0 | -0.028422 | 0.000808 | Yes |
| 670 | 1 | female | NA | 52.0000 | 1 | 0.877626 | 1 | 0.122374 | 0.014975 | Yes |
| 671 | 2 | female | 40.00 | 39.0000 | 1 | 0.709257 | 1 | 0.290743 | 0.084531 | Yes |
| 674 | 2 | male | 31.00 | 13.0000 | 1 | 0.285271 | 0 | 0.714729 | 0.510838 | No |
| 683 | 3 | male | 20.00 | 9.2250 | 0 | 0.133620 | 0 | -0.133620 | 0.017854 | Yes |
| 689 | 3 | male | 18.00 | 7.7958 | 0 | 0.141988 | 0 | -0.141988 | 0.020161 | Yes |
| 697 | 3 | male | 44.00 | 8.0500 | 0 | 0.023875 | 0 | -0.023875 | 0.000570 | Yes |
| 710 | 3 | male | NA | 15.2458 | 1 | 0.119157 | 0 | 0.880843 | 0.775885 | No |
| 713 | 1 | male | 48.00 | 52.0000 | 1 | 0.275403 | 0 | 0.724597 | 0.525041 | No |
| 721 | 2 | female | 6.00 | 33.0000 | 1 | 0.898919 | 1 | 0.101081 | 0.010217 | Yes |
| 737 | 3 | female | 48.00 | 34.3750 | 0 | 0.427784 | 0 | -0.427784 | 0.182999 | Yes |
| 740 | 3 | male | NA | 7.8958 | 0 | 0.096561 | 0 | -0.096561 | 0.009324 | Yes |
| 768 | 3 | female | 30.50 | 7.7500 | 0 | 0.650658 | 1 | -0.650658 | 0.423356 | No |
| 773 | 2 | female | 57.00 | 10.5000 | 0 | 0.677023 | 1 | -0.677023 | 0.458361 | No |
| 774 | 3 | male | NA | 7.2250 | 0 | 0.174656 | 0 | -0.174656 | 0.030505 | Yes |
| 778 | 3 | female | 5.00 | 12.4750 | 1 | 0.714758 | 1 | 0.285242 | 0.081363 | Yes |
| 779 | 3 | male | NA | 7.7375 | 0 | 0.150754 | 0 | -0.150754 | 0.022727 | Yes |
| 786 | 3 | male | 25.00 | 7.2500 | 0 | 0.109876 | 0 | -0.109876 | 0.012073 | Yes |
| 791 | 3 | male | NA | 7.7500 | 0 | 0.150760 | 0 | -0.150760 | 0.022729 | Yes |
| 794 | 1 | male | NA | 30.6958 | 0 | 0.472045 | 0 | -0.472045 | 0.222827 | Yes |
| 797 | 1 | female | 49.00 | 25.9292 | 1 | 0.806947 | 1 | 0.193053 | 0.037270 | Yes |
| 803 | 1 | male | 11.00 | 120.0000 | 1 | 0.435292 | 0 | 0.564708 | 0.318895 | No |
| 809 | 2 | male | 39.00 | 13.0000 | 0 | 0.248888 | 0 | -0.248888 | 0.061945 | Yes |
| 815 | 3 | male | 30.50 | 8.0500 | 0 | 0.085270 | 0 | -0.085270 | 0.007271 | Yes |
| 820 | 3 | male | 10.00 | 27.9000 | 0 | 0.031347 | 0 | -0.031347 | 0.000983 | Yes |
| 821 | 1 | female | 52.00 | 93.5000 | 1 | 0.768106 | 1 | 0.231894 | 0.053775 | Yes |
| 823 | 1 | male | 38.00 | 0.0000 | 0 | 0.332511 | 0 | -0.332511 | 0.110563 | Yes |
| 831 | 3 | female | 15.00 | 14.4542 | 1 | 0.710633 | 1 | 0.289367 | 0.083733 | Yes |
| 834 | 3 | male | 23.00 | 7.8542 | 0 | 0.119279 | 0 | -0.119279 | 0.014227 | Yes |
| 836 | 1 | female | 39.00 | 83.1583 | 1 | 0.900402 | 1 | 0.099598 | 0.009920 | Yes |
| 838 | 3 | male | NA | 8.0500 | 0 | 0.096639 | 0 | -0.096639 | 0.009339 | Yes |
| 841 | 3 | male | 20.00 | 7.9250 | 0 | 0.132958 | 0 | -0.132958 | 0.017678 | Yes |
| 843 | 1 | female | 30.00 | 31.0000 | 1 | 0.974372 | 1 | 0.025628 | 0.000657 | Yes |
| 846 | 3 | male | 42.00 | 7.5500 | 0 | 0.032716 | 0 | -0.032716 | 0.001070 | Yes |
| 850 | 1 | female | NA | 89.1042 | 1 | 0.974944 | 1 | 0.025056 | 0.000628 | Yes |
| 853 | 3 | female | 9.00 | 15.2458 | 0 | 0.716832 | 1 | -0.716832 | 0.513848 | No |
| 854 | 1 | female | 16.00 | 39.4000 | 1 | 0.942388 | 1 | 0.057612 | 0.003319 | Yes |
| 855 | 2 | female | 44.00 | 26.0000 | 0 | 0.705941 | 1 | -0.705941 | 0.498353 | No |
| 877 | 3 | male | 20.00 | 9.8458 | 0 | 0.133936 | 0 | -0.133936 | 0.017939 | Yes |
| 878 | 3 | male | 19.00 | 7.8958 | 0 | 0.137491 | 0 | -0.137491 | 0.018904 | Yes |
| 886 | 3 | female | 39.00 | 29.1250 | 0 | 0.515424 | 1 | -0.515424 | 0.265662 | No |
| 887 | 2 | male | 27.00 | 13.0000 | 0 | 0.303462 | 0 | -0.303462 | 0.092089 | Yes |
| 890 | 1 | male | 26.00 | 30.0000 | 1 | 0.480787 | 0 | 0.519213 | 0.269582 | No |

---

## 23. Converting scores to classes

The raw fitted score is converted only for the survived/not-survived comparison:

$$
\hat{c}=1 \quad \text{when} \quad \hat{y} \ge 0.50
$$

$$
\hat{c}=0 \quad \text{when} \quad \hat{y} < 0.50
$$

Examples:

| Linear score | Threshold comparison | Class |
|---:|---|---:|
| -0.08 | -0.08 < 0.50 | 0 |
| 0.42 | 0.42 < 0.50 | 0 |
| 0.50 | 0.50 >= 0.50 | 1 |
| 0.83 | 0.83 >= 0.50 | 1 |
| 1.06 | 1.06 >= 0.50 | 1 |

The class conversion does not alter or retrain the regression equation.

---

## 24. Confusion matrix calculations

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | TN = 89 | FP = 16 |
| Actual 1 | FN = 23 | TP = 51 |

Total check:

$$89+16+23+51=179=179$$

Correct classifications:

$$TN+TP=89+51=140$$

Incorrect classifications:

$$FP+FN=16+23=39$$

---

## 25. Classification evaluation metrics

### Accuracy

$$\frac{TP+TN}{N}=\frac{51+89}{179}=0.782123=78.21\%$$

### Precision

$$\frac{TP}{TP+FP}=\frac{51}{51+16}=0.761194$$

### Recall

$$\frac{TP}{TP+FN}=\frac{51}{51+23}=0.689189$$

### F1 score

$$F1=\frac{2\times Precision\times Recall}{Precision+Recall}=0.723404$$

---

## 26. MAE, MSE, RMSE, and R-squared

Regression metrics use the raw scores, not the thresholded classes.

### Mean Absolute Error

$$MAE=\frac1n\sum|y_i-\hat{y}_i|=0.286623$$

### Mean Squared Error

$$MSE=\frac1n\sum(y_i-\hat{y}_i)^2=0.137346$$

### Root Mean Squared Error

$$RMSE=\sqrt{MSE}=\sqrt{0.137346}=0.370602$$

### R-squared

$$R^2=1-\frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}=0.433630$$

R-squared measures improvement over always predicting the test target mean. It is not classification accuracy.

---

## 27. Coefficient interpretation

For standardized continuous variables, a coefficient describes the fitted-score change for a one-training-standard-deviation increase. For dummy variables, a coefficient compares that category with the reference category.

- **age:** holding other predictors fixed, changing this prepared predictor by one unit changes the fitted survival score by **-0.059116**.
- **sibsp:** holding other predictors fixed, changing this prepared predictor by one unit changes the fitted survival score by **-0.044778**.
- **parch:** holding other predictors fixed, changing this prepared predictor by one unit changes the fitted survival score by **-0.017002**.
- **fare:** holding other predictors fixed, changing this prepared predictor by one unit changes the fitted survival score by **+0.026427**.
- **Class=2:** holding other predictors fixed, changing this prepared predictor by one unit changes the fitted survival score by **-0.085690**.
- **Class=3:** holding other predictors fixed, changing this prepared predictor by one unit changes the fitted survival score by **-0.285446**.
- **Sex=male:** holding other predictors fixed, changing this prepared predictor by one unit changes the fitted survival score by **-0.511267**.
- **Embarked=Q:** holding other predictors fixed, changing this prepared predictor by one unit changes the fitted survival score by **-0.024164**.
- **Embarked=S:** holding other predictors fixed, changing this prepared predictor by one unit changes the fitted survival score by **-0.078437**.

These are associations in the fitted data, not automatic causal effects.

---

## 28. Threshold experiments

Threshold selection changes classification results but not regression scores.

| Threshold | TN | FP | FN | TP | Accuracy | Precision | Recall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 73 | 32 | 9 | 65 | 0.7709 | 0.6701 | 0.8784 |
| 0.40 | 84 | 21 | 17 | 57 | 0.7877 | 0.7308 | 0.7703 |
| 0.50 | 89 | 16 | 23 | 51 | 0.7821 | 0.7612 | 0.6892 |
| 0.60 | 96 | 9 | 23 | 51 | 0.8212 | 0.8500 | 0.6892 |
| 0.70 | 101 | 4 | 33 | 41 | 0.7933 | 0.9111 | 0.5541 |

---

## 29. Assumptions and limitations

1. The model is linear in its coefficients.
2. Passenger records are treated as independent after conditioning on predictors.
3. The conditional mean is assumed to be a linear combination of prepared predictors.
4. Exact multicollinearity must be avoided.
5. Ordinary Least Squares normally assumes constant residual variance, but a binary target necessarily violates this because the variance depends on the fitted mean.
6. Residuals for a binary target are not normally distributed.
7. Linear fitted scores can fall below 0 or above 1.
8. A score of 0.80 is not guaranteed to be a calibrated 80% probability.
9. Coefficients show conditional association, not necessarily causation.
10. Missing-age replacement loses information.
11. Results depend on the documented split and preprocessing.
12. Logistic regression is usually preferred when valid probability estimates are required.
13. “Exact” here means exact for the stated dataset, split, preparation, Ordinary Least Squares equation, and threshold, subject only to displayed rounding.

---

## 30. Formula reference and practice answers

### Core formula

$$\hat{y}=b_0+\sum_{{j=1}}^k b_jx_j$$

### Residual

$$e_i=y_i-\hat{y}_i$$

### SSE

$$SSE=\sum e_i^2$$

### Normal equation

$$\hat{\mathbf{b}}=(X^T X)^{-1}X^T\mathbf{y}$$

### MAE

$$MAE=\frac1n\sum|e_i|$$

### MSE

$$MSE=\frac1n\sum e_i^2$$

### RMSE

$$RMSE=\sqrt{{MSE}}$$

### R-squared

$$R^2=1-\frac{{SSE}}{{SST}}$$

### Practice 1
If the intercept is 0.25, the coefficient is 0.10, and x = 3:

$$\hat{y}=0.25+(0.10)(3)=0.55$$

At threshold 0.50, predict survived.

### Practice 2
If actual y = 1 and fitted score is 0.72:

$$e=1-0.72=0.28$$
$$e^2=0.0784$$

### Practice 3
For errors 0.20, -0.10, and 0.30:

$$MSE=\frac{{0.20^2+(-0.10)^2+0.30^2}}{{3}}=\frac{{0.14}}{{3}}=0.046667$$

$$RMSE=\sqrt{{0.046667}}=0.216025$$

---

## Appendix A. Reproducibility specification

- Preserve the original 891-row order.
- Use predictors pclass, sex, age, sibsp, parch, fare, and embarked.
- Use the fixed 80/20 shuffled split with state 42.
- Fit all replacement and scaling quantities on the 712 training rows only.
- Use class 1, female, and embarked C as references.
- Include an intercept.
- Estimate coefficients by unregularized Ordinary Least Squares.
- Retain raw scores for MAE, MSE, RMSE, and R-squared.
- Use a score threshold of 0.50 only for classification metrics.
- Keep full precision during arithmetic and round only for display.

## Appendix B. Final result summary

| Measure | Result |
|---|---:|
| Training observations | 712 |
| Test observations | 179 |
| Test MAE | 0.286623 |
| Test MSE | 0.137346 |
| Test RMSE | 0.370602 |
| Test R-squared | 0.433630 |
| Threshold accuracy | 0.782123 |
| Precision | 0.761194 |
| Recall | 0.689189 |
| F1 score | 0.723404 |
| TN | 89 |
| FP | 16 |
| FN | 23 |
| TP | 51 |