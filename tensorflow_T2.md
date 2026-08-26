# Complete Mathematical Explanation of the Model

> This guide explains the model using only **numbers, formulas, calculations, and mathematical reasoning**. It avoids Python and computer-programming explanations and shows the learning process step by step in very simple words.

---

## Table of Contents

1. [What problem are we trying to solve?](#1-what-problem-are-we-trying-to-solve)
2. [Mathematical shape of the training data](#2-mathematical-shape-of-the-training-data)
3. [The mathematical model](#3-the-mathematical-model)
4. [Why does the model have two parameters?](#4-why-does-the-model-have-two-parameters)
5. [Initial weight and bias](#5-initial-weight-and-bias)
6. [Reproducible starting values](#6-reproducible-starting-values)
7. [How predictions are calculated](#7-how-predictions-are-calculated)
8. [What is an error?](#8-what-is-an-error)
9. [Mean Squared Error](#9-mean-squared-error)
10. [Why do we square the errors?](#10-why-do-we-square-the-errors)
11. [Mean Absolute Error](#11-mean-absolute-error)
12. [Difference between MSE and MAE](#12-difference-between-mse-and-mae)
13. [The main goal of training](#13-the-main-goal-of-training)
14. [How does the model know which direction to move?](#14-how-does-the-model-know-which-direction-to-move)
15. [Why does the input appear in the weight gradient?](#15-why-does-the-input-appear-in-the-weight-gradient)
16. [Calculating the first weight gradient](#16-calculating-the-first-weight-gradient)
17. [Calculating the first bias gradient](#17-calculating-the-first-bias-gradient)
18. [What is the learning rate?](#18-what-is-the-learning-rate)
19. [Why is Adam used?](#19-why-is-adam-used)
20. [Adam optimizer formulas](#20-adam-optimizer-formulas)
21. [Example of Adam's first weight update](#21-example-of-adams-first-weight-update)
22. [Example of Adam's first bias update](#22-example-of-adams-first-bias-update)
23. [What is one epoch?](#23-what-is-one-epoch)
24. [What do 1,000 epochs mean?](#24-what-do-1000-epochs-mean)
25. [What is stored after every epoch?](#25-what-is-stored-after-every-epoch)
26. [First and final loss](#26-first-and-final-loss)
27. [Evaluating the final model](#27-evaluating-the-final-model)
28. [Learned weight and bias](#28-learned-weight-and-bias)
29. [How can we prove that weight is 10 and bias is 0?](#29-how-can-we-prove-that-weight-is-10-and-bias-is-0)
30. [All model weights](#30-all-model-weights)
31. [Trainable parameters](#31-trainable-parameters)
32. [Optimizer iterations](#32-optimizer-iterations)
33. [Making new predictions](#33-making-new-predictions)
34. [What if the learned values are not exact?](#34-what-if-the-learned-values-are-not-exact)
35. [Manual prediction check](#35-manual-prediction-check)
36. [Loss at selected epochs](#36-loss-at-selected-epochs)
37. [The complete mathematical learning cycle](#37-the-complete-mathematical-learning-cycle)
38. [Final simple summary](#final-simple-summary)

---

## 1. What problem are we trying to solve?

We are given five input-output examples:

$$
X=
\begin{bmatrix}
1\\
2\\
3\\
4\\
5
\end{bmatrix}
$$

and:

$$
Y=
\begin{bmatrix}
10\\
20\\
30\\
40\\
50
\end{bmatrix}
$$

This means:

| Input $X$ | Actual output $Y$ |
|---:|---:|
| 1 | 10 |
| 2 | 20 |
| 3 | 30 |
| 4 | 40 |
| 5 | 50 |

We want to discover the mathematical relationship between $X$ and $Y$.

Looking at the numbers:

$$
1\times10=10
$$

$$
2\times10=20
$$

$$
3\times10=30
$$

$$
4\times10=40
$$

$$
5\times10=50
$$

Therefore, the actual relationship is:

$$
\boxed{Y=10X}
$$

However, the model does not initially know this formula. It must discover it by repeatedly:

1. Making predictions.
2. Measuring its mistakes.
3. Correcting its weight and bias.
4. Repeating the process.

---

## 2. Mathematical shape of the training data

Each input is treated as one separate observation containing one value:

$$
X=
\begin{bmatrix}
1\\
2\\
3\\
4\\
5
\end{bmatrix}
$$

The mathematical size of this matrix is:

$$
\boxed{5\times1}
$$

This means:

- **5 rows** represent 5 observations.
- **1 column** represents 1 input variable.

Similarly:

$$
Y=
\begin{bmatrix}
10\\
20\\
30\\
40\\
50
\end{bmatrix}
$$

Its size is also:

$$
\boxed{5\times1}
$$

The complete dataset contains these five mathematical pairs:

$$
(1,10),\ (2,20),\ (3,30),\ (4,40),\ (5,50)
$$

---

## 3. The mathematical model

The model uses the straight-line formula:

$$
\boxed{\hat{Y}=wX+b}
$$

Where:

- $X$ is the input.
- $\hat{Y}$ is the predicted output.
- $w$ is the weight.
- $b$ is the bias.

The symbol $\hat{Y}$ is pronounced **“Y hat.”** It represents the model's predicted value.

- Actual value: $Y$
- Predicted value: $\hat{Y}$

### What is the weight?

The weight tells us how much the output changes when the input increases by 1.

For example, if:

$$
w=10
$$

then increasing $X$ by 1 increases the prediction by 10.

### What is the bias?

The bias is the prediction when:

$$
X=0
$$

Starting with:

$$
\hat{Y}=wX+b
$$

If $X=0$:

$$
\hat{Y}=w(0)+b
$$

$$
\hat{Y}=b
$$

Therefore, the bias is where the straight line crosses the vertical $Y$-axis.

---

## 4. Why does the model have two parameters?

The formula is:

$$
\hat{Y}=wX+b
$$

There are two unknown values:

1. Weight $w$
2. Bias $b$

Therefore:

$$
\boxed{\text{Total number of parameters}=2}
$$

The weight has one value:

$$
w=[w_1]
$$

Its shape is:

$$
\boxed{1\times1}
$$

The bias also has one value:

$$
b=[b_1]
$$

Its shape is:

$$
\boxed{1}
$$

Therefore:

$$
1\text{ weight}+1\text{ bias}=2\text{ parameters}
$$

---

## 5. Initial weight and bias

Before learning starts, the model begins with an initial weight and bias.

For a simple mathematical demonstration, suppose:

$$
w=0
$$

and:

$$
b=0
$$

The prediction formula becomes:

$$
\hat{Y}=0X+0
$$

Therefore:

$$
\hat{Y}=0
$$

For every input, the initial prediction is zero:

| $X$ | Actual $Y$ | Predicted $\hat{Y}$ |
|---:|---:|---:|
| 1 | 10 | 0 |
| 2 | 20 | 0 |
| 3 | 30 | 0 |
| 4 | 40 | 0 |
| 5 | 50 | 0 |

The model is currently wrong, but it can now measure how wrong it is.

> **Important:** Real training may begin with a small random weight rather than exactly zero. Zero is used here only to make the mathematical steps easier to understand.

---

## 6. Reproducible starting values

A fixed seed value, such as 42, is not part of the prediction formula.

It does **not** mean:

$$
w=42
$$

It also does **not** mean:

$$
b=42
$$

It simply makes the same starting random values appear whenever the same experiment is repeated.

Mathematically:

$$
\text{Same starting rule}
\Rightarrow
\text{Same initial }w,b
\Rightarrow
\text{Same training path}
$$

The number 42 has no special importance. Another fixed number could be used.

---

## 7. How predictions are calculated

For every observation, the model applies:

$$
\boxed{\hat{Y}_i=wX_i+b}
$$

The subscript $i$ represents the observation number.

For five observations:

$$
\hat{Y}_1=wX_1+b
$$

$$
\hat{Y}_2=wX_2+b
$$

$$
\hat{Y}_3=wX_3+b
$$

$$
\hat{Y}_4=wX_4+b
$$

$$
\hat{Y}_5=wX_5+b
$$

If $w=0$ and $b=0$:

$$
\hat{Y}_1=0(1)+0=0
$$

$$
\hat{Y}_2=0(2)+0=0
$$

$$
\hat{Y}_3=0(3)+0=0
$$

$$
\hat{Y}_4=0(4)+0=0
$$

$$
\hat{Y}_5=0(5)+0=0
$$

Therefore:

$$
\hat{Y}=
\begin{bmatrix}
0\\
0\\
0\\
0\\
0
\end{bmatrix}
$$

---

## 8. What is an error?

The error for one observation is:

$$
\boxed{e_i=\hat{Y}_i-Y_i}
$$

Where:

- $e_i$ is the error.
- $\hat{Y}_i$ is the predicted value.
- $Y_i$ is the actual value.

Using the initial predictions:

### First observation

$$
e_1=0-10=-10
$$

### Second observation

$$
e_2=0-20=-20
$$

### Third observation

$$
e_3=0-30=-30
$$

### Fourth observation

$$
e_4=0-40=-40
$$

### Fifth observation

$$
e_5=0-50=-50
$$

Therefore:

$$
e=
\begin{bmatrix}
-10\\
-20\\
-30\\
-40\\
-50
\end{bmatrix}
$$

A negative error means that the prediction is lower than the actual value.

---

## 9. Mean Squared Error

The primary loss is **Mean Squared Error**, or **MSE**.

Its formula is:

$$
\boxed{
MSE=\frac{1}{n}\sum_{i=1}^{n}(Y_i-\hat{Y}_i)^2
}
$$

Where:

- $n$ is the number of observations.
- $Y_i$ is the actual value.
- $\hat{Y}_i$ is the predicted value.
- $\sum$ means add all the results.

Here:

$$
n=5
$$

Therefore:

$$
MSE=
\frac{
(Y_1-\hat{Y}_1)^2+
(Y_2-\hat{Y}_2)^2+
(Y_3-\hat{Y}_3)^2+
(Y_4-\hat{Y}_4)^2+
(Y_5-\hat{Y}_5)^2
}{5}
$$

Using predictions of zero:

$$
MSE=
\frac{
(10-0)^2+
(20-0)^2+
(30-0)^2+
(40-0)^2+
(50-0)^2
}{5}
$$

Calculate each squared error:

$$
(10-0)^2=100
$$

$$
(20-0)^2=400
$$

$$
(30-0)^2=900
$$

$$
(40-0)^2=1600
$$

$$
(50-0)^2=2500
$$

Add them:

$$
100+400+900+1600+2500=5500
$$

Divide by 5:

$$
MSE=\frac{5500}{5}
$$

$$
\boxed{MSE=1100}
$$

This large loss means the predictions are far away from the actual values.

---

## 10. Why do we square the errors?

Suppose we simply added errors. Positive and negative errors could cancel each other.

For example:

$$
-10+10=0
$$

This would incorrectly suggest that there is no error.

Squaring solves this problem:

$$
(-10)^2=100
$$

$$
(10)^2=100
$$

Both results become positive.

Squaring also penalizes larger mistakes more heavily:

$$
2^2=4
$$

but:

$$
10^2=100
$$

Although 10 is five times larger than 2, its squared error is twenty-five times larger:

$$
\frac{100}{4}=25
$$

Therefore, MSE strongly encourages the model to correct large mistakes.

---

## 11. Mean Absolute Error

**Mean Absolute Error**, or **MAE**, is another way to measure mistakes.

Its formula is:

$$
\boxed{
MAE=\frac{1}{n}\sum_{i=1}^{n}|Y_i-\hat{Y}_i|
}
$$

The vertical bars represent absolute value. Absolute value removes the negative sign:

$$
|-10|=10
$$

$$
|10|=10
$$

Using the initial predictions:

$$
MAE=
\frac{
|10-0|+|20-0|+|30-0|+|40-0|+|50-0|
}{5}
$$

$$
MAE=\frac{10+20+30+40+50}{5}
$$

$$
MAE=\frac{150}{5}
$$

$$
\boxed{MAE=30}
$$

This means the predictions are wrong by an average of 30 units.

---

## 12. Difference between MSE and MAE

MSE uses squared errors:

$$
MSE=\frac{1}{n}\sum(Y_i-\hat{Y}_i)^2
$$

MAE uses absolute errors:

$$
MAE=\frac{1}{n}\sum|Y_i-\hat{Y}_i|
$$

For the initial predictions:

$$
MSE=1100
$$

$$
MAE=30
$$

Their numbers cannot be directly compared because they are measured differently.

| Measure | What it does | Main purpose |
|---|---|---|
| MSE | Squares every error | Strongly penalizes large mistakes and guides learning |
| MAE | Uses the absolute size of every error | Shows the average mistake in the original units |

---

## 13. The main goal of training

The objective is to find values of $w$ and $b$ that make MSE as small as possible.

Mathematically:

$$
\boxed{
\min_{w,b}
\frac{1}{n}
\sum_{i=1}^{n}
\left(Y_i-(wX_i+b)\right)^2
}
$$

This means:

> Find the weight and bias that produce the smallest possible average squared error.

The complete loss function is:

$$
L(w,b)=
\frac{1}{n}
\sum_{i=1}^{n}
\left(Y_i-wX_i-b\right)^2
$$

The letter $L$ represents loss.

---

## 14. How does the model know which direction to move?

The model uses **derivatives**.

A derivative tells us:

- Whether increasing a parameter will increase or decrease the loss.
- How strongly the loss will change.
- In which direction the parameter should move.

The derivative of the loss with respect to the weight is:

$$
\boxed{
\frac{\partial L}{\partial w}
=
\frac{2}{n}
\sum_{i=1}^{n}
X_i(\hat{Y}_i-Y_i)
}
$$

The derivative of the loss with respect to the bias is:

$$
\boxed{
\frac{\partial L}{\partial b}
=
\frac{2}{n}
\sum_{i=1}^{n}
(\hat{Y}_i-Y_i)
}
$$

These derivatives are called **gradients**.

---

## 15. Why does the input appear in the weight gradient?

The prediction formula is:

$$
\hat{Y}_i=wX_i+b
$$

The weight is multiplied by $X_i$. Therefore, changing the weight affects larger inputs more strongly.

If the weight increases by $0.1$:

When $X=1$:

$$
0.1\times1=0.1
$$

When $X=5$:

$$
0.1\times5=0.5
$$

Therefore, $X_i$ must appear in the weight gradient.

The bias is not multiplied by $X$, so the bias gradient does not contain $X_i$.

---

## 16. Calculating the first weight gradient

Suppose:

$$
w=0,\qquad b=0
$$

The predictions are:

$$
\hat{Y}_i=0
$$

The weight gradient is:

$$
\frac{\partial L}{\partial w}
=
\frac{2}{5}
\sum_{i=1}^{5}X_i(\hat{Y}_i-Y_i)
$$

Substitute the values:

$$
\frac{\partial L}{\partial w}
=
\frac{2}{5}
[
1(0-10)+
2(0-20)+
3(0-30)+
4(0-40)+
5(0-50)
]
$$

Calculate each part:

$$
1(-10)=-10
$$

$$
2(-20)=-40
$$

$$
3(-30)=-90
$$

$$
4(-40)=-160
$$

$$
5(-50)=-250
$$

Add them:

$$
-10-40-90-160-250=-550
$$

Therefore:

$$
\frac{\partial L}{\partial w}
=
\frac{2}{5}(-550)
$$

$$
\boxed{
\frac{\partial L}{\partial w}=-220
}
$$

The negative gradient tells us that the weight should increase to reduce the loss.

---

## 17. Calculating the first bias gradient

The bias gradient is:

$$
\frac{\partial L}{\partial b}
=
\frac{2}{5}
\sum_{i=1}^{5}(\hat{Y}_i-Y_i)
$$

Substitute the values:

$$
\frac{\partial L}{\partial b}
=
\frac{2}{5}
[(0-10)+(0-20)+(0-30)+(0-40)+(0-50)]
$$

$$
=
\frac{2}{5}[-10-20-30-40-50]
$$

$$
=
\frac{2}{5}(-150)
$$

$$
\boxed{
\frac{\partial L}{\partial b}=-60
}
$$

The negative gradient tells us that the bias should also increase.

---

## 18. What is the learning rate?

The learning rate is:

$$
\alpha=0.05
$$

The Greek letter $\alpha$, pronounced **alpha**, commonly represents the learning rate.

It controls the size of each parameter correction.

The basic update formulas are:

$$
\boxed{
w_{\text{new}}
=
w_{\text{old}}
-
\alpha\frac{\partial L}{\partial w}
}
$$

$$
\boxed{
b_{\text{new}}
=
b_{\text{old}}
-
\alpha\frac{\partial L}{\partial b}
}
$$

The minus sign is used because we want to move downward toward a smaller loss.

### Why not use a very large learning rate?

If the learning rate is too large, the parameter can jump over the best answer.

For example, suppose the best weight is 10. A very large update could cause:

$$
0\rightarrow15
$$

The next correction might cause:

$$
15\rightarrow-5
$$

The weight may keep jumping around the correct value.

### Why not use a very small learning rate?

A very small learning rate creates tiny updates:

$$
0\rightarrow0.0001\rightarrow0.0002\rightarrow0.0003
$$

It may eventually reach the correct answer, but learning could take a very long time.

---

## 19. Why is Adam used?

Adam maintains two moving averages:

1. A moving average of the gradients.
2. A moving average of the squared gradients.

These help Adam control both the direction and size of each update.

---

## 20. Adam optimizer formulas

Let the gradient at step $t$ be:

$$
g_t
$$

For the weight:

$$
g_t=\frac{\partial L}{\partial w}
$$

For the bias:

$$
g_t=\frac{\partial L}{\partial b}
$$

Adam calculates the first moving average:

$$
\boxed{
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
}
$$

It calculates the second moving average:

$$
\boxed{
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
}
$$

Common values are:

$$
\beta_1=0.9
$$

$$
\beta_2=0.999
$$

The first moving average follows the general direction of the gradients. The second follows the size of the squared gradients.

### Bias correction

At the beginning:

$$
m_0=0
$$

$$
v_0=0
$$

This initially makes the averages artificially small, so Adam corrects them:

$$
\boxed{
\hat{m}_t=\frac{m_t}{1-\beta_1^t}
}
$$

$$
\boxed{
\hat{v}_t=\frac{v_t}{1-\beta_2^t}
}
$$

The final Adam update is:

$$
\boxed{
\theta_t
=
\theta_{t-1}
-
\alpha
\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
}
$$

Here, $\theta$ means any trainable parameter, such as $w$ or $b$.

The value $\epsilon$ is a tiny positive number, commonly:

$$
\epsilon=10^{-7}
$$

It prevents division by zero.

---

## 21. Example of Adam's first weight update

The first weight gradient was:

$$
g_1=-220
$$

Initially:

$$
m_0=0,\qquad v_0=0
$$

Using $\beta_1=0.9$:

$$
m_1=0.9(0)+(1-0.9)(-220)
$$

$$
m_1=0.1(-220)
$$

$$
m_1=-22
$$

Now calculate the second moving average:

$$
v_1=0.999(0)+(1-0.999)(-220)^2
$$

$$
v_1=0.001(48400)
$$

$$
v_1=48.4
$$

Correct the first moving average:

$$
\hat{m}_1=\frac{-22}{1-0.9^1}
$$

$$
\hat{m}_1=\frac{-22}{0.1}
$$

$$
\hat{m}_1=-220
$$

Correct the second moving average:

$$
\hat{v}_1=\frac{48.4}{1-0.999^1}
$$

$$
\hat{v}_1=\frac{48.4}{0.001}
$$

$$
\hat{v}_1=48400
$$

Its square root is:

$$
\sqrt{48400}=220
$$

Using $\alpha=0.05$:

$$
w_1=w_0-0.05\frac{-220}{220+\epsilon}
$$

Ignoring the extremely small $\epsilon$ for simplicity:

$$
w_1=0-0.05(-1)
$$

$$
\boxed{w_1\approx0.05}
$$

The weight moves upward because the model needs larger predictions.

---

## 22. Example of Adam's first bias update

The first bias gradient was:

$$
g_1=-60
$$

For the first Adam update, the corrected averages produce approximately:

$$
\frac{\hat{m}_1}{\sqrt{\hat{v}_1}}\approx-1
$$

Therefore:

$$
b_1=0-0.05(-1)
$$

$$
\boxed{b_1\approx0.05}
$$

After the first simplified Adam update, the formula becomes approximately:

$$
\hat{Y}=0.05X+0.05
$$

The model is still far from the correct formula:

$$
Y=10X
$$

However, it has moved in the correct direction.

---

## 23. What is one epoch?

One epoch means all five training examples are used once to calculate predictions and update the model.

One epoch includes:

### Step 1: Calculate all predictions

$$
\hat{Y}_i=wX_i+b
$$

### Step 2: Calculate the loss

$$
L=\frac{1}{5}\sum_{i=1}^{5}(Y_i-\hat{Y}_i)^2
$$

### Step 3: Calculate the gradients

$$
\frac{\partial L}{\partial w}
\qquad\text{and}\qquad
\frac{\partial L}{\partial b}
$$

### Step 4: Update the parameters

$$
w_{\text{new}}\leftarrow w_{\text{old}}+\text{correction}
$$

$$
b_{\text{new}}\leftarrow b_{\text{old}}+\text{correction}
$$

---

## 24. What do 1,000 epochs mean?

The complete learning cycle is repeated 1,000 times:

$$
\boxed{\text{Epochs}=1000}
$$

Conceptually:

$$
(w_0,b_0)
\rightarrow
(w_1,b_1)
\rightarrow
(w_2,b_2)
\rightarrow
\cdots
\rightarrow
(w_{1000},b_{1000})
$$

During successful learning:

$$
w_t\rightarrow10
$$

$$
b_t\rightarrow0
$$

The loss moves toward zero:

$$
L_t\rightarrow0
$$

---

## 25. What is stored after every epoch?

After every epoch, MSE and MAE are calculated.

The loss history is a sequence:

$$
[L_1,L_2,L_3,\ldots,L_{1000}]
$$

Where:

- $L_1$ is the MSE after epoch 1.
- $L_2$ is the MSE after epoch 2.
- $L_{1000}$ is the final MSE.

The MAE history is another sequence:

$$
[A_1,A_2,A_3,\ldots,A_{1000}]
$$

Each $A_t$ is the MAE at epoch $t$.

A successful training pattern may look conceptually like:

$$
1100\rightarrow900\rightarrow700\rightarrow400\rightarrow100
\rightarrow10\rightarrow1\rightarrow0.01
$$

These are only example values. Exact values depend on the starting parameters and every Adam update.

---

## 26. First and final loss

The first loss is:

$$
\boxed{L_1}
$$

The final loss is:

$$
\boxed{L_{1000}}
$$

Usually:

$$
L_{1000}<L_1
$$

This indicates that later predictions are better than earlier predictions.

Ideally:

$$
L_{1000}\approx0
$$

---

## 27. Evaluating the final model

After training, the final weight and bias are used on the original five inputs.

Suppose:

$$
w=10,\qquad b=0
$$

Then:

$$
\hat{Y}=10X+0
$$

For $X=1$:

$$
\hat{Y}=10(1)+0=10
$$

For $X=2$:

$$
\hat{Y}=10(2)+0=20
$$

For $X=3$:

$$
\hat{Y}=10(3)+0=30
$$

For $X=4$:

$$
\hat{Y}=10(4)+0=40
$$

For $X=5$:

$$
\hat{Y}=10(5)+0=50
$$

The predictions match the actual values exactly. Therefore:

$$
MSE=0
$$

and:

$$
MAE=0
$$

In real iterative training, the results may be extremely close to zero rather than exactly zero because of decimal precision.

---

## 28. Learned weight and bias

At the end of training, suppose the model learns:

$$
w\approx10
$$

$$
b\approx0
$$

Then the learned formula is:

$$
\boxed{\hat{Y}=10X+0}
$$

which simplifies to:

$$
\boxed{\hat{Y}=10X}
$$

This is the same relationship present in the original data.

---

## 29. How can we prove that weight is 10 and bias is 0?

The best-fitting straight-line weight can be calculated directly:

$$
\boxed{
w=
\frac{
\sum_{i=1}^{n}(X_i-\bar{X})(Y_i-\bar{Y})
}{
\sum_{i=1}^{n}(X_i-\bar{X})^2
}
}
$$

The bias is:

$$
\boxed{b=\bar{Y}-w\bar{X}}
$$

Where:

$$
\bar{X}=\text{mean of }X
$$

$$
\bar{Y}=\text{mean of }Y
$$

### Calculate the mean of $X$

$$
\bar{X}=\frac{1+2+3+4+5}{5}
$$

$$
\bar{X}=\frac{15}{5}
$$

$$
\boxed{\bar{X}=3}
$$

### Calculate the mean of $Y$

$$
\bar{Y}=\frac{10+20+30+40+50}{5}
$$

$$
\bar{Y}=\frac{150}{5}
$$

$$
\boxed{\bar{Y}=30}
$$

### Calculate the numerator for $w$

$$
\sum(X_i-\bar{X})(Y_i-\bar{Y})
$$

Substitute the means:

$$
(1-3)(10-30)
+(2-3)(20-30)
+(3-3)(30-30)
+(4-3)(40-30)
+(5-3)(50-30)
$$

Calculate each part:

$$
(-2)(-20)=40
$$

$$
(-1)(-10)=10
$$

$$
(0)(0)=0
$$

$$
(1)(10)=10
$$

$$
(2)(20)=40
$$

Add them:

$$
40+10+0+10+40=100
$$

The numerator is 100.

### Calculate the denominator for $w$

$$
\sum(X_i-\bar{X})^2
$$

$$
(1-3)^2+(2-3)^2+(3-3)^2+(4-3)^2+(5-3)^2
$$

$$
(-2)^2+(-1)^2+0^2+1^2+2^2
$$

$$
4+1+0+1+4=10
$$

Therefore:

$$
w=\frac{100}{10}
$$

$$
\boxed{w=10}
$$

Now calculate the bias:

$$
b=\bar{Y}-w\bar{X}
$$

$$
b=30-(10)(3)
$$

$$
b=30-30
$$

$$
\boxed{b=0}
$$

Therefore, the exact best formula is:

$$
\boxed{Y=10X}
$$

---

## 30. All model weights

This model contains only:

$$
w
$$

and:

$$
b
$$

Therefore, the complete parameter collection is:

$$
\boxed{\{w,b\}}
$$

If the learned values are:

$$
w=10,\qquad b=0
$$

then the complete parameter collection is:

$$
\boxed{\{10,0\}}
$$

There are no additional prediction parameters in this simple model.

---

## 31. Trainable parameters

A trainable parameter is a value that can be changed to reduce the loss.

The weight is trainable because changing $w$ changes the slope of the line.

The bias is trainable because changing $b$ moves the line upward or downward.

Therefore:

$$
\boxed{\text{Trainable parameters}=\{w,b\}}
$$

---

## 32. Optimizer iterations

An optimizer iteration means one parameter update:

$$
(w_t,b_t)\rightarrow(w_{t+1},b_{t+1})
$$

Because all five observations fit into one group, one epoch uses one update:

$$
1\text{ epoch}=1\text{ update}
$$

For 1,000 epochs:

$$
1000\text{ epochs}\times1\text{ update per epoch}
=1000\text{ updates}
$$

Therefore:

$$
\boxed{\text{Iterations}=1000}
$$

If observations were divided into multiple groups, one epoch could contain several optimizer iterations.

---

## 33. Making new predictions

The new inputs are:

$$
X_{\text{new}}=
\begin{bmatrix}
6\\
7\\
8
\end{bmatrix}
$$

The final learned formula is approximately:

$$
\hat{Y}=10X
$$

### Prediction for $X=6$

$$
\hat{Y}=10(6)
$$

$$
\boxed{\hat{Y}=60}
$$

### Prediction for $X=7$

$$
\hat{Y}=10(7)
$$

$$
\boxed{\hat{Y}=70}
$$

### Prediction for $X=8$

$$
\hat{Y}=10(8)
$$

$$
\boxed{\hat{Y}=80}
$$

Therefore:

$$
\hat{Y}_{\text{new}}=
\begin{bmatrix}
60\\
70\\
80
\end{bmatrix}
$$

---

## 34. What if the learned values are not exact?

Suppose the learned values are:

$$
w=9.95
$$

$$
b=0.12
$$

Then:

$$
\hat{Y}=9.95X+0.12
$$

For $X=6$:

$$
\hat{Y}=9.95(6)+0.12
$$

$$
\hat{Y}=59.70+0.12
$$

$$
\boxed{\hat{Y}=59.82}
$$

For $X=7$:

$$
\hat{Y}=9.95(7)+0.12
$$

$$
\hat{Y}=69.65+0.12
$$

$$
\boxed{\hat{Y}=69.77}
$$

For $X=8$:

$$
\hat{Y}=9.95(8)+0.12
$$

$$
\hat{Y}=79.60+0.12
$$

$$
\boxed{\hat{Y}=79.72}
$$

These are very close to 60, 70, and 80, but they are not perfectly exact.

---

## 35. Manual prediction check

A manual prediction uses the learned formula:

$$
\boxed{\hat{Y}=wX+b}
$$

Suppose:

$$
w=10,\qquad b=0
$$

For the new inputs:

$$
X_{\text{new}}=
\begin{bmatrix}
6\\
7\\
8
\end{bmatrix}
$$

Calculate:

$$
\hat{Y}_{\text{new}}
=
\begin{bmatrix}
6\\
7\\
8
\end{bmatrix}
(10)+0
$$

Apply the multiplication to every input:

$$
\hat{Y}_{\text{new}}
=
\begin{bmatrix}
6(10)+0\\
7(10)+0\\
8(10)+0
\end{bmatrix}
$$

$$
\boxed{
\hat{Y}_{\text{new}}
=
\begin{bmatrix}
60\\
70\\
80
\end{bmatrix}
}
$$

The manual check confirms that every prediction follows the learned mathematical equation.

---

## 36. Loss at selected epochs

Suppose we examine these epochs:

$$
1,\ 10,\ 50,\ 100,\ 200,\ 500,\ 1000
$$

The corresponding losses are:

$$
L_1,\ L_{10},\ L_{50},\ L_{100},\ L_{200},\ L_{500},\ L_{1000}
$$

A successful learning process should generally show:

$$
L_1>L_{10}>L_{50}>L_{100}>\cdots>L_{1000}
$$

The loss may not decrease perfectly at every single step, but its overall direction should be downward:

$$
\text{Large error}
\rightarrow
\text{Medium error}
\rightarrow
\text{Small error}
\rightarrow
\text{Almost zero error}
$$

---

## 37. The complete mathematical learning cycle

### Step 1: Start with parameters

$$
w=w_0
$$

$$
b=b_0
$$

### Step 2: Calculate predictions

$$
\hat{Y}_i=wX_i+b
$$

### Step 3: Calculate errors

$$
e_i=\hat{Y}_i-Y_i
$$

### Step 4: Calculate MSE

$$
L=\frac{1}{n}\sum_{i=1}^{n}e_i^2
$$

### Step 5: Calculate the weight gradient

$$
\frac{\partial L}{\partial w}
=
\frac{2}{n}\sum_{i=1}^{n}X_ie_i
$$

### Step 6: Calculate the bias gradient

$$
\frac{\partial L}{\partial b}
=
\frac{2}{n}\sum_{i=1}^{n}e_i
$$

### Step 7: Use Adam to convert gradients into corrections

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

$$
\hat{m}_t=\frac{m_t}{1-\beta_1^t}
$$

$$
\hat{v}_t=\frac{v_t}{1-\beta_2^t}
$$

### Step 8: Update each parameter

$$
\theta_t
=
\theta_{t-1}
-
\alpha
\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$

### Step 9: Repeat 1,000 times

$$
t=1,2,3,\ldots,1000
$$

### Step 10: Obtain the final formula

$$
\boxed{\hat{Y}\approx10X}
$$

---

# Final Simple Summary

The original data follows:

$$
\boxed{Y=10X}
$$

The model starts with an unknown formula:

$$
\hat{Y}=wX+b
$$

It measures its mistake using:

$$
\boxed{
MSE=\frac{1}{n}\sum(Y_i-\hat{Y}_i)^2
}
$$

It calculates how the weight should change using:

$$
\boxed{
\frac{\partial L}{\partial w}
=
\frac{2}{n}\sum X_i(\hat{Y}_i-Y_i)
}
$$

It calculates how the bias should change using:

$$
\boxed{
\frac{\partial L}{\partial b}
=
\frac{2}{n}\sum(\hat{Y}_i-Y_i)
}
$$

Adam repeatedly adjusts the parameters:

$$
w\rightarrow10
$$

$$
b\rightarrow0
$$

The final learned equation becomes:

$$
\boxed{\hat{Y}=10X}
$$

Therefore:

$$
X=6\Rightarrow\hat{Y}=60
$$

$$
X=7\Rightarrow\hat{Y}=70
$$

$$
X=8\Rightarrow\hat{Y}=80
$$

---

## Important Note

The exact loss values across all 1,000 epochs cannot be calculated from the final formula alone. They depend on:

- The exact initial weight and bias.
- Every gradient produced during training.
- Adam's moving averages at every step.
- Decimal precision used during the calculations.

However, all the formulas required to calculate those values are included in this guide.
