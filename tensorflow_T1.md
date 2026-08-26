# How a Single-Neuron Model Learns `Y = 10X`: Complete Mathematical Explanation

> This document explains the model entirely through mathematics. It does not focus on Python, TensorFlow, programming syntax, or computer operations.

---

## 1. The Mathematical Problem

We have five examples:

$$
\begin{aligned}
1 &\rightarrow 10\\
2 &\rightarrow 20\\
3 &\rightarrow 30\\
4 &\rightarrow 40\\
5 &\rightarrow 50
\end{aligned}
$$

We can represent the input values as:

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

The corresponding correct output values are:

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

By looking at the examples, we can see that the actual relationship is:

$$
Y=10X
$$

For example:

$$
\begin{aligned}
10 &= 10(1)\\
20 &= 10(2)\\
30 &= 10(3)\\
40 &= 10(4)\\
50 &= 10(5)
\end{aligned}
$$

But the model does not initially know that the multiplier is `10`. The model must discover that value through repeated mathematical calculations.

---

## 2. The Mathematical Model

The model assumes that the relationship has this general form:

$$
\boxed{\hat{Y}=wX+b}
$$

Where:

- $X$ is the input.
- $\hat{Y}$ is the model's predicted output.
- $w$ is the **weight**.
- $b$ is the **bias**.
- The hat over $\hat{Y}$ means **predicted value**.

The model must learn the correct values of $w$ and $b$.

For this dataset, the perfect values are:

$$
w=10
$$

and:

$$
b=0
$$

Therefore, the perfect formula is:

$$
\hat{Y}=10X+0
$$

or simply:

$$
\boxed{\hat{Y}=10X}
$$

---

## 3. Why Do We Need a Weight?

The weight controls how strongly the input affects the output.

Consider:

$$
\hat{Y}=wX+b
$$

If:

$$
w=2,\qquad b=0
$$

then:

$$
\hat{Y}=2X
$$

For $X=5$:

$$
\hat{Y}=2(5)=10
$$

If:

$$
w=10,\qquad b=0
$$

then:

$$
\hat{Y}=10X
$$

For $X=5$:

$$
\hat{Y}=10(5)=50
$$

Therefore, increasing the weight increases how quickly the output grows. In this dataset, every input is multiplied by `10`, so the correct weight is:

$$
\boxed{w=10}
$$

---

## 4. Why Do We Need a Bias?

The bias allows the entire line to move upward or downward.

Consider:

$$
\hat{Y}=wX+b
$$

Suppose:

$$
w=10,\qquad b=5
$$

Then:

$$
\hat{Y}=10X+5
$$

For different values of $X$:

$$
\begin{aligned}
X=1 &: \hat{Y}=10(1)+5=15\\
X=2 &: \hat{Y}=10(2)+5=25\\
X=3 &: \hat{Y}=10(3)+5=35
\end{aligned}
$$

The bias added `5` to every prediction. However, the correct outputs are:

$$
10,\ 20,\ 30,\ 40,\ 50
$$

No additional amount is added. Therefore, the ideal bias is:

$$
\boxed{b=0}
$$

---

## 5. The Model Starts with Initial Values

The model does not start with:

$$
w=10,\qquad b=0
$$

That would mean the model already knew the answer.

Instead, the weight usually starts with a small random value. For this mathematical demonstration, suppose the model starts with:

$$
w=1
$$

and:

$$
b=0
$$

The starting formula is:

$$
\hat{Y}=1X+0
$$

or:

$$
\hat{Y}=X
$$

The initial predictions are therefore:

| Input $X$ | Actual $Y$ | Prediction $\hat{Y}$ |
|---:|---:|---:|
| 1 | 10 | 1 |
| 2 | 20 | 2 |
| 3 | 30 | 3 |
| 4 | 40 | 4 |
| 5 | 50 | 5 |

The predictions are far below the correct answers, so the weight must increase.

---

## 6. Finding the Error

For each example, the error is:

$$
\boxed{e_i=\hat{Y}_i-Y_i}
$$

Where:

- $e_i$ is the error for example $i$.
- $\hat{Y}_i$ is the predicted value.
- $Y_i$ is the correct value.

Using the initial predictions:

$$
\begin{aligned}
e_1 &= 1-10=-9\\
e_2 &= 2-20=-18\\
e_3 &= 3-30=-27\\
e_4 &= 4-40=-36\\
e_5 &= 5-50=-45
\end{aligned}
$$

The errors are:

$$
\begin{bmatrix}
-9\\
-18\\
-27\\
-36\\
-45
\end{bmatrix}
$$

The negative errors tell us that the predictions are smaller than the correct answers.

---

## 7. Why We Cannot Simply Add the Errors

If some errors are positive and others are negative, they can cancel each other.

For example:

$$
5+(-5)=0
$$

A total error of zero might suggest there is no mistake, even though both predictions were wrong by `5`.

Therefore, we need a method that prevents positive and negative errors from cancelling each other. One method is to square every error.

---

## 8. Mean Squared Error

The training loss is **Mean Squared Error**, or **MSE**.

The formula is:

$$
\boxed{
\operatorname{MSE}
=
\frac{1}{n}
\sum_{i=1}^{n}
(Y_i-\hat{Y}_i)^2
}
$$

Where:

- $n$ is the number of examples.
- $Y_i$ is the correct value.
- $\hat{Y}_i$ is the predicted value.
- $\sum$ means add all the values.
- Squaring removes negative signs.
- Dividing by $n$ calculates the average.

We have five examples, so:

$$
n=5
$$

The squared errors are:

$$
\begin{aligned}
(-9)^2 &= 81\\
(-18)^2 &= 324\\
(-27)^2 &= 729\\
(-36)^2 &= 1296\\
(-45)^2 &= 2025
\end{aligned}
$$

Add them:

$$
81+324+729+1296+2025=4455
$$

Now divide by the number of examples:

$$
\operatorname{MSE}=\frac{4455}{5}
$$

Therefore:

$$
\boxed{\operatorname{MSE}=891}
$$

This is a high loss because the predictions are far from the correct answers.

---

## 9. Why Do We Square the Errors?

### Reason 1: Negative values become positive

$$
(-9)^2=81
$$

Positive and negative errors therefore do not cancel each other.

### Reason 2: Large mistakes receive a larger penalty

Compare an error of `2` with an error of `10`:

$$
2^2=4
$$

$$
10^2=100
$$

The error of `10` receives a much larger penalty.

### Reason 3: The squared function is differentiable

We can calculate its derivative. The derivative tells the model how the weight and bias should change to reduce the loss.

---

## 10. Mean Absolute Error

**MAE** means **Mean Absolute Error**.

The formula is:

$$
\boxed{
\operatorname{MAE}
=
\frac{1}{n}
\sum_{i=1}^{n}
|Y_i-\hat{Y}_i|
}
$$

The vertical lines mean absolute value. Absolute value removes the negative sign:

$$
|-9|=9
$$

Using our errors:

$$
\begin{aligned}
|-9| &= 9\\
|-18| &= 18\\
|-27| &= 27\\
|-36| &= 36\\
|-45| &= 45
\end{aligned}
$$

Add them:

$$
9+18+27+36+45=135
$$

Divide by `5`:

$$
\operatorname{MAE}=\frac{135}{5}
$$

Therefore:

$$
\boxed{\operatorname{MAE}=27}
$$

This means that, on average, the predictions are `27` units away from the correct answers.

---

## 11. Difference Between MSE and MAE

For the starting model:

$$
\operatorname{MSE}=891
$$

and:

$$
\operatorname{MAE}=27
$$

MSE squares the errors:

$$
\operatorname{MSE}=\text{average of squared errors}
$$

MAE uses the absolute errors:

$$
\operatorname{MAE}=\text{average distance from the correct answers}
$$

MSE is used as the main loss to train this model. MAE is an additional measurement that is often easier for a person to understand.

---

## 12. MSE in Terms of Weight and Bias

The prediction formula is:

$$
\hat{Y}_i=wX_i+b
$$

Substitute that into the MSE formula:

$$
\operatorname{MSE}
=
\frac{1}{n}
\sum_{i=1}^{n}
\left(Y_i-(wX_i+b)\right)^2
$$

It can also be written as:

$$
\boxed{
L(w,b)
=
\frac{1}{n}
\sum_{i=1}^{n}
(wX_i+b-Y_i)^2
}
$$

The letter $L$ represents **loss**. This formula says that the loss depends on $w$ and $b$.

The goal is to find the values of $w$ and $b$ that make the loss as small as possible:

$$
\boxed{\min_{w,b}L(w,b)}
$$

This means:

> Find the weight and bias that minimize the loss.

---

## 13. How Does the Model Know Which Direction to Move?

The model uses **derivatives**.

A derivative tells us:

- Whether increasing a value will increase or decrease the loss.
- How strongly the loss will change.
- Which direction should be used to reduce the loss.

The derivative of the loss with respect to the weight is:

$$
\boxed{
\frac{\partial L}{\partial w}
=
\frac{2}{n}
\sum_{i=1}^{n}
X_i(wX_i+b-Y_i)
}
$$

The derivative of the loss with respect to the bias is:

$$
\boxed{
\frac{\partial L}{\partial b}
=
\frac{2}{n}
\sum_{i=1}^{n}
(wX_i+b-Y_i)
}
$$

These derivatives are also called **gradients**.

---

## 14. Calculating the Weight Gradient

Suppose:

$$
w=1,\qquad b=0
$$

The errors using $\hat{Y}-Y$ are:

$$
-9,-18,-27,-36,-45
$$

The weight gradient is:

$$
\frac{\partial L}{\partial w}
=
\frac{2}{5}
\sum_{i=1}^{5}X_i(\hat{Y}_i-Y_i)
$$

Substitute the numbers:

$$
\frac{\partial L}{\partial w}
=
\frac{2}{5}
\left[1(-9)+2(-18)+3(-27)+4(-36)+5(-45)\right]
$$

Calculate each multiplication:

$$
\begin{aligned}
1(-9)&=-9\\
2(-18)&=-36\\
3(-27)&=-81\\
4(-36)&=-144\\
5(-45)&=-225
\end{aligned}
$$

Add them:

$$
-9-36-81-144-225=-495
$$

Multiply by $\frac{2}{5}$:

$$
\frac{\partial L}{\partial w}
=
\frac{2}{5}(-495)
$$

Therefore:

$$
\boxed{\frac{\partial L}{\partial w}=-198}
$$

The gradient is negative. That means the model must increase the weight to reduce the loss.

---

## 15. Calculating the Bias Gradient

The bias gradient is:

$$
\frac{\partial L}{\partial b}
=
\frac{2}{5}
\sum_{i=1}^{5}(\hat{Y}_i-Y_i)
$$

Substitute the errors:

$$
\frac{\partial L}{\partial b}
=
\frac{2}{5}[-9-18-27-36-45]
$$

Add the errors:

$$
-9-18-27-36-45=-135
$$

Therefore:

$$
\frac{\partial L}{\partial b}
=
\frac{2}{5}(-135)
$$

$$
\boxed{\frac{\partial L}{\partial b}=-54}
$$

The bias gradient is also negative, so the bias will initially increase.

---

## 16. Learning Rate

The learning rate is:

$$
\alpha=0.05
$$

The Greek letter $\alpha$, pronounced **alpha**, is commonly used for the learning rate.

The learning rate controls the size of each mathematical update.

A smaller learning rate creates smaller changes:

$$
\alpha=0.0001
$$

A larger learning rate creates larger changes:

$$
\alpha=0.05
$$

If the learning rate is too small, learning may be very slow. If the learning rate is too large, the model may repeatedly move past the best answer.

---

## 17. Basic Gradient Descent Update

In ordinary gradient descent, the update formulas are:

$$
\boxed{
w_{\text{new}}
=
w_{\text{old}}
-
\alpha\frac{\partial L}{\partial w}
}
$$

and:

$$
\boxed{
b_{\text{new}}
=
b_{\text{old}}
-
\alpha\frac{\partial L}{\partial b}
}
$$

The gradient tells us the direction. The learning rate controls the size of the movement.

For example, using:

$$
w_{\text{old}}=1
$$

$$
\alpha=0.001
$$

and:

$$
\frac{\partial L}{\partial w}=-198
$$

we get:

$$
w_{\text{new}}=1-0.001(-198)
$$

A negative multiplied by a negative becomes positive:

$$
w_{\text{new}}=1+0.198
$$

Therefore:

$$
w_{\text{new}}=1.198
$$

The weight increased because the original predictions were too small.

> The example above uses ordinary gradient descent with a learning rate of `0.001` only to make the basic update easy to see. Adam uses a different update calculation, explained next.

---

## 18. What Adam Does Mathematically

Adam keeps track of two things:

1. The recent average of the gradients.
2. The recent average of the squared gradients.

For a parameter such as the weight, let the gradient at training step $t$ be:

$$
g_t
$$

For the first step:

$$
g_1=-198
$$

Adam normally uses two constants:

$$
\beta_1=0.9
$$

and:

$$
\beta_2=0.999
$$

### 18.1 First Moment

The first moment is:

$$
\boxed{
m_t
=
\beta_1m_{t-1}
+
(1-\beta_1)g_t
}
$$

The first moment is similar to a moving average of the gradients.

Initially:

$$
m_0=0
$$

For the first step:

$$
m_1=0.9(0)+0.1(-198)
$$

$$
m_1=-19.8
$$

### 18.2 Second Moment

The second moment is:

$$
\boxed{
v_t
=
\beta_2v_{t-1}
+
(1-\beta_2)g_t^2
}
$$

The second moment tracks the squared gradients.

Initially:

$$
v_0=0
$$

First calculate:

$$
g_1^2=(-198)^2=39204
$$

Then:

$$
v_1=0.999(0)+0.001(39204)
$$

$$
v_1=39.204
$$

---

## 19. Adam Bias Correction

During the first few steps, the moving averages are close to zero because the moving averages started from zero. Adam corrects this effect.

The corrected first moment is:

$$
\boxed{
\hat{m}_t
=
\frac{m_t}{1-\beta_1^t}
}
$$

For the first step:

$$
\hat{m}_1
=
\frac{-19.8}{1-0.9^1}
$$

$$
\hat{m}_1
=
\frac{-19.8}{0.1}
$$

$$
\boxed{\hat{m}_1=-198}
$$

The corrected second moment is:

$$
\boxed{
\hat{v}_t
=
\frac{v_t}{1-\beta_2^t}
}
$$

For the first step:

$$
\hat{v}_1
=
\frac{39.204}{1-0.999^1}
$$

$$
\hat{v}_1
=
\frac{39.204}{0.001}
$$

$$
\boxed{\hat{v}_1=39204}
$$

---

## 20. Adam's Final Update Formula

Adam updates a parameter using:

$$
\boxed{
\theta_t
=
\theta_{t-1}
-
\alpha
\frac{\hat{m}_t}
{\sqrt{\hat{v}_t}+\epsilon}
}
$$

Where:

- $\theta$ represents the parameter, such as weight or bias.
- $\alpha$ is the learning rate.
- $\hat{m}_t$ is the corrected first moment.
- $\hat{v}_t$ is the corrected second moment.
- $\epsilon$ is a tiny number used to avoid division by zero.

For the weight:

$$
w_1
=
1
-
0.05
\frac{-198}{\sqrt{39204}+\epsilon}
$$

Because:

$$
\sqrt{39204}=198
$$

we get approximately:

$$
w_1
=
1
-
0.05\left(\frac{-198}{198}\right)
$$

$$
w_1=1-0.05(-1)
$$

$$
w_1=1+0.05
$$

Therefore:

$$
\boxed{w_1\approx1.05}
$$

Adam similarly updates the bias from approximately:

$$
b_0=0
$$

to:

$$
\boxed{b_1\approx0.05}
$$

After the first update, the new formula is approximately:

$$
\boxed{\hat{Y}=1.05X+0.05}
$$

---

## 21. Checking the New Predictions

Using:

$$
\hat{Y}=1.05X+0.05
$$

we get:

$$
\begin{aligned}
X=1 &: \hat{Y}=1.05(1)+0.05=1.10\\
X=2 &: \hat{Y}=1.05(2)+0.05=2.15\\
X=3 &: \hat{Y}=1.05(3)+0.05=3.20\\
X=4 &: \hat{Y}=1.05(4)+0.05=4.25\\
X=5 &: \hat{Y}=1.05(5)+0.05=5.30
\end{aligned}
$$

The new predictions are:

$$
\begin{bmatrix}
1.10\\
2.15\\
3.20\\
4.25\\
5.30
\end{bmatrix}
$$

These values are still far from the actual answers, but the new values are slightly better than the original predictions.

---

## 22. Calculating the New Loss

The new errors are:

$$
\begin{aligned}
1.10-10&=-8.90\\
2.15-20&=-17.85\\
3.20-30&=-26.80\\
4.25-40&=-35.75\\
5.30-50&=-44.70
\end{aligned}
$$

Square the errors:

$$
\begin{aligned}
(-8.90)^2 &=79.21\\
(-17.85)^2 &=318.6225\\
(-26.80)^2 &=718.24\\
(-35.75)^2 &=1278.0625\\
(-44.70)^2 &=1998.09
\end{aligned}
$$

Add them:

$$
79.21+318.6225+718.24+1278.0625+1998.09=4392.225
$$

Divide by `5`:

$$
\operatorname{MSE}=\frac{4392.225}{5}
$$

$$
\boxed{\operatorname{MSE}=878.445}
$$

The original loss was:

$$
891
$$

The new loss is:

$$
878.445
$$

Therefore:

$$
878.445<891
$$

The loss decreased. This means the mathematical update moved the weight and bias in the correct direction.

---

## 23. What Is One Epoch?

One epoch means the complete set of five examples is used once:

$$
(1,10),\ (2,20),\ (3,30),\ (4,40),\ (5,50)
$$

During one epoch, the general process is:

$$
\boxed{
\text{Predict}
\rightarrow
\text{Calculate errors}
\rightarrow
\text{Calculate loss}
\rightarrow
\text{Calculate gradients}
\rightarrow
\text{Update parameters}
}
$$

This process is repeated for 1,000 epochs:

$$
\text{Epoch }1\rightarrow(w_1,b_1)
$$

$$
\text{Epoch }2\rightarrow(w_2,b_2)
$$

$$
\text{Epoch }3\rightarrow(w_3,b_3)
$$

and eventually:

$$
\text{Epoch }1000\rightarrow(w_{1000},b_{1000})
$$

Ideally:

$$
w_{1000}\approx10
$$

and:

$$
b_{1000}\approx0
$$

---

## 24. Training History

The training history is a mathematical record of the loss after every epoch.

For example:

$$
\begin{aligned}
L_1 &=843.9588\\
L_2 &=831.7410\\
L_3 &=819.6156\\
L_4 &=807.5841\\
&\vdots\\
L_{1000}&\approx0
\end{aligned}
$$

Where $L_t$ means the loss after epoch $t$.

The first-epoch loss is:

$$
\boxed{L_1}
$$

The final-epoch loss is:

$$
\boxed{L_{1000}}
$$

If:

$$
L_{1000}<L_1
$$

then the model improved during training.

---

## 25. Final MAE Example

Suppose the model learns:

$$
w=9.99
$$

and:

$$
b=0.03
$$

The final formula is:

$$
\hat{Y}=9.99X+0.03
$$

The predictions are:

$$
\begin{aligned}
X=1 &: 9.99(1)+0.03=10.02\\
X=2 &: 9.99(2)+0.03=20.01\\
X=3 &: 9.99(3)+0.03=30.00\\
X=4 &: 9.99(4)+0.03=39.99\\
X=5 &: 9.99(5)+0.03=49.98
\end{aligned}
$$

The absolute errors are:

$$
\begin{aligned}
|10-10.02|&=0.02\\
|20-20.01|&=0.01\\
|30-30.00|&=0.00\\
|40-39.99|&=0.01\\
|50-49.98|&=0.02
\end{aligned}
$$

Therefore:

$$
\operatorname{MAE}
=
\frac{0.02+0.01+0+0.01+0.02}{5}
$$

$$
\operatorname{MAE}=\frac{0.06}{5}
$$

$$
\boxed{\operatorname{MAE}=0.012}
$$

This means the predictions are, on average, only `0.012` units away from the correct answers.

---

## 26. Model Structure and Parameter Count

The mathematical model has:

- One input value.
- One output value.
- One weight.
- One bias.

The formula is:

$$
\hat{Y}=wX+b
$$

Therefore, the total number of parameters is:

$$
1\text{ weight}+1\text{ bias}=2\text{ parameters}
$$

Both parameters can change during training:

$$
\boxed{\text{Trainable parameters}=2}
$$

There are no fixed parameters in this simple model:

$$
\boxed{\text{Non-trainable parameters}=0}
$$

---

## 27. Weight Shape

There is one input and one output. The weight connects the single input to the single output:

$$
X\xrightarrow{w}\hat{Y}
$$

Therefore, the weight can be represented as a $1\times1$ matrix:

$$
W=
\begin{bmatrix}
w
\end{bmatrix}
$$

Its dimensions are:

$$
\boxed{1\times1}
$$

---

## 28. Bias Shape

There is one output, so there is one bias:

$$
B=
\begin{bmatrix}
b
\end{bmatrix}
$$

Its size is:

$$
\boxed{1}
$$

For example:

$$
W=
\begin{bmatrix}
9.99
\end{bmatrix}
$$

and:

$$
B=
\begin{bmatrix}
0.03
\end{bmatrix}
$$

---

## 29. Mathematical Evaluation

Evaluation means using the final weight and bias to calculate predictions for the known examples and then calculating MSE and MAE.

The evaluation formulas are:

$$
\boxed{
\operatorname{MSE}
=
\frac{1}{5}
\sum_{i=1}^{5}
(Y_i-\hat{Y}_i)^2
}
$$

and:

$$
\boxed{
\operatorname{MAE}
=
\frac{1}{5}
\sum_{i=1}^{5}
|Y_i-\hat{Y}_i|
}
$$

No weight or bias is changed during evaluation. Evaluation only measures how well the current formula performs.

---

## 30. Learned Formula

When training finishes, suppose the learned values are:

$$
w=9.99
$$

and:

$$
b=0.03
$$

Then the learned formula is:

$$
\boxed{\hat{Y}=9.99X+0.03}
$$

The ideal formula is:

$$
\boxed{Y=10X}
$$

The learned formula is close to the ideal formula. A learned formula does not always need to contain exactly `10` and `0`. Iterative training usually produces approximate values.

---

## 31. Predicting New Values

After learning the weight and bias, we provide new input values:

$$
X_{\text{new}}=
\begin{bmatrix}
6\\
7\\
8
\end{bmatrix}
$$

Suppose:

$$
w=9.99,\qquad b=0.03
$$

### Prediction for 6

$$
\hat{Y}=9.99(6)+0.03
$$

$$
\hat{Y}=59.94+0.03
$$

$$
\boxed{\hat{Y}=59.97}
$$

The expected answer is:

$$
10(6)=60
$$

### Prediction for 7

$$
\hat{Y}=9.99(7)+0.03
$$

$$
\hat{Y}=69.93+0.03
$$

$$
\boxed{\hat{Y}=69.96}
$$

The expected answer is:

$$
10(7)=70
$$

### Prediction for 8

$$
\hat{Y}=9.99(8)+0.03
$$

$$
\hat{Y}=79.92+0.03
$$

$$
\boxed{\hat{Y}=79.95}
$$

The expected answer is:

$$
10(8)=80
$$

---

## 32. Manual Prediction Check

The mathematical model uses:

$$
\hat{Y}=wX+b
$$

Therefore, manually calculating $(X\times w)+b$ must produce the same result.

Using:

$$
w=9.99,\qquad b=0.03
$$

we get:

$$
(6\times9.99)+0.03=59.97
$$

$$
(7\times9.99)+0.03=69.96
$$

$$
(8\times9.99)+0.03=79.95
$$

This manual check shows that a single-neuron model is performing:

$$
\boxed{\text{Multiplication followed by addition}}
$$

---

## 33. Selected Epoch Losses

For 1,000 epochs, we have:

$$
L_1,L_2,L_3,\ldots,L_{1000}
$$

Instead of inspecting every value, selected epochs can be inspected:

$$
L_1,\ L_{10},\ L_{50},\ L_{100},\ L_{200},\ L_{500},\ L_{1000}
$$

An illustrative sequence could look like:

$$
\begin{aligned}
L_1 &=843.9588\\
L_{10} &=737.4502\\
L_{50} &=377.4169\\
L_{100} &=134.5834\\
L_{200} &=20.2554\\
L_{500} &=0.1000\\
L_{1000} &=0.0001
\end{aligned}
$$

The important pattern is:

$$
L_1>L_{10}>L_{50}>L_{100}>L_{200}>L_{500}>L_{1000}
$$

This pattern shows that the loss is decreasing.

> The values above are illustrative. The exact values depend on the starting parameters and the optimizer updates.

---

## 34. Why Epoch Minus One Is Used to Locate a Recorded Loss

Mathematically, epochs are named:

$$
1,2,3,\ldots,1000
$$

In a zero-based recorded sequence, the positions are:

$$
0,1,2,\ldots,999
$$

Therefore:

$$
\text{position}=\text{epoch}-1
$$

Examples:

$$
\begin{aligned}
\text{Epoch }1 &: 1-1=0\\
\text{Epoch }10 &: 10-1=9\\
\text{Epoch }100 &: 100-1=99\\
\text{Epoch }1000 &: 1000-1=999
\end{aligned}
$$

This is not part of the learning mathematics. It is only the method used to locate a particular recorded loss in a zero-based sequence.

---

## 35. Why Use a Fixed Random Starting Point?

Suppose one training run starts with:

$$
w_0=0.75
$$

Another training run might start with:

$$
w_0=-0.40
$$

Because the starting values are different, the early losses and updates will also be different.

Using the same starting random condition means:

$$
w_0\text{ is the same in repeated runs}
$$

This makes experiments easier to compare.

The chosen seed number does not appear in the prediction formula. It does not mean that the weight is equal to that number. It only identifies a particular repeatable random starting sequence.

---

## 36. Optimizer Iterations

An optimizer iteration means one parameter update.

If all five examples are processed together and the parameters are updated once per epoch, then:

$$
1\text{ epoch}=1\text{ update}
$$

Therefore:

$$
1000\text{ epochs}=1000\text{ updates}
$$

So:

$$
\boxed{\text{Optimizer iterations}=1000}
$$

If the data were divided into multiple groups per epoch, there could be multiple updates in each epoch.

---

## 37. Complete Mathematical Training Cycle

For epoch $t$, the complete process is as follows.

### Step 1: Current Parameters

$$
w_t,\qquad b_t
$$

### Step 2: Calculate Predictions

$$
\hat{Y}_i=w_tX_i+b_t
$$

### Step 3: Calculate Errors

$$
e_i=\hat{Y}_i-Y_i
$$

### Step 4: Calculate MSE

$$
L_t=
\frac{1}{n}
\sum_{i=1}^{n}e_i^2
$$

### Step 5: Calculate Gradients

$$
\frac{\partial L}{\partial w}
=
\frac{2}{n}
\sum_{i=1}^{n}X_ie_i
$$

$$
\frac{\partial L}{\partial b}
=
\frac{2}{n}
\sum_{i=1}^{n}e_i
$$

### Step 6: Adam Processes the Gradients

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

### Step 7: Correct the Moving Averages

$$
\hat{m}_t=\frac{m_t}{1-\beta_1^t}
$$

$$
\hat{v}_t=\frac{v_t}{1-\beta_2^t}
$$

### Step 8: Update Each Parameter

$$
\theta_t
=
\theta_{t-1}
-
\alpha
\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$

### Step 9: Repeat

The cycle continues until the loss becomes sufficiently small or all epochs finish.

---

## 38. All Formulas in One Place

### Prediction Formula

$$
\boxed{\hat{Y}=wX+b}
$$

### Error Formula

$$
\boxed{e_i=\hat{Y}_i-Y_i}
$$

### Mean Squared Error

$$
\boxed{
\operatorname{MSE}
=
\frac{1}{n}
\sum_{i=1}^{n}
(Y_i-\hat{Y}_i)^2
}
$$

### Mean Absolute Error

$$
\boxed{
\operatorname{MAE}
=
\frac{1}{n}
\sum_{i=1}^{n}
|Y_i-\hat{Y}_i|
}
$$

### Weight Gradient

$$
\boxed{
\frac{\partial L}{\partial w}
=
\frac{2}{n}
\sum_{i=1}^{n}
X_i(wX_i+b-Y_i)
}
$$

### Bias Gradient

$$
\boxed{
\frac{\partial L}{\partial b}
=
\frac{2}{n}
\sum_{i=1}^{n}
(wX_i+b-Y_i)
}
$$

### Basic Gradient Descent

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

### Adam First Moment

$$
\boxed{
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
}
$$

### Adam Second Moment

$$
\boxed{
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
}
$$

### Bias-Corrected Moments

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

### Adam Parameter Update

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

---

# Final Very Simple Summary

The model is trying to find two numbers:

$$
w\qquad\text{and}\qquad b
$$

The model uses those numbers in:

$$
\hat{Y}=wX+b
$$

The complete learning process is:

$$
\boxed{\text{Start with initial }w,b}
$$

$$
\downarrow
$$

$$
\boxed{\text{Make predictions}}
$$

$$
\downarrow
$$

$$
\boxed{\text{Compare predictions with correct answers}}
$$

$$
\downarrow
$$

$$
\boxed{\text{Calculate MSE}}
$$

$$
\downarrow
$$

$$
\boxed{\text{Calculate gradients}}
$$

$$
\downarrow
$$

$$
\boxed{\text{Adam updates }w\text{ and }b}
$$

$$
\downarrow
$$

$$
\boxed{\text{Repeat for 1,000 epochs}}
$$

Eventually, the model learns approximately:

$$
w\approx10
$$

and:

$$
b\approx0
$$

Therefore:

$$
\boxed{\hat{Y}\approx10X}
$$

The entire single-neuron model is fundamentally learning one multiplication value and one addition value.
