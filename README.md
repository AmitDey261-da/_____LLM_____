# TensorFlow and Keras: Beginner-to-Advanced Practical Reference

> A GitHub-friendly learning guide covering the most important TensorFlow and Keras functions, with simple explanations and working examples.

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Important Note

TensorFlow contains thousands of public APIs, so no single README can meaningfully demonstrate every function. This guide covers the **most important and commonly used functions** from beginner to advanced level. Use the [official TensorFlow API reference](https://www.tensorflow.org/api_docs/python/tf) and [Keras API reference](https://keras.io/api/) for the complete API.

## Table of Contents

1. [Installation and Imports](#1-installation-and-imports)
2. [Tensors](#2-tensors)
3. [Shapes, Dimensions, and Reshaping](#3-shapes-dimensions-and-reshaping)
4. [Data Types and Conversion](#4-data-types-and-conversion)
5. [Tensor Selection and Manipulation](#5-tensor-selection-and-manipulation)
6. [Mathematical Operations](#6-mathematical-operations)
7. [Random Numbers](#7-random-numbers)
8. [Variables and Automatic Differentiation](#8-variables-and-automatic-differentiation)
9. [Building Models with Keras](#9-building-models-with-keras)
10. [Important Layers](#10-important-layers)
11. [Activation Functions](#11-activation-functions)
12. [Compile, Train, Evaluate, and Predict](#12-compile-train-evaluate-and-predict)
13. [Loss Functions](#13-loss-functions)
14. [Optimizers](#14-optimizers)
15. [Metrics](#15-metrics)
16. [Callbacks](#16-callbacks)
17. [Data Pipelines with tf.data](#17-data-pipelines-with-tfdata)
18. [Preprocessing Layers](#18-preprocessing-layers)
19. [Saving and Loading Models](#19-saving-and-loading-models)
20. [Custom Training with GradientTape](#20-custom-training-with-gradienttape)
21. [CNN, RNN, LSTM, and Transfer Learning](#21-cnn-rnn-lstm-and-transfer-learning)
22. [Insurance and Risk Analytics Examples](#22-insurance-and-risk-analytics-examples)
23. [Debugging and Device Functions](#23-debugging-and-device-functions)
24. [Quick Function Cheat Sheet](#24-quick-function-cheat-sheet)
25. [Learning Roadmap](#25-learning-roadmap)

---

## 1. Installation and Imports

### Install TensorFlow

```bash
pip install tensorflow
```

### Import TensorFlow and Keras

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

print("TensorFlow version:", tf.__version__)
```

### Check available CPU and GPU devices

```python
print(tf.config.list_physical_devices())
print("CPUs:", tf.config.list_physical_devices("CPU"))
print("GPUs:", tf.config.list_physical_devices("GPU"))
```

---

## 2. Tensors

A tensor is a container of numbers. A scalar is 0D, a list is 1D, a table is 2D, and an image batch is commonly 4D.

### `tf.constant()`

Creates a tensor whose value is not intended to change.

```python
scalar = tf.constant(10)
vector = tf.constant([1, 2, 3])
matrix = tf.constant([[1, 2], [3, 4]])

print(scalar)
print(vector)
print(matrix)
```

### `tf.Variable()`

Creates a value that can change. Model weights and biases are variables.

```python
weight = tf.Variable(2.0)
print(weight)

weight.assign(10.0)
print(weight)

weight.assign_add(5.0)
print(weight)

weight.assign_sub(3.0)
print(weight)
```

### Common tensor properties

```python
x = tf.constant([[1.0, 2.0], [3.0, 4.0]])

print("Shape:", x.shape)
print("Dimensions:", x.ndim)
print("Data type:", x.dtype)
print("Total elements:", tf.size(x).numpy())
```

---

## 3. Shapes, Dimensions, and Reshaping

### `tf.shape()`

Returns the dynamic shape of a tensor.

```python
x = tf.constant([[1, 2, 3], [4, 5, 6]])
print(tf.shape(x))
```

### `tf.reshape()`

Changes a tensor's arrangement without changing its values.

```python
x = tf.constant([1, 2, 3, 4, 5, 6])

column = tf.reshape(x, (-1, 1))
two_columns = tf.reshape(x, (-1, 2))

print(column)
print(two_columns)
```

`-1` means: TensorFlow should automatically calculate that dimension.

### NumPy version used for model input

```python
X = np.array([1, 2, 3, 4, 5], dtype=np.float32).reshape(-1, 1)
Y = np.array([10, 20, 30, 40, 50], dtype=np.float32).reshape(-1, 1)

print(X)
print(X.shape)  # (5, 1) means 5 records and 1 feature
```

### `tf.expand_dims()` and `tf.squeeze()`

```python
x = tf.constant([1, 2, 3])
expanded = tf.expand_dims(x, axis=1)
restored = tf.squeeze(expanded)

print(expanded.shape)  # (3, 1)
print(restored.shape)  # (3,)
```

### `tf.transpose()`

Swaps rows and columns.

```python
x = tf.constant([[1, 2, 3], [4, 5, 6]])
print(tf.transpose(x))
```

---

## 4. Data Types and Conversion

### `tf.cast()`

Converts one data type to another.

```python
x = tf.constant([1, 2, 3], dtype=tf.int32)
x_float = tf.cast(x, tf.float32)

print(x.dtype)
print(x_float.dtype)
```

### `tf.convert_to_tensor()`

Converts lists, NumPy arrays, or compatible values into tensors.

```python
numpy_data = np.array([10, 20, 30], dtype=np.float32)
tensor_data = tf.convert_to_tensor(numpy_data)
print(tensor_data)
```

### `.numpy()`

Converts an eager tensor to a NumPy value.

```python
x = tf.constant([1, 2, 3])
print(x.numpy())
```

### Common TensorFlow data types

- `tf.float32`: standard neural-network decimal type
- `tf.float64`: higher-precision decimal type
- `tf.int32`: integer type
- `tf.int64`: larger integer type
- `tf.bool`: `True` or `False`
- `tf.string`: text values

---

## 5. Tensor Selection and Manipulation

### Indexing and slicing

```python
x = tf.constant([10, 20, 30, 40, 50])

print(x[0])
print(x[-1])
print(x[1:4])
```

### `tf.concat()`

Joins tensors along an existing axis.

```python
a = tf.constant([[1, 2]])
b = tf.constant([[3, 4]])

print(tf.concat([a, b], axis=0))
print(tf.concat([a, b], axis=1))
```

### `tf.stack()`

Joins tensors by creating a new axis.

```python
a = tf.constant([1, 2])
b = tf.constant([3, 4])
print(tf.stack([a, b], axis=0))
```

### `tf.split()`

```python
x = tf.constant([1, 2, 3, 4, 5, 6])
parts = tf.split(x, num_or_size_splits=3)
print(parts)
```

### `tf.gather()`

Selects values using index positions.

```python
x = tf.constant([100, 200, 300, 400])
print(tf.gather(x, [0, 3]))
```

### `tf.where()`

Returns values based on a condition.

```python
losses = tf.constant([100.0, 5000.0, 250.0])
capped = tf.where(losses > 1000, 1000.0, losses)
print(capped)
```

### `tf.one_hot()`

Converts category numbers into one-hot vectors.

```python
peril_ids = tf.constant([0, 2, 1])
encoded = tf.one_hot(peril_ids, depth=3)
print(encoded)
```

---

## 6. Mathematical Operations

```python
a = tf.constant([1.0, 2.0, 3.0])
b = tf.constant([4.0, 5.0, 6.0])

print(tf.add(a, b))
print(tf.subtract(a, b))
print(tf.multiply(a, b))
print(tf.divide(a, b))
print(tf.square(a))
print(tf.sqrt(a))
print(tf.pow(a, 2))
print(tf.abs(tf.constant([-5.0, 2.0])))
```

### Aggregation functions

```python
x = tf.constant([10.0, 20.0, 30.0])

print("Sum:", tf.reduce_sum(x))
print("Mean:", tf.reduce_mean(x))
print("Minimum:", tf.reduce_min(x))
print("Maximum:", tf.reduce_max(x))
print("Product:", tf.reduce_prod(x))
```

### `tf.argmax()` and `tf.argmin()`

```python
probabilities = tf.constant([0.10, 0.75, 0.15])
print("Highest position:", tf.argmax(probabilities).numpy())
print("Lowest position:", tf.argmin(probabilities).numpy())
```

### Matrix multiplication

```python
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0], [6.0]])

print(tf.matmul(a, b))
print(a @ b)
```

### `tf.math` examples

```python
x = tf.constant([1.0, 2.0, 3.0])

print(tf.math.log(x))
print(tf.math.exp(x))
print(tf.math.sigmoid(x))
print(tf.math.round(tf.constant([1.2, 1.8])))
```

---

## 7. Random Numbers

### Reproducibility

```python
tf.random.set_seed(42)
```

### Random uniform and normal values

```python
uniform_values = tf.random.uniform(shape=(2, 3), minval=0, maxval=1)
normal_values = tf.random.normal(shape=(2, 3), mean=0, stddev=1)

print(uniform_values)
print(normal_values)
```

### Shuffle values

```python
x = tf.constant([1, 2, 3, 4, 5])
print(tf.random.shuffle(x))
```

---

## 8. Variables and Automatic Differentiation

### `tf.GradientTape()`

Records calculations so TensorFlow can calculate gradients.

```python
w = tf.Variable(2.0)

with tf.GradientTape() as tape:
    y = w ** 2

gradient = tape.gradient(y, w)
print(gradient)  # 4 because the derivative of w^2 is 2w
```

### Simple weight update

```python
w = tf.Variable(1.0)
learning_rate = 0.01

with tf.GradientTape() as tape:
    prediction = w * 5.0
    loss = (prediction - 50.0) ** 2

gradient = tape.gradient(loss, w)
w.assign_sub(learning_rate * gradient)

print("Updated weight:", w.numpy())
```

### `@tf.function`

Converts a Python function into a TensorFlow graph for reusable, optimized execution.

```python
@tf.function
def multiply_by_ten(x):
    return x * 10

print(multiply_by_ten(tf.constant(6)))
```

---

## 9. Building Models with Keras

Keras supports three main model-building styles.

### Sequential API

Use when layers follow one after another.

```python
model = keras.Sequential([
    keras.Input(shape=(4,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1)
])
```

### Functional API

Use for multiple inputs, multiple outputs, shared layers, or branch connections.

```python
inputs = keras.Input(shape=(4,), name="policy_features")
x = keras.layers.Dense(16, activation="relu")(inputs)
x = keras.layers.Dense(8, activation="relu")(x)
outputs = keras.layers.Dense(1, name="predicted_loss")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
```

### Model subclassing

Use when the model needs custom Python logic.

```python
class RiskModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.hidden = keras.layers.Dense(16, activation="relu")
        self.output_layer = keras.layers.Dense(1)

    def call(self, inputs):
        x = self.hidden(inputs)
        return self.output_layer(x)

model = RiskModel()
```

### Inspect a model

```python
model.summary()
print("Layers:", model.layers)
print("Trainable variables:", model.trainable_variables)
```

---

## 10. Important Layers

### Dense layer

```python
keras.layers.Dense(64, activation="relu")
```

A Dense neuron calculates approximately:

```text
output = activation((input x weight) + bias)
```

### Dropout

Randomly disables a percentage of units during training to reduce overfitting.

```python
keras.layers.Dropout(0.30)
```

### Batch normalization

```python
keras.layers.BatchNormalization()
```

### Flatten

Converts multidimensional data into one vector per record.

```python
keras.layers.Flatten()
```

### Embedding

Converts integer tokens or category IDs into learned vectors.

```python
keras.layers.Embedding(input_dim=10000, output_dim=64)
```

### Convolution and pooling

```python
keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu")
keras.layers.MaxPooling2D(pool_size=(2, 2))
keras.layers.GlobalAveragePooling2D()
```

### Recurrent layers

```python
keras.layers.SimpleRNN(32)
keras.layers.LSTM(64)
keras.layers.GRU(64)
keras.layers.Bidirectional(keras.layers.LSTM(32))
```

---

## 11. Activation Functions

### ReLU

Commonly used in hidden layers.

```python
keras.layers.Dense(32, activation="relu")
```

### Sigmoid

Produces a value between 0 and 1 for binary probability.

```python
keras.layers.Dense(1, activation="sigmoid")
```

### Softmax

Produces probabilities across multiple classes.

```python
keras.layers.Dense(3, activation="softmax")
```

### Other activations

```python
keras.layers.Dense(32, activation="tanh")
keras.layers.Dense(32, activation="elu")
keras.layers.LeakyReLU(negative_slope=0.01)
```

---

## 12. Compile, Train, Evaluate, and Predict

### Complete single-neuron example: learn `Y = 10X`

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Create reproducible results.
tf.keras.utils.set_random_seed(42)

# Five records and one feature.
X = np.array([1, 2, 3, 4, 5], dtype=np.float32).reshape(-1, 1)
Y = np.array([10, 20, 30, 40, 50], dtype=np.float32).reshape(-1, 1)

# One input feature and one output neuron.
model = keras.Sequential([
    keras.Input(shape=(1,)),
    keras.layers.Dense(1)
])

# Configure training.
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.05),
    loss="mse",
    metrics=["mae"]
)

# Train and save the training history.
history = model.fit(
    X,
    Y,
    epochs=1000,
    verbose=0
)

# Evaluate known records.
loss, mae = model.evaluate(X, Y, verbose=0)
print("MSE:", loss)
print("MAE:", mae)

# Predict new records.
new_X = np.array([6, 7, 8], dtype=np.float32).reshape(-1, 1)
predictions = model.predict(new_X, verbose=0)
print("Predictions:", predictions.flatten())

# Read learned weight and bias.
weight, bias = model.layers[0].get_weights()
print("Weight:", weight[0, 0])
print("Bias:", bias[0])
print(f"Learned formula: Y = ({weight[0, 0]:.6f} x X) + ({bias[0]:.6f})")
```

### Main model methods

- `model.compile()`: defines optimizer, loss, and metrics
- `model.fit()`: trains the model
- `model.evaluate()`: measures performance without training
- `model.predict()`: generates predictions
- `model.summary()`: displays the architecture and parameter count
- `model.get_weights()`: returns model parameters
- `model.set_weights()`: replaces model parameters

### Read training history

```python
print(history.history.keys())

for epoch, loss_value in enumerate(history.history["loss"], start=1):
    print(f"Epoch {epoch:4d}: Loss = {loss_value:.6f}")
```

### Validation split

```python
history = model.fit(
    X,
    Y,
    epochs=100,
    validation_split=0.20,
    batch_size=32,
    shuffle=True,
    verbose=1
)
```

---

## 13. Loss Functions

### Regression losses

```python
model.compile(optimizer="adam", loss="mse")
model.compile(optimizer="adam", loss="mae")
model.compile(optimizer="adam", loss=keras.losses.Huber())
```

- MSE heavily penalizes large errors.
- MAE represents average absolute distance.
- Huber combines useful behavior from MSE and MAE.

### Binary classification

```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

### Multiclass classification

```python
# Integer labels such as 0, 1, and 2
loss = "sparse_categorical_crossentropy"

# One-hot labels such as [1, 0, 0]
loss = "categorical_crossentropy"
```

---

## 14. Optimizers

```python
keras.optimizers.Adam(learning_rate=0.001)
keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
keras.optimizers.RMSprop(learning_rate=0.001)
keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001)
```

### Gradient clipping

Useful when gradients become excessively large.

```python
optimizer = keras.optimizers.Adam(
    learning_rate=0.001,
    clipnorm=1.0
)
```

---

## 15. Metrics

### Regression

```python
metrics = [
    keras.metrics.MeanAbsoluteError(name="mae"),
    keras.metrics.RootMeanSquaredError(name="rmse")
]
```

### Classification

```python
metrics = [
    keras.metrics.BinaryAccuracy(name="accuracy"),
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),
    keras.metrics.AUC(name="auc")
]
```

Accuracy alone can be misleading for highly imbalanced fraud or claim datasets. Precision, recall, AUC, and the confusion matrix can provide additional information.

---

## 16. Callbacks

### Early stopping

Stops training when validation performance stops improving.

```python
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)
```

### Model checkpoint

```python
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath="best_model.keras",
    monitor="val_loss",
    save_best_only=True
)
```

### Reduce learning rate

```python
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6
)
```

### TensorBoard and CSV log

```python
tensorboard = keras.callbacks.TensorBoard(log_dir="logs")
csv_logger = keras.callbacks.CSVLogger("training_history.csv")
```

### Use callbacks

```python
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_validation, y_validation),
    epochs=200,
    callbacks=[early_stopping, checkpoint, reduce_lr, csv_logger]
)
```

---

## 17. Data Pipelines with `tf.data`

`tf.data.Dataset` creates scalable input pipelines.

### Create a dataset

```python
X = np.array([[1], [2], [3], [4]], dtype=np.float32)
y = np.array([[10], [20], [30], [40]], dtype=np.float32)

dataset = tf.data.Dataset.from_tensor_slices((X, y))
```

### Shuffle, batch, map, cache, and prefetch

```python
dataset = (
    dataset
    .shuffle(buffer_size=len(X))
    .batch(2)
    .map(lambda features, target: (features, target))
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

for features, target in dataset:
    print(features, target)
```

### Train from a dataset

```python
model.fit(dataset, epochs=100)
```

Important methods include:

- `from_tensor_slices()`
- `map()`
- `filter()`
- `shuffle()`
- `batch()`
- `repeat()`
- `take()`
- `skip()`
- `cache()`
- `prefetch()`

---

## 18. Preprocessing Layers

### Numerical normalization

```python
normalizer = keras.layers.Normalization(axis=-1)
normalizer.adapt(X_train)

model = keras.Sequential([
    keras.Input(shape=(X_train.shape[1],)),
    normalizer,
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1)
])
```

### String categories

```python
county_lookup = keras.layers.StringLookup(output_mode="one_hot")
county_lookup.adapt(county_values)
```

### Integer categories

```python
class_lookup = keras.layers.IntegerLookup(output_mode="one_hot")
class_lookup.adapt(class_values)
```

### Text vectorization

```python
vectorizer = keras.layers.TextVectorization(
    max_tokens=10000,
    output_mode="int",
    output_sequence_length=100
)
vectorizer.adapt(claim_descriptions)
```

---

## 19. Saving and Loading Models

### Save and load the complete Keras model

```python
model.save("risk_model.keras")
loaded_model = keras.models.load_model("risk_model.keras")
```

### Save and load weights only

```python
model.save_weights("risk_model.weights.h5")
model.load_weights("risk_model.weights.h5")
```

### Export for serving

```python
model.export("exported_risk_model")
```

### Low-level SavedModel API

```python
tf.saved_model.save(model, "saved_model_directory")
loaded = tf.saved_model.load("saved_model_directory")
```

Only load models from trusted sources because serialized models may represent executable computation.

---

## 20. Custom Training with `GradientTape`

```python
model = keras.Sequential([
    keras.Input(shape=(1,)),
    keras.layers.Dense(1)
])

optimizer = keras.optimizers.Adam(learning_rate=0.05)
loss_function = keras.losses.MeanSquaredError()

X = tf.constant([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = tf.constant([[10.0], [20.0], [30.0], [40.0], [50.0]])

for epoch in range(1000):
    with tf.GradientTape() as tape:
        predictions = model(X, training=True)
        loss = loss_function(y, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}: Loss = {loss.numpy():.6f}")
```

This manually performs prediction, loss calculation, gradient calculation, and parameter updating.

---

## 21. CNN, RNN, LSTM, and Transfer Learning

### CNN for image classification

```python
cnn_model = keras.Sequential([
    keras.Input(shape=(128, 128, 3)),
    keras.layers.Rescaling(1.0 / 255),
    keras.layers.Conv2D(32, 3, activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(1, activation="sigmoid")
])
```

Possible use: classify roof images as damaged or not damaged.

### LSTM for time series

```python
lstm_model = keras.Sequential([
    keras.Input(shape=(12, 5)),
    keras.layers.LSTM(64),
    keras.layers.Dense(1)
])
```

Possible use: use 12 months and 5 features per month to forecast next month's claim count.

### Text classification

```python
text_model = keras.Sequential([
    keras.Input(shape=(100,)),
    keras.layers.Embedding(input_dim=10000, output_dim=64),
    keras.layers.Bidirectional(keras.layers.LSTM(32)),
    keras.layers.Dense(1, activation="sigmoid")
])
```

Possible use: classify claim descriptions into claim or no-claim review categories.

### Transfer learning

```python
base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

inputs = keras.Input(shape=(224, 224, 3))
x = keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(1, activation="sigmoid")(x)

transfer_model = keras.Model(inputs, outputs)
```

---

## 22. Insurance and Risk Analytics Examples

### Example A: Claim severity prediction

Possible features:

- Total Insured Value
- Year built
- Roof age
- Distance to coast
- Construction type
- Occupancy type
- Protection class

Target:

- Historical claim amount or expected loss

```python
claim_model = keras.Sequential([
    keras.Input(shape=(7,)),
    keras.layers.Normalization(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.20),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1)
])

claim_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.Huber(),
    metrics=[keras.metrics.MeanAbsoluteError(name="mae")]
)
```

### Example B: Claim occurrence prediction

```python
claim_probability_model = keras.Sequential([
    keras.Input(shape=(10,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])

claim_probability_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        keras.metrics.AUC(name="auc"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall")
    ]
)
```

### Example C: Data-quality anomaly score

A model may learn normal combinations of fields, but deterministic business validations should generally remain explicit rules. Machine learning can supplement those rules by flagging unusual records for review.

### Example D: Portfolio workflow

```text
Policy or location data
        |
        v
Clean missing and invalid values
        |
        v
Encode category fields
        |
        v
Normalize numerical fields
        |
        v
Split into train, validation, and test sets
        |
        v
Train model
        |
        v
Evaluate business and statistical performance
        |
        v
Score new portfolio records
```

Never train with personally identifiable or confidential portfolio data in a public notebook or repository.

---

## 23. Debugging and Device Functions

### Check values for NaN or infinity

```python
x = tf.constant([1.0, 2.0, 3.0])
tf.debugging.check_numerics(x, "Invalid numerical values found")
```

### Assertions

```python
tf.debugging.assert_equal(tf.shape(X)[0], tf.shape(y)[0])
tf.debugging.assert_rank(X, 2)
tf.debugging.assert_all_finite(X, "X contains NaN or infinity")
```

### Execute on a selected device

```python
with tf.device("/CPU:0"):
    result = tf.matmul(
        tf.constant([[1.0, 2.0]]),
        tf.constant([[3.0], [4.0]])
    )

print(result)
```

### Clear Keras session

Useful when repeatedly creating models in a notebook.

```python
keras.backend.clear_session()
```

---

## 24. Quick Function Cheat Sheet

### Tensor creation

- `tf.constant()` creates fixed tensor values.
- `tf.Variable()` creates changeable values.
- `tf.zeros()` creates zeros.
- `tf.ones()` creates ones.
- `tf.fill()` fills a shape with one value.
- `tf.range()` creates a numeric sequence.
- `tf.linspace()` creates evenly spaced values.
- `tf.eye()` creates an identity matrix.

### Shape operations

- `tf.shape()` reads dynamic shape.
- `tf.reshape()` rearranges dimensions.
- `tf.expand_dims()` adds a dimension.
- `tf.squeeze()` removes size-one dimensions.
- `tf.transpose()` reorders dimensions.
- `tf.broadcast_to()` broadcasts values to a target shape.

### Data manipulation

- `tf.concat()` joins along an existing dimension.
- `tf.stack()` joins along a new dimension.
- `tf.split()` divides a tensor.
- `tf.gather()` selects indexed values.
- `tf.where()` selects according to a condition.
- `tf.unique()` returns unique values.
- `tf.sort()` sorts values.
- `tf.argsort()` returns sorted index positions.

### Mathematics

- `tf.add()`, `tf.subtract()`, `tf.multiply()`, `tf.divide()`
- `tf.matmul()` performs matrix multiplication.
- `tf.reduce_sum()`, `tf.reduce_mean()`, `tf.reduce_min()`, `tf.reduce_max()`
- `tf.square()`, `tf.sqrt()`, `tf.pow()`, `tf.abs()`
- `tf.argmax()`, `tf.argmin()`

### Model workflow

- `keras.Input()` defines input shape.
- `keras.Sequential()` creates a sequential stack.
- `keras.Model()` creates a Functional or subclassed model.
- `model.compile()` configures learning.
- `model.fit()` trains.
- `model.evaluate()` measures.
- `model.predict()` predicts.
- `model.summary()` describes the model.
- `model.save()` saves the model.
- `keras.models.load_model()` loads the model.

### Training utilities

- `tf.GradientTape()` records operations for gradients.
- `tape.gradient()` calculates gradients.
- `optimizer.apply_gradients()` updates variables.
- `tf.function()` creates a traced TensorFlow function.
- `tf.data.Dataset` builds data pipelines.

---

## 25. Learning Roadmap

### Beginner

1. Tensors and data types
2. Shapes and reshaping
3. Mathematical operations
4. Dense layers
5. `compile()`, `fit()`, `evaluate()`, and `predict()`
6. Regression and binary classification

### Intermediate

1. Train, validation, and test datasets
2. Normalization and category encoding
3. Callbacks and regularization
4. Functional API
5. `tf.data` pipelines
6. Saving and loading models
7. Model diagnostics and imbalanced-data metrics

### Advanced

1. Custom layers and models
2. Custom training loops
3. CNNs
4. RNNs, GRUs, and LSTMs
5. Attention and transformers
6. Transfer learning and fine-tuning
7. Distributed training
8. TensorFlow Lite, TensorFlow Serving, and production monitoring

---

## Official References

- [TensorFlow Python API](https://www.tensorflow.org/api_docs/python/tf)
- [Keras Guide](https://www.tensorflow.org/guide/keras)
- [Keras API](https://keras.io/api/)
- [`tf.data` Guide](https://www.tensorflow.org/guide/data)
- [Save and Load Keras Models](https://www.tensorflow.org/guide/keras/serialization_and_saving)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)

## Repository Suggestion

```text
TensorFlow-Learning/
|
|-- README.md
|-- notebooks/
|   |-- 01_tensor_basics.ipynb
|   |-- 02_single_neuron_regression.ipynb
|   |-- 03_binary_classification.ipynb
|   |-- 04_claim_severity_model.ipynb
|
|-- data/
|   |-- sample_policy_data.csv
|
|-- models/
|   |-- README.md
|
|-- requirements.txt
|-- .gitignore
|-- LICENSE
```

## Final Summary

TensorFlow performs numerical operations using tensors. Keras provides the main high-level model-building workflow:

```text
Prepare data
    -> Build model
    -> Compile model
    -> Train model
    -> Evaluate model
    -> Predict new values
    -> Save and deploy model
```

Start with small regression and classification projects. Then move to `tf.data`, CNNs, LSTMs, transfer learning, and custom training only when a real project requires those tools.
