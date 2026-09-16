# Diabetes Prediction — Full Code Walkthrough


## Step 1: Importing Libraries

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
```

**What's happening:**
- `numpy` — handles numerical arrays. ML models expect data in array form.
- `pandas` — loads and organizes tabular data (rows and columns) so it's easy to inspect and manipulate.
- `StandardScaler` — a tool that rescales numeric features so they're all on a comparable scale.
- `train_test_split` — splits data into a training portion and a testing portion.
- `svm` — the module containing the Support Vector Machine algorithm.
- `accuracy_score` — measures what percentage of predictions the model got right.

**Why this matters:** These imports set up the full toolkit for the project — pandas/numpy for handling data, and scikit-learn (`sklearn`) for scaling, splitting, modeling, and evaluating.

---

## Step 2: Loading the Data

```python
diabetes_dataset = pd.read_csv('/content/diabetes.csv')
diabetes_dataset.head()
```

**What's happening:** `pd.read_csv(...)` reads the CSV file into a pandas DataFrame. Unlike the Sonar project, this file *does* have column headers already (Pregnancies, Glucose, etc.), so there's no need for `header=None`. `.head()` previews the first 5 rows.

**Why this matters — features and labels:** The columns Pregnancies through Age are the **features** (the inputs the model learns from), and the `Outcome` column is the **label** (0 = not diabetic, 1 = diabetic) — the answer the model is trying to predict.

---

## Step 3: Checking the Shape

```python
diabetes_dataset.shape
```

**What's happening:** Returns `(768, 9)` — 768 patients, 9 columns (8 features + 1 label).

**Why this matters:** Knowing your dataset size upfront tells you what to expect from the model. 768 samples is a decent-sized dataset for a project like this — more data than the Sonar project's 208, which generally gives a model more to learn from and reduces overfitting risk.

---

## Step 4: Statistical Summary

```python
diabetes_dataset.describe()
```

**What's happening:** Shows count, mean, std, min, max, and quartiles for every column.

**Why this matters — spotting scale differences:** This is the step where an important issue becomes visible: the features are *not* on the same scale. For example, Glucose ranges from 0–199, while DiabetesPedigreeFunction ranges from about 0.08–2.42. This difference in scale matters a lot for the model we'll use later (SVM), which is sensitive to the relative size of feature values — a detail that will be addressed in Step 7.

---

## Step 5: Checking Class Balance

```python
diabetes_dataset['Outcome'].value_counts()
# 0 = Non Diabetic, 1 = Diabetic
```

**What's happening:** Counts how many patients fall into each Outcome category: 500 non-diabetic (0), 268 diabetic (1).

**Why this matters — class imbalance:** This is *not* a perfectly balanced dataset — about 65% of patients are non-diabetic. This is mild enough that accuracy is still a reasonably useful metric, but it's worth remembering: a model could get roughly 65% accuracy just by always predicting "non-diabetic," so we should judge the model's real 77%+ accuracy in that light — it's doing meaningfully better than a naive guess, but there's room for deeper metrics like precision/recall to fully trust it.

---

## Step 6: Comparing Averages Between Classes

```python
diabetes_dataset.groupby('Outcome').mean()
```

**What's happening:** Groups all rows by Outcome (0 or 1), then averages every feature within each group.

**Why this matters:** This is a quick way to see whether diabetic and non-diabetic patients actually look different in the data — before training anything. In this dataset, diabetic patients (Outcome=1) have a noticeably higher average Glucose (141 vs. 110) and slightly higher BMI, Age, and Pregnancies than non-diabetic patients. This is an encouraging early sign that these features carry a real, learnable signal.

---

## Step 7: Separating Features from Labels

```python
X = diabetes_dataset.drop(columns = 'Outcome', axis = 1)
Y = diabetes_dataset['Outcome']
```

**What's happening:** `X` becomes every column except `Outcome` (the features). `Y` becomes just the `Outcome` column (the label).

**Why this matters:** Same convention as always in supervised learning — `X` holds what you know (patient measurements), `Y` holds what you're trying to predict (diabetic or not).

---

## Step 8: Standardizing the Data

```python
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
```

**What's happening, line by line:**
- `StandardScaler()` creates a scaler object — it doesn't do anything to the data yet.
- `scaler.fit(X)` calculates the mean and standard deviation of *each column* in `X`, without changing the data. Think of this as the scaler "studying" the data to learn how to rescale it.
- `scaler.transform(X)` actually applies the rescaling: for every value, it subtracts that column's mean and divides by that column's standard deviation. The result is that every feature ends up with a mean of 0 and a standard deviation of 1.

```python
X = standardized_data
Y = diabetes_dataset['Outcome']
```
This just reassigns `X` to be the standardized version going forward, and keeps `Y` as-is (labels don't need scaling).

**Why this matters — ML concept, feature scaling:** Recall from Step 4 that Glucose (0–199) and DiabetesPedigreeFunction (0.08–2.42) are on wildly different scales. Many ML algorithms — SVM included — calculate distances or margins between data points mathematically, and if one feature's numbers are naturally hundreds of times larger than another's, it will dominate those calculations even if it isn't actually more important. Standardizing puts every feature on equal footing so the model judges importance based on the *pattern* in the data, not the *units* it happens to be measured in.

**Important detail to remember:** Because the model is trained on standardized data, any *new* data you want to predict on later must go through this exact same scaler before being fed to the model (see Step 13).

---

## Step 9: Train/Test Split

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)
```

**What's happening:**
- `test_size=0.2` — holds back 20% of the data for testing (~154 samples), leaving 80% (~614 samples) for training.
- `stratify=Y` — makes sure the 65/35 non-diabetic/diabetic ratio is preserved in both the training and test sets, so neither ends up accidentally skewed.
- `random_state=2` — fixes the randomness so the split is reproducible every time the code runs.

```python
print(X.shape, X_train.shape, X_test.shape)
```
Confirms the split: `(768, 8) (614, 8) (154, 8)`.

**Why this matters — training vs. testing:** The model must be evaluated on data it has never seen, or its accuracy score would be meaningless — it could just be "remembering" the answers rather than genuinely learning to predict diabetes from patient measurements.

---

## Step 10: Creating and Training the Model

```python
classifier = svm.SVC(kernel = 'linear')
classifier.fit(X_train, Y_train)
```

**What's happening:**
- `svm.SVC(kernel='linear')` creates a Support Vector Classifier that will try to separate the two classes using a **straight-line boundary** (in higher dimensions, a flat "hyperplane") rather than a curved one.
- `classifier.fit(X_train, Y_train)` is the actual training step — the model looks at the 614 training examples and their correct labels, and works out where to draw that boundary line so it best separates diabetic from non-diabetic patients.

**Why this matters — ML concept, what is SVM:** A Support Vector Machine tries to find the boundary that not only separates the two classes, but does so with the **widest possible margin** — the biggest possible gap — between the boundary and the nearest data points of each class (called the "support vectors," which is where the algorithm gets its name). A wider margin generally means the model will generalize better to new, unseen data. The `linear` kernel keeps that boundary a straight line/plane, which works well when the classes are reasonably separable without needing a more complex curved boundary.

---

## Step 11: Checking Accuracy on Training Data

```python
X_train_predicition = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_predicition, Y_train)
print('Accuracy Score of the training data:', training_data_accuracy)
```

**What's happening:** The trained model predicts on the training data itself, and those predictions are compared against the real answers.
- Result: **78.7%** training accuracy.

**Why this matters:** This shows how well the model fits the data it learned from. It's a helpful number, but on its own it doesn't tell you how the model will perform on new patients — that's what the test accuracy is for.

---

## Step 12: Checking Accuracy on Test Data

```python
X_test_predicition = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_predicition, Y_test)
print('Accuracy Score of the test data:', test_data_accuracy)
```

**What's happening:** Same process, but on the 154 test samples the model has never seen.
- Result: **77.3%** test accuracy.

**Why this matters — ML concept, overfitting vs. good generalization:** Training accuracy (78.7%) and test accuracy (77.3%) are very close to each other. This is actually a *good* sign — it means the model isn't overfitting (memorizing the training data at the expense of generalizing). If training accuracy had been, say, 95% while test accuracy stayed around 77%, that gap would signal the model learned quirks specific to the training set rather than a pattern that holds up on new data.

---

## Step 13: Making a Prediction on a New Patient

```python
input_data = (4,110,92,0,0,37.6,0.191,30)

input_data_as_numpy_array = np.asanyarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')
```

**What's happening, line by line:**
- `input_data = (...)` — a new patient's 8 measurements, in the same order as the original columns (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age).
- `np.asanyarray(input_data)` — converts the plain Python tuple into a numpy array, the format scikit-learn expects.
- `.reshape(1, -1)` — reshapes the flat 8-value array into "1 row, 8 columns" — the 2D shape `model.predict()` requires, even for a single example.
- `scaler.transform(input_data_reshaped)` — **this step is essential and easy to forget**: since the model was trained on *standardized* data, any new input must be scaled using the exact same scaler (with the same means/standard deviations learned back in Step 8) before the model can meaningfully compare it to what it learned. Skipping this step would feed the model raw, unscaled numbers it was never trained to interpret correctly.
- `classifier.predict(std_data)` — runs the standardized input through the trained model to get a prediction (`0` or `1`).
- The `if`/`else` block translates that raw number into a readable sentence.

**Why this matters:** This is the payoff of the whole pipeline — using the trained, validated model to make a real prediction on a new patient's data. Notice that the same standardization step from training must be replayed here; forgetting it is one of the most common real-world bugs people run into when deploying a model that was trained on scaled data.

---

## Summary of Key ML Concepts Covered

| Concept | Where it appears | Why it matters |
|---|---|---|
| Features vs. labels | Step 7 | Inputs (X) vs. the answer you're predicting (Y) |
| Class imbalance | Step 5 | A skewed label distribution can make accuracy misleading |
| Feature scaling / standardization | Step 8 | Puts features with different units/ranges on equal footing |
| Train/test split | Step 9 | Lets you fairly measure how well a model generalizes |
| Support Vector Machine (SVM) | Step 10 | Finds the boundary that best separates classes with the widest margin |
| Accuracy | Steps 11–12 | Basic measure of how good a classifier is |
| Overfitting vs. generalization | Steps 11–12 | Comparing train vs. test accuracy reveals whether a model is memorizing or learning |
| Applying the same scaler to new data | Step 13 | New inputs must be transformed exactly like the training data |
