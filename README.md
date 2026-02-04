KNN Classification on Sklearn Digits Dataset
This project builds a K‑Nearest Neighbors (KNN) classifier to recognize handwritten digits using the built‑in load_digits dataset from scikit‑learn.

1. Project overview
Task: Multi‑class classification of digits 0–9 from small grayscale images.

Algorithm: K‑Nearest Neighbors with different values of K (3, 5, 7, 9).

Goal: Understand distance‑based classification, the need for feature scaling, and how to tune K using accuracy and a confusion matrix.

2. Dataset details
Source: sklearn.datasets.load_digits() (no external download needed).

Each sample: 8×8 image of a handwritten digit (0–9) flattened into 64 features.

Number of classes: 10 (digits 0–9).

Total samples: 1797 images.

In code, the main attributes are:

digits.data → feature matrix of shape (1797, 64).

digits.target → labels of shape (1797,).
​

3. Tech stack
Language: Python

Libraries:

scikit-learn (datasets, model_selection, preprocessing, neighbors, metrics)

matplotlib (plots)

numpy

4. Workflow / steps
Notebook: notebooks/knn_digits.ipynb (file name as used in your repo).

Main steps implemented:

Load dataset

Use load_digits() to get X (features) and y (labels).

Explore and visualize

Print shapes of X and y.

Display a few 8×8 images with their labels using matplotlib to confirm that data and labels match.

Train–test split

Split into training and test sets (e.g., 80% train, 20% test) using train_test_split with stratify=y to keep class balance.
​

Feature scaling

Apply StandardScaler to standardize features before KNN because it is distance‑based and sensitive to feature scales.

Fit scaler on training data, transform both training and test sets.

Train baseline KNN (K=3)

Train KNeighborsClassifier(n_neighbors=3) on scaled training data.

Predict on test set and compute accuracy using accuracy_score.
​

Tune K (3, 5, 7, 9)

Loop over K values [3, 5, 7, 9].

For each K, train KNN, predict on test data, record accuracy.

Store results in lists and print them.

Accuracy vs K plot

Create a line plot with K on the x‑axis and accuracy on the y‑axis.

Use markers to highlight each point and choose the best K based on highest accuracy.

Confusion matrix

Train the best KNN model (using the K with highest test accuracy).

Plot confusion matrix using confusion_matrix and ConfusionMatrixDisplay to inspect misclassified digits.
​

Sample predictions

Show 5 test images with predicted and true labels to visually confirm model performance.

5. Results
Replace the example values with your actual numbers once you run the notebook.

Best K: K = 3 (example)

Test accuracy (K=3): 0.98 (example)
​

Accuracies for different K values (example):

K	Accuracy
3	0.98
5	0.97
7	0.975
9	0.965
Plots (saved in plots/ folder if you used plt.savefig):

accuracy_vs_k.png – Accuracy vs K line chart.

confusion_matrix.png – Confusion matrix for best K.

6. How KNN works (short explanation)
KNN is a distance‑based algorithm: for a new point, it finds the K closest training samples using a distance metric (often Euclidean) and assigns the majority class among them.

No explicit training phase: it is a lazy learner, storing all training data and doing most of the work at prediction time.
​

Choice of K:

Very small K → high variance, sensitive to noise and overfitting.

Very large K → smoother decision boundary, but can underfit and mix classes.

7. Why scaling is important for KNN
KNN relies on distances between feature vectors; features with larger numeric ranges can dominate the distance calculation.

Standardizing features (zero mean, unit variance) ensures each feature contributes more fairly to the distance, usually improving performance and stability.
​

8. Limitations of KNN
Prediction is slow on large datasets because it must compute distance to many training points.

Performance degrades in high‑dimensional spaces (curse of dimensionality).
​

Sensitive to irrelevant or noisy features and to how you scale the data.

Need to choose K and distance metric carefully, often using validation or cross‑validation.

9. How to run this project
Clone the repository:


git clone <https://github.com/PranithaBokketi/task10-Handwritten-digit-classification>


Install dependencies:


pip install scikit-learn matplotlib numpy
Launch Jupyter:


jupyter notebook
primary sklearn digits.ipynb and run all cells.
