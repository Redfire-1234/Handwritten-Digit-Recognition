# 1. Data aquisation and loading
!wget https://huggingface.co/datasets/Redfire-1234/mnist/resolve/main/mnist.zip -O mnist.zip    


import pandas as pd
import zipfile

zip_path = "mnist.zip"

with zipfile.ZipFile(zip_path) as z:
    with z.open("mnist_train.csv") as f:
        train_df = pd.read_csv(f)

    with z.open("mnist_test.csv") as f:
        test_df = pd.read_csv(f)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

print(train_df['label'].value_counts())

# 2. Exploratory data analysis
#2.1 Visualize Sample Images


import matplotlib.pyplot as plt

# Select first 5 samples
samples = train_df.iloc[:5]

plt.figure(figsize=(10, 3))

for i in range(5):
    image = samples.iloc[i, 1:].values.reshape(28, 28)
    label = samples.iloc[i, 0]

    plt.subplot(1, 5, i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis('off')

plt.show()

# 2.2 Data Quality Check

# Check missing values in training and test datasets
print(train_df.isnull().sum().sum())
print(test_df.isnull().sum().sum())

# 3. Data preprocessing
# 3.1 Feature-Label Separation and Normalization

# Separate features and labels
X_train = train_df.drop('label', axis=1).values
y_train = train_df['label'].values

X_test = test_df.drop('label', axis=1).values
y_test = test_df['label'].values

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# 3.2 Train-Test Split

from sklearn.model_selection import train_test_split

X = train_df.drop('label', axis=1).values / 255.0
y = train_df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3.3 Dimensionality Reduction with PCA
from sklearn.decomposition import PCA

# Apply PCA
pca = PCA(n_components=50)   # you can change components (e.g., 30, 50, 100)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# 4. Model training and hyperparameter tuning
# 4.1 K-Nearest Neighbors (KNN)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

k_values = [3, 5, 7, 9]
accuracies = {}

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_pca, y_train)
    
    y_pred = knn.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)
    
    accuracies[k] = acc
    print(f"k = {k}, Accuracy = {acc:.4f}")

best_k = max(accuracies, key=accuracies.get)
print("Best k:", best_k)
print("Best Accuracy:", accuracies[best_k])

# 4.2 Support Vector Machine (SVM)
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

C_values = [0.1, 1, 10]
gamma_values = ['scale', 0.01, 0.001]

best_acc = 0
best_params = {}

for C in C_values:
    for gamma in gamma_values:
        svm = SVC(kernel='rbf', C=C, gamma=gamma)
        svm.fit(X_train_pca, y_train)
        
        y_pred = svm.predict(X_test_pca)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"C={C}, gamma={gamma}, Accuracy={acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_params = {'C': C, 'gamma': gamma}

print("\nBest Parameters:", best_params)
print("Best Accuracy:", best_acc)

# 4.3 Decision Tree

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

max_depth_values = [5, 10, 15, 20]
min_samples_split_values = [2, 5, 10]

best_acc = 0
best_params = {}

for depth in max_depth_values:
    for min_samples in min_samples_split_values:
        dt = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_split=min_samples,
            random_state=42
        )
        
        dt.fit(X_train_pca, y_train)
        y_pred = dt.predict(X_test_pca)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"max_depth={depth}, min_samples_split={min_samples}, Accuracy={acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_params = {
                'max_depth': depth,
                'min_samples_split': min_samples
            }

print("\nBest Parameters:", best_params)
print("Best Accuracy:", best_acc)


# 5. Model evaluation
# 5.1 Final Accuracy Comparison

from sklearn.metrics import accuracy_score

# KNN Accuracy
y_pred_knn = knn.predict(X_test_pca)
acc_knn = accuracy_score(y_test, y_pred_knn)
print("KNN Accuracy:", acc_knn)

# SVM Accuracy
y_pred_svm = svm.predict(X_test_pca)
acc_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy:", acc_svm)

# Decision Tree Accuracy
y_pred_dt = dt.predict(X_test_pca)
acc_dt = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Accuracy:", acc_dt)

# 5.2 Confusion Matrix Visualization

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Function to plot confusion matrix
def plot_conf_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# KNN
plot_conf_matrix(y_test, y_pred_knn, "KNN")

# SVM
plot_conf_matrix(y_test, y_pred_svm, "SVM")

# Decision Tree
plot_conf_matrix(y_test, y_pred_dt, "Decision Tree")

# 5.3 Error Analysis: Misclassified Examples

import numpy as np

# Find misclassified indices
misclassified_idx = np.where(y_test != y_pred_knn)[0]

# Select first 5 misclassified images
plt.figure(figsize=(10, 3))
for i, idx in enumerate(misclassified_idx[:5]):
    image = X_test[idx].reshape(28, 28)  # reshape pixels
    true_label = y_test[idx]
    pred_label = y_pred_knn[idx]
    
    plt.subplot(1, 5, i+1)
    plt.imshow(image, cmap='gray')
    plt.title(f"True:{true_label}\nPred:{pred_label}")
    plt.axis('off')
plt.show()

By seeing the images and there prediction one can see that the miscalssification happened because of the poor writing, writing style and poor slanting of digits.
Not to ignore that digits are 28x28 pixels which can lose fine details that leads to error. There are some small stroks on the digits that the model is unable to classify.

# 5.4 Performance Comparison Summary

import pandas as pd

# Store accuracies
model_accuracies = {
    "Model": ["KNN", "SVM", "Decision Tree"],
    "Accuracy": [acc_knn, acc_svm, acc_dt]  # values computed earlier
}

# Create DataFrame for a neat table
accuracy_df = pd.DataFrame(model_accuracies)
print("Model Performance Comparison:")
display(accuracy_df)

import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
plt.bar(accuracy_df["Model"], accuracy_df["Accuracy"], color=['skyblue','lightgreen','salmon'])
plt.ylim(0,1)
plt.ylabel("Accuracy")
plt.title("Model Performance Comparison")
plt.show()

PERFORMANCE ANALYSIS:

Amonf the three classical machine learning models evaluated, KNN achieved achieved the highest accuracy with the 97.3% followed by SVM with 94.9% of accuracy where as decision tree achived the lowest accuracy with the 84.2%. KNN has a benefit from the promiximity based classification which works good on handwritten digits as similar digits cluster closely in pixels space. SVM also performed well and very close to KNN due to its RBF kernel, which can seperate non linear patterns. Decision tree has the lowest accuracy percentage which is not even close in comparision with the other Classification models. Decision tree got so much of low accuracy can be likely due to overfitting and dificulty of generalization across handwritten styles.


Misclassified digits show a pattern that all the digits which are misclassified shows similar pairs such 1 and 7, 2 and 7. We can say the most misclassfied digit is 7 that pairs with 1 and 2 and somwtimes with 4 and 9 also. Becuase of the small strokes and variations in the digits make the classification challenging.


To improve the performance of the classification models PCA, feature scaling and data augumentation can be applied that can help model to generalize better and reduce misclassification.

# 7. Ensemble method
# 7.1 Voting Classifier

from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

# Use already trained models
voting_clf = VotingClassifier(
    estimators=[
        ('knn', knn),
        ('svm', svm),
        ('dt', dt)
    ],
    voting='hard'   # majority voting
)

# Fit ensemble
voting_clf.fit(X_train_pca, y_train)

# Predict
y_pred_ensemble = voting_clf.predict(X_test_pca)

# Accuracy
acc_ensemble = accuracy_score(y_test, y_pred_ensemble)
print("Voting Ensemble Accuracy:", acc_ensemble)

ENSEMBLE PERFORMANCE:

A voting ensemble combining SVM, KNN, and decision tree classifier achieved the accuracy of 96.1%. That makes the conclusion that enseble method was not able to outperform the KNN 97.3% accuracy. But here one can say ensemble method provided more stable and robust prediction by aggregating the outputs from multiple classifier. The reduced performence can be by the weaker decision tree model, which influenced majority voting in some cases.
This demonstrate that the ensemble model are effective when case models have comparable performence level

# 8. PCA impact analysis
# 8.1 Models Without PCA

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# KNN without PCA
knn_no_pca = KNeighborsClassifier(n_neighbors=best_k)
knn_no_pca.fit(X_train, y_train)
y_pred_knn_no_pca = knn_no_pca.predict(X_test)
acc_knn_no_pca = accuracy_score(y_test, y_pred_knn_no_pca)

# SVM without PCA (use SAME best hyperparameters manually)
svm_no_pca = SVC(kernel='rbf', C=1, gamma='scale')  # use best found values
svm_no_pca.fit(X_train, y_train)
y_pred_svm_no_pca = svm_no_pca.predict(X_test)
acc_svm_no_pca = accuracy_score(y_test, y_pred_svm_no_pca)

print("KNN Accuracy without PCA:", acc_knn_no_pca)
print("SVM Accuracy without PCA:", acc_svm_no_pca)


# 8.2 PCA Impact Comparison

import pandas as pd

comparison_df = pd.DataFrame({
    "Model": ["KNN", "SVM"],
    "Without PCA": [acc_knn_no_pca, acc_svm_no_pca],
    "With PCA": [acc_knn, acc_svm]
})

display(comparison_df)

PCA IMPACT ON MODEL PERFORMANCE:

In KNN the accuracy slightly incresesd not having much of a difference. PCA likely removed a little noise.
SVM accuracy droped signaficantly after PCA. 
