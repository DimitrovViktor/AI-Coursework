import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from skimage import feature
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load a dataset (e.g., digits dataset from scikit-learn)
digits = datasets.load_digits()

# Visualize some example images
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for ax, image, label in zip(axes.ravel(), digits.images, digits.target):
    ax.axis('off')
    ax.imshow(image, cmap=plt.cm.gray_r)
    ax.set_title('Label: %i' % label)

plt.show()

# Extract HOG features from the images
hog_features = []
for image in digits.images:
    # Calculate HOG features
    hog_feature = feature.hog(image, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1))
    hog_features.append(hog_feature)

# Convert the dataset to a NumPy array
hog_features = np.array(hog_features)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(hog_features, digits.target, test_size=0.2, random_state=42)

# Train a Support Vector Machine (SVM) classifier
clf = svm.SVC(gamma=0.001, C=100)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
