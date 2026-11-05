# %% [markdown]
# # Machine Learning Lab - All Experiments (Code Only)
# Each `# %%` marks a new cell (works in Jupyter, VS Code, Spyder).

# %% [markdown]
# ## Ex. 1 - Univariate, Bivariate and Multivariate Regression

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_model(model, X, y, name="Model"):
    predictions = model.predict(X)
    print(f"\n{name} Performance:")
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)
    print("MSE:", mean_squared_error(y, predictions))
    print("R2 Score:", r2_score(y, predictions))
    return predictions


print("=== Univariate Regression ===")
X_uni = np.random.rand(100, 1) * 10
y_uni = 3 * X_uni.squeeze() + 4 + np.random.randn(100)

model_uni = LinearRegression()
model_uni.fit(X_uni, y_uni)
pred_uni = evaluate_model(model_uni, X_uni, y_uni, "Univariate Regression")

plt.figure(figsize=(6, 4))
plt.scatter(X_uni, y_uni, color="blue", label="Actual")
plt.plot(X_uni, pred_uni, color="red", label="Predicted")
plt.title("Univariate Linear Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

print("\n=== Bivariate Regression ===")
X_bi = np.random.rand(100, 2) * 10
y_bi = 2 * X_bi[:, 0] + 5 * X_bi[:, 1] + 3 + np.random.randn(100)

model_bi = LinearRegression()
model_bi.fit(X_bi, y_bi)
pred_bi = evaluate_model(model_bi, X_bi, y_bi, "Bivariate Regression")

print("\n=== Multivariate Regression ===")
# Generate synthetic data
X_multi = np.random.rand(100, 5) * 10

# Create target with 5 predictors
y_multi = (1.5 * X_multi[:, 0] + 2.2 * X_multi[:, 1] +
           0.5 * X_multi[:, 2] - 1.2 * X_multi[:, 3] +
           3.3 * X_multi[:, 4] + 5 + np.random.randn(100))

model_multi = LinearRegression()
model_multi.fit(X_multi, y_multi)
pred_multi = evaluate_model(model_multi, X_multi, y_multi, "Multivariate Regression")

# %% [markdown]
# ## Ex. 2 - Simple Linear Regression using Least Squares Method

# %%
import numpy as np
import matplotlib.pyplot as plt

X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

mean_x = np.mean(X)
mean_y = np.mean(y)

numerator = np.sum((X - mean_x) * (y - mean_y))
denominator = np.sum((X - mean_x) ** 2)

slope = numerator / denominator
intercept = mean_y - slope * mean_x

print("Slope (m):", slope)
print("Intercept (c):", intercept)

y_pred = slope * X + intercept

plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, y_pred, color='red', label='Regression line')
plt.title('Simple Linear Regression (Least Squares)')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

ss_total = np.sum((y - mean_y) ** 2)
ss_residual = np.sum((y - y_pred) ** 2)
r2_score = 1 - (ss_residual / ss_total)
print("R² Score:", r2_score)

# %% [markdown]
# ## Ex. 3 - Logistic Model

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        self.m, self.n = X.shape
        self.weights = np.zeros(self.n)
        self.bias = 0

        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(linear_model)

            dw = (1 / self.m) * np.dot(X.T, (y_pred - y))
            db = (1 / self.m) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in y_pred]


X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                           n_clusters_per_class=1, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegressionScratch(learning_rate=0.1, epochs=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy (scratch):", accuracy_score(y_test, y_pred))

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_sklearn = clf.predict(X_test)
print("Accuracy (scikit-learn):", accuracy_score(y_test, y_sklearn))


def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = np.array(model.predict(grid)).reshape(xx.shape)

    plt.contourf(xx, yy, preds, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    plt.title("Logistic Regression Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.show()


plot_decision_boundary(model, X, y)

# %% [markdown]
# ## Ex. 4 - Single Layer Perceptron

# %%
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def step_function(x):
    return np.where(x >= 0, 1, 0)


class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        self.weights = np.zeros(self.n_features)
        self.bias = 0

        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = step_function(linear_output)
                update = self.learning_rate * (y[idx] - y_pred)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return step_function(linear_output)


X, y = make_classification(n_samples=100, n_features=2,
                           n_informative=2, n_redundant=0,
                           n_clusters_per_class=1, random_state=42)
y = np.where(y <= 0, 0, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Perceptron(learning_rate=0.01, epochs=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = model.predict(grid).reshape(xx.shape)

    plt.contourf(xx, yy, preds, alpha=0.4, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    plt.title("Single-Layer Perceptron Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.show()


plot_decision_boundary(X, y, model)

# %% [markdown]
# ## Ex. 5 - Multi Layer Perceptron with Backpropagation

# %%
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = sigmoid(self.Z2)
        return self.A2

    def backward(self, X, y, output, learning_rate):
        m = y.shape[0]
        error = output - y
        dZ2 = error * sigmoid_derivative(output)
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dZ1 = np.dot(dZ2, self.W2.T) * sigmoid_derivative(self.A1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train(self, X, y, epochs=1000, learning_rate=0.1):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)
            if epoch % 100 == 0:
                loss = np.mean((y - output) ** 2)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        output = self.forward(X)
        return (output > 0.5).astype(int)


X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
y = y.reshape(-1, 1)  # Make y 2D for output layer compatibility

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlp = MLP(input_size=2, hidden_size=5, output_size=1)
mlp.train(X_train, y_train, epochs=1000, learning_rate=0.1)

y_pred = mlp.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"\nTest Accuracy: {accuracy:.2f}")


def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = model.predict(grid).reshape(xx.shape)

    plt.contourf(xx, yy, preds, alpha=0.3, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="coolwarm", edgecolors='k')
    plt.title("MLP Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.show()


plot_decision_boundary(mlp, X, y)

# %% [markdown]
# ## Ex. 6 - Face Recognition using SVM Classifier
# (Downloads the LFW dataset on first run - needs internet)

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

X = lfw_people.data        # Flattened image data
y = lfw_people.target      # Target labels
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total samples:", X.shape[0])
print("Image shape:", lfw_people.images[0].shape)
print("Number of classes:", n_classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

n_components = 100
pca = PCA(n_components=n_components, whiten=True).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

clf = SVC(kernel='rbf', class_weight='balanced', C=1000, gamma=0.001)
clf.fit(X_train_pca, y_train)

y_pred = clf.predict(X_test_pca)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

plt.figure(figsize=(10, 6))
conf_mat = confusion_matrix(y_test, y_pred, labels=range(n_classes))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


def plot_gallery(images, titles, h, w, n_row=3, n_col=5):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0.01, left=0.01, right=0.99, top=0.90, hspace=0.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


def title(y_pred, y_true, target_names, i):
    pred_name = target_names[y_pred[i]].split()[-1]
    true_name = target_names[y_true[i]].split()[-1]
    return f'Pred: {pred_name}\nTrue: {true_name}'


prediction_titles = [title(y_pred, y_test, target_names, i) for i in range(y_pred.shape[0])]
plot_gallery(X_test, prediction_titles,
             h=lfw_people.images.shape[1], w=lfw_people.images.shape[2])
plt.show()

# %% [markdown]
# ## Ex. 7 - Decision Tree

# %%
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree Visualization")
plt.show()

# %% [markdown]
# ## Ex. 8 - Boosting (AdaBoost)

# %%
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

X, y = make_classification(n_samples=1000, n_features=10,
                           n_informative=5, n_redundant=2,
                           random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42)

base_estimator = DecisionTreeClassifier(max_depth=1)

model = AdaBoostClassifier(estimator=base_estimator,
                           n_estimators=50,
                           learning_rate=1.0,
                           random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), model.feature_importances_)
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.title("Feature Importances from AdaBoost")
plt.show()

# %% [markdown]
# ## Ex. 9 - KNN and K-Means Clustering
# ### KNN

# %%
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("KNN Accuracy:", accuracy_score(y_test, y_pred))

# %% [markdown]
# ### K-Means Clustering

# %%
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=200, c='red', marker='X')
plt.title("K-Means Clustering")
plt.show()

# %% [markdown]
# ## Ex. 10 - Dimensionality Reduction - PCA

# %%
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

pca = PCA(n_components=2)
X_r = pca.fit_transform(X)

print(f"Explained variance ratio of the 2 components: {pca.explained_variance_ratio_}")

plt.figure(figsize=(8, 6))
colors = ['navy', 'turquoise', 'darkorange']
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, lw=2, label=target_name)

plt.legend()
plt.title('PCA of Iris dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()

# %% [markdown]
# ## Ex. 12 - Traffic Sign Detection using CNN (Mini Project)
# Requires the GTSRB dataset. Update `data_dir` to your dataset path.

# %%
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

data_dir = './gtsrb/Train'  # Update if different
num_classes = 43

image_data = []
labels = []

print("Loading images...")
for class_id in range(num_classes):
    class_path = os.path.join(data_dir, str(class_id))
    if not os.path.exists(class_path):
        continue
    for img_file in os.listdir(class_path):
        try:
            img_path = os.path.join(class_path, img_file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (32, 32))
            image_data.append(img)
            labels.append(class_id)
        except:
            continue

X = np.array(image_data)
y = np.array(labels)
print("Total images loaded:", len(X))

X = X / 255.0
y = to_categorical(y, num_classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# %%
print("Training model...")
history = model.fit(X_train, y_train, epochs=10, batch_size=64,
                    validation_data=(X_test, y_test))

print("Evaluating on test set...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

model.save("traffic_sign_cnn.h5")
print("Model saved as traffic_sign_cnn.h5")

# %%
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()


# %%
def predict_sign(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (32, 32))
    img = img / 255.0
    img = img.reshape(1, 32, 32, 3)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    print(f"Predicted Class: {class_index}, Confidence: {confidence:.2f}")


# predict_sign('./gtsrb/Test/00014.png')
