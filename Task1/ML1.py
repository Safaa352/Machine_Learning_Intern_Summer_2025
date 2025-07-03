import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# -----------------------------------
# Step 1: Load and clean data
# -----------------------------------
df = pd.read_csv("first inten project.csv")

# Create binary target column
df['is_canceled'] = df['booking status'].apply(lambda x: 1 if x.strip() == 'Canceled' else 0)

# Encode categorical columns
le = LabelEncoder()
df['market segment type'] = le.fit_transform(df['market segment type'])

# -----------------------------------
# Step 2: EDA - Basic info
# -----------------------------------
print("✅ Preview data:")
print(df.head())

print("\n📊 Shape:", df.shape)
print("\n🔎 Missing values:\n", df.isnull().sum())

# -----------------------------------
# Step 3: EDA - Visualizations
# -----------------------------------
# Cancellation Distribution
sns.countplot(x='is_canceled', data=df)
plt.title("Booking Cancellation Distribution")
plt.xlabel("Is Canceled (0=No, 1=Yes)")
plt.ylabel("Count")
plt.show()

# Lead Time
sns.boxplot(x='is_canceled', y='lead time', data=df)
plt.title("Lead Time vs Cancellation")
plt.show()

# P-C
sns.boxplot(x='is_canceled', y='P-C', data=df)
plt.title("Previous Cancellations (P-C)")
plt.show()

# P-not-C
sns.boxplot(x='is_canceled', y='P-not-C', data=df)
plt.title("Previous Non-Canceled Bookings (P-not-C)")
plt.show()

# Repeated
sns.countplot(x='repeated', hue='is_canceled', data=df)
plt.title("Repeated Guest vs Cancellation")
plt.show()

# Market Segment
sns.countplot(x='market segment type', hue='is_canceled', data=df)
plt.title("Market Segment Type vs Cancellation")
plt.xticks(rotation=45)
plt.show()

# -----------------------------------
# Step 4: Select features and target
# -----------------------------------
features = ['lead time', 'P-C', 'P-not-C', 'repeated', 'market segment type']
X = df[features].values
y = df['is_canceled'].values

# -----------------------------------
# Step 5: Split data
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------------
# Step 6: Train KNN model
# -----------------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# -----------------------------------
# Step 7: Predict and Evaluate
# -----------------------------------
y_pred = knn.predict(X_test)

# Accuracy
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))

# Classification report
print("\n📄 Classification Report:\n", classification_report(y_test, y_pred))

# -----------------------------------
# Step 8: Confusion Matrix
# -----------------------------------
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Canceled", "Canceled"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - KNN (sklearn)")
plt.show()
