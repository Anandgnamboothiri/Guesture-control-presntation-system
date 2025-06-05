import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the collected gesture data
with open('gesture_data/gesture_data.pkl', 'rb') as f:
    gesture_data = pickle.load(f)

# Prepare the dataset for training
X = []  # Features (flattened landmarks)
y = []  # Labels

for label, landmarks_list in gesture_data.items():
    for landmarks in landmarks_list:
        # Flatten the landmarks (21 points, each with x, y, z coordinates)
        flat_landmarks = np.array(landmarks).flatten()
        X.append(flat_landmarks)
        y.append(label)

X = np.array(X)
y = np.array(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')

# Save the trained model
with open('gesture_classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)
