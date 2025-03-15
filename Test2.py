from scipy.io import arff
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from preprocessing.preprocess_urls2 import preprocess_url_dataset
# Load the processed dataset


csv_path = "data/phishing_site_urls.csv"
df = preprocess_url_dataset(csv_path)  # Using the function we built earlier
# Save the DataFrame to a CSV file


# Load the preprocessed DataFrame
#df = pd.read_csv('processed_urls.csv')








# Features and target
X = df.drop(columns=['URL', 'Label', 'label'])  # Drop non-numeric and target columns
y = df['label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
#rf_model = HistGradientBoostingClassifier(max_iter=3000, learning_rate=0.01, max_depth=10, class_weight='balanced')
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)
scores = cross_val_score(rf_model, X, y, cv=5)
print(f"Baseline accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
