import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os
print("Current Working Directory:", os.getcwd())

# Load the data
df = pd.read_csv("src/diabetes_prediction_dataset.csv")


# Preprocessing steps
df.drop_duplicates(inplace=True)
df = df.dropna()

# Feature engineering
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 65, 100], labels=['Child', 'Young Adult', 'Adult', 'Middle-aged', 'Senior'])
df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 24.9, 29.9, 100], labels=['Underweight', 'Normal weight', 'Overweight', 'Obese'])
df['hypertension_heart_disease'] = df['hypertension'] * df['heart_disease']
df['health_risk_score'] = df['hypertension'] + df['heart_disease'] + (df['age'] > 50).astype(int)
df['HbA1c_to_glucose_ratio'] = df['HbA1c_level'] / df['blood_glucose_level']
df['high_glucose'] = (df['blood_glucose_level'] > 150).astype(int)

# Encoding
le = LabelEncoder()
categorical_columns = ['gender', 'smoking_history', 'age_group', 'bmi_category']
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Select features and target
features = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'health_risk_score', 'high_glucose']
X = df[features]
y = df['diabetes']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
clf = GradientBoostingClassifier(random_state=42)
clf.fit(X_train, y_train)

# Save the model to a pickle file
filename = "Diabetes_Prediction_Model.pkl"
pickle.dump(clf, open(filename, "wb"))

print(f"Model saved as {filename}")