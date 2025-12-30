import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle

# 1. LOAD DATA
data = pd.read_csv('loan_data.csv')

# 2. VALIDATE & PREPARE DATA
required = ['age','gender','marital_status','education_level','annual_income','employment_status','credit_score','loan_amount','loan_paid_back']
missing = [c for c in required if c not in data.columns]
if missing:
    raise ValueError(f"Missing required columns in CSV: {missing}")

# Target (rename / use correct column)
y = data['loan_paid_back'].astype(int)

# Numeric and categorical columns
num_cols = ['age','annual_income','credit_score','loan_amount']
cat_cols = ['gender','marital_status','education_level','employment_status']

# Fill missing numeric values with median to be robust
data[num_cols] = data[num_cols].fillna(data[num_cols].median())

# One-hot encode categorical vars (safer than brittle map) and ensure dtype=str to avoid issues
X_num = data[num_cols]
X_cat = pd.get_dummies(data[cat_cols].astype(str), drop_first=True)

# Final feature set
X = pd.concat([X_num, X_cat], axis=1)

# 4. TRAIN / EVALUATE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate on test set
preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

# 5. SAVE: save both model and the feature columns so production code can build the same input
pickle.dump({'model': model, 'features': list(X.columns)}, open('model.pkl', 'wb'))
print("✅ Model trained and saved to model.pkl")