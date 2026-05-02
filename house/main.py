# ================================
# 1. Import Libraries
# ================================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import pickle

# ================================
# 2. Load Dataset
# ================================
df = pd.read_csv("house_price_dataset.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# ================================
# 3. Data Preprocessing
# ================================

# Convert Yes/No to 1/0
binary_cols = ['Main Road', 'Guest Room', 'Basement', 'Air Conditioning']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# Label Encoding for categorical columns
cat_cols = ['City', 'Furnishing', 'Water Supply', 'Preferred Tenant']

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# ================================
# 4. Split Features & Target
# ================================
X = df.drop("Price", axis=1)
y = df["Price"]

# ================================
# 5. Train-Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# 6. Model Training
# ================================
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ================================
# 7. Prediction
# ================================
y_pred = model.predict(X_test)

# ================================
# 8. Evaluation
# ================================
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R2 Score:", r2)

# ================================
# 9. Save Model
# ================================
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")