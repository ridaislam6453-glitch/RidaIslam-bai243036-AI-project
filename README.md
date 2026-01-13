# RidaIslam-bai243036-AI-project
Linear Regression
# Step-by-step outputs

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("Step 1: Libraries imported.\n")

# Step 2: Dataset
data = {
    'humidity': [78, 80, 75, 70, 72, 68, 85, 90, 65, 60],
    'wind_speed': [3.2, 2.9, 3.5, 4.0, 3.8, 4.2, 2.5, 2.0, 4.5, 5.0],
    'temperature': [12.3, 11.8, 13.1, 14.0, 13.5, 15.0, 10.5, 9.8, 16.2, 17.0]
}
df = pd.DataFrame(data)
print("Step 2: Dataset loaded:")
print(df, "\n")

# Step 3: Check missing values
print("Step 3: Checking for missing values:")
print(df.isnull().sum(), "\n")

# Step 4: Features and Target
X = df[['humidity', 'wind_speed']]
y = df['temperature']
print("Step 4: Features (X) and Target (y):")
print(X)
print(y, "\n")

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1  # fix random_state for reproducibility
)
print("Step 5: Train-Test Split done.")
print("X_train:\n", X_train)
print("X_test:\n", X_test)
print("y_train:\n", y_train)
print("y_test:\n", y_test, "\n")

# Step 6: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
print("Step 6: Model trained.")
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_, "\n")

# Step 7: Predict Temperature
y_pred = model.predict(X_test)
print("Step 7: Predictions done.")
print("Predicted Temperatures:", y_pred)
print("Actual Temperatures:", y_test.values, "\n")

# Step 8: Evaluate Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Step 8: Model Evaluation")
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2, "\n")

# Step 9: Visualization
print("Step 9: Visualizing results...")
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Temperature")
plt.ylabel("Predicted Temperature")
plt.title("Actual vs Predicted Temperature")
plt.show()
print("Visualization completed.")
