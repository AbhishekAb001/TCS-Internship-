import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# Example data (replace these with your actual dataset)
# Generating random data for demonstration purposes
np.random.seed(42)
X = np.random.rand(100, 5)  # 100 samples, 5 features
y = np.random.rand(100)     # 100 target values

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a list of hyperparameters to explore
hyperparameters = [
    {'fit_intercept': True},
    {'fit_intercept': False}
]

# Train models with different hyperparameters
models = []
for params in hyperparameters:
    model = make_pipeline(StandardScaler(), LinearRegression(**params))
    model.fit(X_train, y_train)
    models.append(model)

# Evaluate model performance
mse_values = []
for model in models:
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_values.append(mse)

# Compare performance
best_model_index = mse_values.index(min(mse_values))
best_model = models[best_model_index]
best_params = hyperparameters[best_model_index]
best_mse = mse_values[best_model_index]

# Print results
print("Model Performances:")
for i, params in enumerate(hyperparameters):
    print(f"Model {i+1} Hyperparameters:", params)
    print(f"Model {i+1} MSE:", mse_values[i])
    print()

print("Best Model Hyperparameters:", best_params)
print("Best Model MSE:", best_mse)
