import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, r2_score

# === LOAD DATA ===
df = pd.read_csv("trip_summary.csv")

# === GRAVITY MODEL FUNCTIONS ===
def gravity_model_vectorized(P_i, P_j, D_ij, alpha, beta, gamma, G):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = G * (P_i ** alpha) * (P_j ** beta) / (D_ij ** gamma)
        result[np.isinf(result)] = 0
        return np.nan_to_num(result)

def loss_function(params, df):
    alpha, beta, gamma, G = params
    predicted = gravity_model_vectorized(
        df["start_population"].values,
        df["end_population"].values,
        df["distance"].values,
        alpha, beta, gamma, G
    )
    actual = df["trips"].values
    return mean_squared_error(actual, predicted)

# === OPTIMIZATION ===
initial_params = [1, 1, 1, 1]
bounds = [(-10, 20), (-10, 20), (-10, 20), (1e-10, 100)]

result = minimize(loss_function, initial_params, args=(df,), method="L-BFGS-B", bounds=bounds)
alpha, beta, gamma, G = result.x

print("=== Optimized Parameters ===")
print(f"Alpha (α):  {alpha:.4f}")
print(f"Beta  (β):  {beta:.4f}")
print(f"Gamma (γ):  {gamma:.4f}")
print(f"G:         {G:.8f}")

# === PREDICTION ===
df["predicted_trips"] = gravity_model_vectorized(
    df["start_population"].values,
    df["end_population"].values,
    df["distance"].values,
    alpha, beta, gamma, G
)

# === EVALUATION ===
r2 = r2_score(df["trips"], df["predicted_trips"])
mse = mean_squared_error(df["trips"], df["predicted_trips"])

print(f"\n=== Model Evaluation ===")
print(f"R² Score: {r2:.4f}")
print(f"MSE:      {mse:.2f}")

# === SAVE RESULTS ===
df.to_csv("results.csv", index=False)
print(f"\nResults saved to results.csv\n")

