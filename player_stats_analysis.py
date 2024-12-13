# import numpy as np
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# import shap
# from mapie.regression import MapieRegressor

# # Load Data
# data = pd.read_csv("player_stats_data.csv")  
# features = [
#     "player_1_number_of_shots", 
#     "player_1_total_player_speed", 
#     "player_1_last_player_speed",
#     "player_2_number_of_shots",
#     "player_2_total_player_speed",
#     "player_2_last_player_speed"
# ]
# target_1 = "player_1_average_shot_speed"
# target_2 = "player_2_average_shot_speed"

# # Handle Missing Values (if needed)
# data = data.fillna(0)

# # Train-Test Split for player 1
# X1 = data[features[:3]]  # Player 1 features
# y1 = data[target_1]
# X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X1, y1, test_size=0.2, random_state=42)

# # Train-Test Split for player 2
# X2 = data[features[3:]]  # Player 2 features
# y2 = data[target_2]
# X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

# # Train Ensemble Model for player 1
# rf_1 = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_1.fit(X_train_1, y_train_1)

# # Train Ensemble Model for player 2
# rf_2 = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_2.fit(X_train_2, y_train_2)

# # Generate Predictions and Variance for player 1
# ensemble_preds_1 = [tree.predict(X_test_1) for tree in rf_1.estimators_]
# mean_pred_1 = np.mean(ensemble_preds_1, axis=0)
# variance_pred_1 = np.var(ensemble_preds_1, axis=0)

# # Generate Predictions and Variance for player 2
# ensemble_preds_2 = [tree.predict(X_test_2) for tree in rf_2.estimators_]
# mean_pred_2 = np.mean(ensemble_preds_2, axis=0)
# variance_pred_2 = np.var(ensemble_preds_2, axis=0)

# # Output Mean and Variance for Ensemble (player 1)
# results_ensemble_1 = pd.DataFrame({
#     "Mean Prediction": mean_pred_1,
#     "Variance": variance_pred_1,
#     "Actual": y_test_1.reset_index(drop=True)
# })
# print("Ensemble Predictions with Variance (Player 1):")
# print(results_ensemble_1)

# # Output Mean and Variance for Ensemble (player 2)
# results_ensemble_2 = pd.DataFrame({
#     "Mean Prediction": mean_pred_2,
#     "Variance": variance_pred_2,
#     "Actual": y_test_2.reset_index(drop=True)
# })
# print("Ensemble Predictions with Variance (Player 2):")
# print(results_ensemble_2)

# # Conformal Prediction for Confidence Intervals for player 1
# mapie_1 = MapieRegressor(estimator=rf_1, cv=10)  # Set cv to an integer for 10-fold cross-validation
# mapie_1.fit(X_train_1, y_train_1)
# y_pred_1, y_pis_1 = mapie_1.predict(X_test_1, alpha=0.1)  # 90% Confidence Interval

# # Conformal Prediction for Confidence Intervals for player 2
# mapie_2 = MapieRegressor(estimator=rf_2, cv=10)  # Set cv to an integer for 10-fold cross-validation
# mapie_2.fit(X_train_2, y_train_2)
# y_pred_2, y_pis_2 = mapie_2.predict(X_test_2, alpha=0.1)  # 90% Confidence Interval

# # Reshaping the predictions and intervals for player 1
# y_pred_1 = y_pred_1.reshape(-1)  # Ensure 1D array
# lower_bound_1 = y_pis_1[:, 0].reshape(-1)  # Ensure 1D array
# upper_bound_1 = y_pis_1[:, 1].reshape(-1)  # Ensure 1D array

# results_with_intervals_1 = pd.DataFrame({
#     "Prediction": y_pred_1,
#     "Lower Bound": lower_bound_1,
#     "Upper Bound": upper_bound_1,
#     "Actual": y_test_1.reset_index(drop=True)
# })

# print("\nConformal Prediction Results (Player 1):")
# print(results_with_intervals_1)

# # Reshaping the predictions and intervals for player 2
# y_pred_2 = y_pred_2.reshape(-1)  # Ensure 1D array
# lower_bound_2 = y_pis_2[:, 0].reshape(-1)  # Ensure 1D array
# upper_bound_2 = y_pis_2[:, 1].reshape(-1)  # Ensure 1D array

# results_with_intervals_2 = pd.DataFrame({
#     "Prediction": y_pred_2,
#     "Lower Bound": lower_bound_2,
#     "Upper Bound": upper_bound_2,
#     "Actual": y_test_2.reset_index(drop=True)
# })

# print("\nConformal Prediction Results (Player 2):")
# print(results_with_intervals_2)

# # Explain Predictions with SHAP for player 1
# explainer_1 = shap.TreeExplainer(rf_1)
# shap_values_1 = explainer_1.shap_values(X_test_1)

# print("\nGenerating SHAP Summary Plot for Player 1...")
# shap.summary_plot(shap_values_1, X_test_1, feature_names=features[:3])

# # Explain Predictions with SHAP for player 2
# explainer_2 = shap.TreeExplainer(rf_2)
# shap_values_2 = explainer_2.shap_values(X_test_2)

# print("\nGenerating SHAP Summary Plot for Player 2...")
# shap.summary_plot(shap_values_2, X_test_2, feature_names=features[3:])


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import shap
import lime
import lime.lime_tabular
from mapie.regression import MapieRegressor
import matplotlib.pyplot as plt
from pdpbox import pdp, info_plots

# Load Data
data = pd.read_csv("player_stats_data.csv")  
features = [
    "player_1_number_of_shots", 
    "player_1_total_player_speed", 
    "player_1_last_player_speed",
    "player_2_number_of_shots",
    "player_2_total_player_speed",
    "player_2_last_player_speed"
]
target_1 = "player_1_average_shot_speed"
target_2 = "player_2_average_shot_speed"

# Handle Missing Values (if needed)
data = data.fillna(0)

# Train-Test Split for player 1
X1 = data[features[:3]]  # Player 1 features
y1 = data[target_1]
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X1, y1, test_size=0.2, random_state=42)

# Train-Test Split for player 2
X2 = data[features[3:]]  # Player 2 features
y2 = data[target_2]
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Train Ensemble Model for player 1
rf_1 = RandomForestRegressor(n_estimators=100, random_state=42)
rf_1.fit(X_train_1, y_train_1)

# Train Ensemble Model for player 2
rf_2 = RandomForestRegressor(n_estimators=100, random_state=42)
rf_2.fit(X_train_2, y_train_2)

# Generate Predictions and Variance for player 1
ensemble_preds_1 = [tree.predict(X_test_1) for tree in rf_1.estimators_]
mean_pred_1 = np.mean(ensemble_preds_1, axis=0)
variance_pred_1 = np.var(ensemble_preds_1, axis=0)

# Generate Predictions and Variance for player 2
ensemble_preds_2 = [tree.predict(X_test_2) for tree in rf_2.estimators_]
mean_pred_2 = np.mean(ensemble_preds_2, axis=0)
variance_pred_2 = np.var(ensemble_preds_2, axis=0)

# Output Mean and Variance for Ensemble (player 1)
results_ensemble_1 = pd.DataFrame({
    "Mean Prediction": mean_pred_1,
    "Variance": variance_pred_1,
    "Actual": y_test_1.reset_index(drop=True)
})
print("Ensemble Predictions with Variance (Player 1):")
print(results_ensemble_1)

# Output Mean and Variance for Ensemble (player 2)
results_ensemble_2 = pd.DataFrame({
    "Mean Prediction": mean_pred_2,
    "Variance": variance_pred_2,
    "Actual": y_test_2.reset_index(drop=True)
})
print("Ensemble Predictions with Variance (Player 2):")
print(results_ensemble_2)



# Conformal Prediction for Confidence Intervals for player 1
mapie_1 = MapieRegressor(estimator=rf_1, cv=10)  # Set cv to an integer for 10-fold cross-validation
mapie_1.fit(X_train_1, y_train_1)
y_pred_1, y_pis_1 = mapie_1.predict(X_test_1, alpha=0.1)  # 90% Confidence Interval

# Conformal Prediction for Confidence Intervals for player 2
mapie_2 = MapieRegressor(estimator=rf_2, cv=10)  # Set cv to an integer for 10-fold cross-validation
mapie_2.fit(X_train_2, y_train_2)
y_pred_2, y_pis_2 = mapie_2.predict(X_test_2, alpha=0.1)  # 90% Confidence Interval

# Reshaping the predictions and intervals for player 1
y_pred_1 = y_pred_1.reshape(-1)  # Ensure 1D array
lower_bound_1 = y_pis_1[:, 0].reshape(-1)  # Ensure 1D array
upper_bound_1 = y_pis_1[:, 1].reshape(-1)  # Ensure 1D array

results_with_intervals_1 = pd.DataFrame({
    "Prediction": y_pred_1,
    "Lower Bound": lower_bound_1,
    "Upper Bound": upper_bound_1,
    "Actual": y_test_1.reset_index(drop=True)
})

print("\nConformal Prediction Results (Player 1):")
print(results_with_intervals_1)

# Reshaping the predictions and intervals for player 2
y_pred_2 = y_pred_2.reshape(-1)  # Ensure 1D array
lower_bound_2 = y_pis_2[:, 0].reshape(-1)  # Ensure 1D array
upper_bound_2 = y_pis_2[:, 1].reshape(-1)  # Ensure 1D array

results_with_intervals_2 = pd.DataFrame({
    "Prediction": y_pred_2,
    "Lower Bound": lower_bound_2,
    "Upper Bound": upper_bound_2,
    "Actual": y_test_2.reset_index(drop=True)
})

print("\nConformal Prediction Results (Player 2):")
print(results_with_intervals_2)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Calculate metrics for player 1
mse_1 = mean_squared_error(y_test_1, mean_pred_1)
mae_1 = mean_absolute_error(y_test_1, mean_pred_1)
r2_1 = r2_score(y_test_1, mean_pred_1)

print("Metrics for Player 1:")
print(f"Mean Squared Error (MSE): {mse_1}")
print(f"Mean Absolute Error (MAE): {mae_1}")
print(f"R-squared (R²): {r2_1}")

# Calculate metrics for player 2
mse_2 = mean_squared_error(y_test_2, mean_pred_2)
mae_2 = mean_absolute_error(y_test_2, mean_pred_2)
r2_2 = r2_score(y_test_2, mean_pred_2)

print("\nMetrics for Player 2:")
print(f"Mean Squared Error (MSE): {mse_2}")
print(f"Mean Absolute Error (MAE): {mae_2}")
print(f"R-squared (R²): {r2_2}")


# Explain Predictions with SHAP for player 1
explainer_1 = shap.TreeExplainer(rf_1)
shap_values_1 = explainer_1.shap_values(X_test_1)

print("\nGenerating SHAP Summary Plot for Player 1...")
shap.summary_plot(shap_values_1, X_test_1, feature_names=features[:3])

# Explain Predictions with SHAP for player 2
explainer_2 = shap.TreeExplainer(rf_2)
shap_values_2 = explainer_2.shap_values(X_test_2)

print("\nGenerating SHAP Summary Plot for Player 2...")
shap.summary_plot(shap_values_2, X_test_2, feature_names=features[3:])



# LIME Explanation for Player 1
explainer_lime_1 = lime.lime_tabular.LimeTabularExplainer(X_train_1.values, feature_names=features[:3], 
                                                           mode="regression")
lime_exp_1 = explainer_lime_1.explain_instance(X_test_1.iloc[0].values, rf_1.predict, num_features=5)
lime_exp_1.as_pyplot_figure()
plt.title("LIME Explanation for Player 1")
plt.show()

# LIME Explanation for Player 2
explainer_lime_2 = lime.lime_tabular.LimeTabularExplainer(X_train_2.values, feature_names=features[3:], 
                                                           mode="regression")
lime_exp_2 = explainer_lime_2.explain_instance(X_test_2.iloc[0].values, rf_2.predict, num_features=5)
lime_exp_2.as_pyplot_figure()
plt.title("LIME Explanation for Player 2")
plt.show()







