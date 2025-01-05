import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib
from datetime import datetime
import os

# df = pd.read_excel('../data/pitcher_prediction_dataset/pitcher_prediction_dataset_V4.xlsx')
df = pd.read_csv('../data/pitcher_prediction_dataset/pitcher_prediction_dataset_V4.csv', low_memory=False)
print("Finished Loading Data")
df = df[~df['pitch_type'].isin([15, 16])]
cols_to_convert = [
    'B_release_speed_level', 'B_release_pos_x_level', 'B_release_pos_y_level', 'B_release_pos_z_level',
    'B_pfx_x_level', 'B_pfx_z_level', 'B_vx0_level', 'B_vy0_level', 'B_vz0_level', 'B_ax_level',
    'B_ay_level', 'B_az_level', 'B_effective_speed_level', 'B_release_spin_rate_level',
    'B_release_extension_level', 'B_plate_x_level', 'B_plate_z_level'
]

for col in cols_to_convert:
    df[col] = df[col].str.extract('([0-9.-]+)', expand=False)
    df[col] = df[col].astype(float)
df['prev_delta_run_exp_1'] = df['prev_delta_run_exp_1'].fillna(0)
df['prev_delta_run_exp_2'] = df['prev_delta_run_exp_2'].fillna(0)
df['prev_delta_run_exp_3'] = df['prev_delta_run_exp_3'].fillna(0)
df['prev_delta_run_exp_4'] = df['prev_delta_run_exp_4'].fillna(0)
df['prev_delta_run_exp_5'] = df['prev_delta_run_exp_5'].fillna(0)

df = df.dropna()
df = df.drop(columns=['pitcher', 'game_pk', 'delta_run_exp'])


X = df.drop(columns=['pitch_type'])
y = df['pitch_type']
# print(y.sort_values())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Finished standardizing data")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
print("Finished Splitting data")
# print(sorted(y_train))
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 0.9],
    # 'colsample_bytree': [0.6, 0.8, 1.0],
    # 'min_child_weight': [1, 3, 5, 7],
    # 'gamma': [0, 0.1, 0.2, 0.3],
    # 'reg_alpha': [0, 0.01, 0.1, 1],
    # 'reg_lambda': [1, 1.5, 2, 3],
    # 'scale_pos_weight': [1, 2, 5, 10]
}

pca_components_list = [30, 40, 50, 60, 70, 80, 90, None]
pca_accuracies = []
best_accuracy = 0
best_model = None
best_pca_components = 0

print("Starting PCA & GridSearchCV")
for n_components in pca_components_list:
    print('n_components:', n_components)
    if n_components is not None:
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
    else:
        X_train_pca, X_test_pca = X_train, X_test

    xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
    grid_search = GridSearchCV(xgb_model, param_grid, cv=2, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_pca, y_train)

    y_pred = grid_search.best_estimator_.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    pca_accuracies.append((n_components, accuracy))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = grid_search.best_estimator_
        best_pca_components = n_components


os.makedirs('saved_models', exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_filename = f"../model/best_xgboost_pitcher_agent_prediction_model_{timestamp}.joblib"

joblib.dump({
    'model': best_model,
    'pca': pca if best_pca_components is not None else None,
    'n_components': best_pca_components,
    'accuracy': best_accuracy
}, os.path.join('saved_models', model_filename))


components, accuracies = zip(*pca_accuracies)
components = ['None' if c is None else str(c) for c in components]

plt.figure(figsize=(10, 6))
plt.bar(components, accuracies, color='skyblue')
plt.xlabel('PCA Components')
plt.ylabel('Accuracy')
plt.title('PCA Components vs Model Accuracy')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


feature_importances = best_model.feature_importances_

if best_pca_components is None:
    features = X.columns
else:
    features = [f'PC{i+1}' for i in range(best_pca_components)]

plt.figure(figsize=(12, 6))
plt.barh(features, feature_importances, color='coral')
plt.xlabel('Importance')
plt.title('Feature Importance (Best Model)')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()


print(f"Best accuracy: {best_accuracy:.4f} with PCA components: {best_pca_components}")



saved_model = joblib.load(os.path.join('saved_models', model_filename))
loaded_model = saved_model['model']
loaded_pca = saved_model['pca']

if loaded_pca is not None:
    X_test_pca = loaded_pca.transform(X_test)
else:
    X_test_pca = X_test

y_pred_loaded = loaded_model.predict(X_test_pca)
loaded_accuracy = accuracy_score(y_test, y_pred_loaded)
print(f"Accuracy of loaded model: {loaded_accuracy:.4f}")