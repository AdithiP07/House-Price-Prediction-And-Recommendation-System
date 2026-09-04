"""
Generates the comprehensive Jupyter Notebook 'notebooks/model_development.ipynb'
with all 10 stages, detailed markdown descriptions, formulas, code cells, and annotations.
"""

import os
import nbformat as nbf

def build_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0"
        }
    }

    cells = []

    # Title & Metadata
    cells.append(nbf.v4.new_markdown_cell("""# AI-Powered House Price Prediction and Smart Property Recommendation System

**End-to-End Machine Learning & Deep Learning Project**  
**Author:** AI Engineering Team  
**Dataset:** Metropolitan Real Estate Housing Dataset (Bangalore Market)

---

## Project Overview
This project presents an intelligent real estate intelligence platform that:
1. Accurately predicts house prices based on multidimensional property features (Area, BHK, Bathrooms, Age, Location, Amenities, Furnishing).
2. Segments real estate assets into **Budget**, **Mid-Range**, and **Luxury** tiers using Unsupervised **K-Means Clustering**.
3. Projects high-dimensional feature spaces using **Principal Component Analysis (PCA)**.
4. Compares traditional machine learning algorithms with a **Deep Neural Network (TensorFlow/Keras)**.
5. Deploys production-grade inference pipelines for a FastAPI backend and property recommendation engine.

---

## Table of Contents
1. [Data Collection and Exploration](#1.-Data-Collection-and-Exploration)
2. [Data Cleaning and Preprocessing](#2.-Data-Cleaning-and-Preprocessing)
3. [Feature Engineering](#3.-Feature-Engineering)
4. [Exploratory Data Analysis (EDA)](#4.-Exploratory-Data-Analysis)
5. [Supervised Learning Models](#5.-Supervised-Learning-Models)
6. [Model Evaluation & Cross Validation](#6.-Model-Evaluation)
7. [Unsupervised Learning (K-Means Clustering)](#7.-Unsupervised-Learning)
8. [Dimensionality Reduction (PCA)](#8.-Dimensionality-Reduction)
9. [Deep Learning with TensorFlow/Keras](#9.-Deep-Learning)
10. [Model Selection & Joblib Serialization](#10.-Model-Selection)
"""))

    # Section 1
    cells.append(nbf.v4.new_markdown_cell("""---
## 1. Data Collection and Exploration
In this section, we load the raw housing dataset using Pandas, inspect its structural schema, summary statistics, and check for missing values and duplicates.
"""))
    cells.append(nbf.v4.new_code_cell("""import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Load the dataset
data_path = os.path.join('..', 'data', 'raw', 'housing_raw.csv')
df_raw = pd.read_csv(data_path)

print(f"Dataset Shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
df_raw.head()"""))

    cells.append(nbf.v4.new_code_cell("""# Display structural information and datatypes
df_raw.info()"""))

    cells.append(nbf.v4.new_code_cell("""# Statistical summary of numerical variables
df_raw.describe().T"""))

    cells.append(nbf.v4.new_code_cell("""# Check for missing values and duplicate records
missing_counts = df_raw.isna().sum()
missing_pct = (df_raw.isna().sum() / len(df_raw)) * 100
duplicates_count = df_raw.duplicated().sum()

missing_df = pd.DataFrame({'Missing Count': missing_counts, 'Percentage (%)': missing_pct.round(2)})
print(f"Total Duplicate Rows: {duplicates_count}")
missing_df[missing_df['Missing Count'] > 0]"""))

    # Section 2
    cells.append(nbf.v4.new_markdown_cell("""---
## 2. Data Cleaning and Preprocessing
Real-world datasets contain anomalies, missing entries, duplicate recordings, and extreme outliers. In this section we:
1. **Deduplicate:** Eliminate duplicate observations.
2. **Impute Missing Values:** Median imputation for numerical features and mode imputation for categorical features.
3. **Outlier Detection and Treatment:** Utilize the Interquartile Range (IQR) method to cap extreme distortions.
4. **Encoding & Scaling:** Prepare transformations for downstream modeling.
"""))

    cells.append(nbf.v4.new_code_cell("""df_clean = df_raw.copy()

# 1. Remove duplicate rows
init_len = len(df_clean)
df_clean = df_clean.drop_duplicates().reset_index(drop=True)
print(f"Deduplication: Removed {init_len - len(df_clean)} duplicates. Remaining: {len(df_clean)}")

# 2. Impute missing values
# Bathrooms: Impute by BHK median (e.g. 2 BHK usually has 2 baths)
df_clean['Bathrooms'] = df_clean.groupby('BHK')['Bathrooms'].transform(lambda s: s.fillna(s.median()))
df_clean['Bathrooms'] = df_clean['Bathrooms'].fillna(df_clean['Bathrooms'].median()).astype(int)

# Property Age: Impute with overall median
df_clean['Property Age'] = df_clean['Property Age'].fillna(df_clean['Property Age'].median()).astype(int)

# Furnishing Status: Impute with mode
mode_furnishing = df_clean['Furnishing Status'].mode()[0]
df_clean['Furnishing Status'] = df_clean['Furnishing Status'].fillna(mode_furnishing)

print("Remaining Missing Values:")
print(df_clean.isna().sum())"""))

    cells.append(nbf.v4.new_code_cell("""# 3. Detect and Treat Outliers using Interquartile Range (IQR)
# Capping extreme bounds on Area and Price
for col in ['Price', 'Area']:
    Q1 = df_clean[col].quantile(0.01)
    Q3 = df_clean[col].quantile(0.99)
    IQR = Q3 - Q1
    lower_bound = max(0, Q1 - 1.5 * IQR)
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
    print(f"Column '{col}': Capping {outliers} extreme values outside [{lower_bound:.1f}, {upper_bound:.1f}]")
    df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)"""))

    # Section 3
    cells.append(nbf.v4.new_markdown_cell("""---
## 3. Feature Engineering
We synthesize three powerful domain-driven features:
1. **`price_per_sqft`**: $\\text{Price} / \\text{Area}$ (critical real estate valuation metric).
2. **`property_age_group`**: Categorizes construction vintage into *New (0-3 yrs)*, *Modern (4-10 yrs)*, *Established (11-20 yrs)*, and *Old (20+ yrs)*.
3. **`luxury_score`**: Weighted assessment based on available lifestyle amenities (Swimming Pool, Tennis Court, Clubhouse, 24/7 Security, High-Speed Elevators, etc.).
"""))

    cells.append(nbf.v4.new_code_cell("""# 1. price_per_sqft
df_clean['price_per_sqft'] = (df_clean['Price'] / df_clean['Area']).round(2)

# 2. property_age_group
def get_age_group(age):
    if age <= 3:
        return 'New (0-3 yrs)'
    elif age <= 10:
        return 'Modern (4-10 yrs)'
    elif age <= 20:
        return 'Established (11-20 yrs)'
    else:
        return 'Old (20+ yrs)'

df_clean['property_age_group'] = df_clean['Property Age'].apply(get_age_group)

# 3. luxury_score calculation
premium_weights = {
    'Swimming Pool': 1.5,
    'Clubhouse': 1.2,
    'Tennis Court': 1.4,
    'Gym': 1.0,
    'High-Speed Elevators': 1.0,
    'Landscaped Garden': 0.9,
    '24/7 Security': 0.8,
    'Power Backup': 0.8,
    'Covered Parking': 0.8,
    'Children\\'s Play Area': 0.7,
    'Intercom': 0.5,
    'Jogging Track': 0.6
}

def calculate_luxury_score(amenities_str):
    if pd.isna(amenities_str) or not str(amenities_str).strip():
        return 1.0
    items = [a.strip() for a in str(amenities_str).split(',')]
    raw_score = sum(premium_weights.get(item, 0.5) for item in items)
    normalized = min(10.0, max(1.0, (raw_score / 11.2) * 10.0))
    return round(normalized, 2)

df_clean['luxury_score'] = df_clean['Amenities'].apply(calculate_luxury_score)

df_clean[['Location', 'Area', 'BHK', 'Price', 'price_per_sqft', 'luxury_score', 'property_age_group']].head()"""))

    # Section 4
    cells.append(nbf.v4.new_markdown_cell("""---
## 4. Exploratory Data Analysis (EDA)
We construct 8 high-impact visualizations examining price distributions, feature interactions, correlations, and location drivers.
"""))

    cells.append(nbf.v4.new_code_cell("""# Visualization 1: Price Distribution (Histogram + Boxplot)
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
sns.histplot(df_clean['Price'] / 1e5, kde=True, ax=ax[0], color='#1f77b4', bins=35)
ax[0].set_title('House Price Distribution (in Lakhs INR)', fontsize=13, fontweight='bold')
ax[0].set_xlabel('Price (₹ Lakhs)')
ax[0].set_ylabel('Frequency')

sns.boxplot(x=df_clean['Price'] / 1e5, ax=ax[1], color='#17becf')
ax[1].set_title('House Price Boxplot', fontsize=13, fontweight='bold')
ax[1].set_xlabel('Price (₹ Lakhs)')
plt.tight_layout()
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# Visualization 2: Correlation Heatmap
plt.figure(figsize=(9, 7))
numeric_vars = ['Area', 'BHK', 'Bathrooms', 'Property Age', 'luxury_score', 'price_per_sqft', 'Price']
corr = df_clean[numeric_vars].corr()
sns.heatmap(corr, annot=True, cmap='Blues', fmt='.2f', linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('Correlation Heatmap of Property Variables', fontsize=14, fontweight='bold', pad=12)
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# Visualization 3: Area vs Price with BHK hue & Trend Line
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_clean, x='Area', y=df_clean['Price'] / 1e5, hue='BHK', palette='viridis', alpha=0.65, s=50)
sns.regplot(data=df_clean, x='Area', y=df_clean['Price'] / 1e5, scatter=False, color='#e74c3c', line_kws={'linewidth': 2, 'label': 'Regression Trend'})
plt.title('Property Area (Sq.Ft) vs House Price (₹ Lakhs)', fontsize=14, fontweight='bold')
plt.xlabel('Area (Square Feet)')
plt.ylabel('Price (₹ Lakhs)')
plt.legend(title='BHK Configuration')
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# Visualization 4: BHK vs Price Distribution
plt.figure(figsize=(9, 6))
sns.boxplot(data=df_clean, x='BHK', y=df_clean['Price'] / 1e5, palette='Set2')
plt.title('Price Dispersion Across BHK Configurations', fontsize=14, fontweight='bold')
plt.xlabel('BHK')
plt.ylabel('Price (₹ Lakhs)')
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# Visualization 5: Bathroom Count vs Average Price
plt.figure(figsize=(9, 6))
sns.barplot(data=df_clean, x='Bathrooms', y=df_clean['Price'] / 1e5, estimator=np.mean, palette='mako', ci=None, edgecolor='black')
plt.title('Average House Price by Bathroom Count', fontsize=14, fontweight='bold')
plt.xlabel('Number of Bathrooms')
plt.ylabel('Average Price (₹ Lakhs)')
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# Visualization 6: Location-wise Average Price
plt.figure(figsize=(11, 7))
loc_order = df_clean.groupby('Location')['Price'].mean().sort_values(ascending=False).index
sns.barplot(data=df_clean, y='Location', x=df_clean['Price'] / 1e5, order=loc_order, palette='coolwarm', ci=None, edgecolor='black')
plt.title('Location-wise Average House Price (₹ Lakhs)', fontsize=14, fontweight='bold')
plt.xlabel('Average Price (₹ Lakhs)')
plt.ylabel('Location')
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# Visualization 7: Property Age vs Price
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_clean, x='Property Age', y=df_clean['Price'] / 1e5, alpha=0.5, color='#2980b9')
sns.regplot(data=df_clean, x='Property Age', y=df_clean['Price'] / 1e5, scatter=False, color='#c0392b', line_kws={'linewidth': 2.5, 'label': 'Depreciation Line'})
plt.title('Property Age vs House Price (₹ Lakhs)', fontsize=14, fontweight='bold')
plt.xlabel('Property Age (Years)')
plt.ylabel('Price (₹ Lakhs)')
plt.legend()
plt.show()"""))

    # Section 5 & 6
    cells.append(nbf.v4.new_markdown_cell("""---
## 5. Supervised Learning Models & 6. Model Evaluation
We construct a production-ready `ColumnTransformer` preprocessing pipeline and benchmark 4 algorithms:
1. **Linear Regression** (Ordinary Least Squares Baseline)
2. **Decision Tree Regressor** (Non-linear Rule Partitioning)
3. **Random Forest Regressor** (Ensemble Bagging)
4. **XGBoost Regressor** (Gradient Boosting Trees)

Metrics evaluated:
- **MAE** (Mean Absolute Error): $\\frac{1}{n} \\sum |y - \\hat{y}|$
- **MSE** (Mean Squared Error): $\\frac{1}{n} \\sum (y - \\hat{y})^2$
- **RMSE** (Root Mean Squared Error): $\\sqrt{\\text{MSE}}$
- **$R^2$ Score** (Coefficient of Determination)
- **5-Fold Cross Validation** (Mean $R^2$ ± Std)
"""))

    cells.append(nbf.v4.new_code_cell("""from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# Feature definitions
num_features = ['Area', 'BHK', 'Bathrooms', 'Property Age', 'luxury_score']
cat_features = ['Location', 'Furnishing Status']
feature_cols = num_features + cat_features

X = df_clean[feature_cols]
y = df_clean['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat_features)
    ]
)

preprocessor.fit(X_train)
X_train_trans = preprocessor.transform(X_train)
X_test_trans = preprocessor.transform(X_test)

# Candidate Models
models_dict = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(max_depth=12, min_samples_split=8, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=150, max_depth=16, min_samples_split=6, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.08, subsample=0.85, colsample_bytree=0.85, random_state=42)
}

results = []
trained_models = {}
cv = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models_dict.items():
    model.fit(X_train_trans, y_train)
    trained_models[name] = model
    y_pred = model.predict(X_test_trans)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    cv_scores = cross_val_score(model, X_train_trans, y_train, cv=cv, scoring='r2', n_jobs=-1)
    
    results.append({
        'Model': name,
        'MAE (₹)': round(mae, 2),
        'RMSE (₹)': round(rmse, 2),
        'R² Score': round(r2, 4),
        'CV R² Mean': round(cv_scores.mean(), 4),
        'CV R² Std': round(cv_scores.std(), 4)
    })

pd.DataFrame(results).sort_values(by='R² Score', ascending=False)"""))

    cells.append(nbf.v4.new_code_cell("""# Visualization 8: Feature Importance Plot
rf_model = trained_models['Random Forest']
cat_encoder_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features).tolist()
all_features = num_features + cat_encoder_names

feat_imp = pd.DataFrame({
    'Feature': all_features,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False).head(12)

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp, y='Feature', x='Importance', palette='magma', edgecolor='black')
plt.title('Top 12 Feature Importances (Random Forest)', fontsize=14, fontweight='bold')
plt.xlabel('Relative Importance')
plt.show()"""))

    # Section 7
    cells.append(nbf.v4.new_markdown_cell("""---
## 7. Unsupervised Learning (K-Means Clustering)
We cluster properties across Area, Price, and Luxury Score to automatically discover natural market segments:
- **Budget Tier**
- **Mid-Range Tier**
- **Luxury Tier**
"""))

    cells.append(nbf.v4.new_code_cell("""from sklearn.cluster import KMeans

cluster_features = ['Area', 'Price', 'luxury_score']
scaler_cluster = StandardScaler()
X_clust = scaler_cluster.fit_transform(df_clean[cluster_features])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_clean['cluster_id'] = kmeans.fit_predict(X_clust)

# Sort clusters by mean price to label Budget, Mid-Range, Luxury
ordered_clusters = df_clean.groupby('cluster_id')['Price'].mean().sort_values().index.tolist()
label_map = {
    ordered_clusters[0]: 'Budget',
    ordered_clusters[1]: 'Mid-Range',
    ordered_clusters[2]: 'Luxury'
}
df_clean['property_category'] = df_clean['cluster_id'].map(label_map)

# Visualization of Clusters
plt.figure(figsize=(10, 6))
category_order = ['Budget', 'Mid-Range', 'Luxury']
sns.scatterplot(
    data=df_clean,
    x='Area',
    y=df_clean['Price'] / 1e5,
    hue='property_category',
    hue_order=category_order,
    palette={'Budget': '#27ae60', 'Mid-Range': '#2980b9', 'Luxury': '#8e44ad'},
    alpha=0.7,
    s=60
)
plt.title('K-Means Property Market Segmentation (Budget, Mid-Range, Luxury)', fontsize=14, fontweight='bold')
plt.xlabel('Area (Sq.Ft)')
plt.ylabel('Price (₹ Lakhs)')
plt.legend(title='Category')
plt.show()"""))

    # Section 8
    cells.append(nbf.v4.new_markdown_cell("""---
## 8. Dimensionality Reduction (PCA)
Principal Component Analysis (PCA) orthogonalizes the multi-feature space, capturing maximal variance in the top 2 principal components.
"""))

    cells.append(nbf.v4.new_code_cell("""from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
pca_res = pca.fit_transform(X_clust)
df_clean['PCA1'] = pca_res[:, 0]
df_clean['PCA2'] = pca_res[:, 1]

print(f"Explained Variance Ratio: PC1={pca.explained_variance_ratio_[0]:.4f}, PC2={pca.explained_variance_ratio_[1]:.4f}")
print(f"Total Variance Retained: {sum(pca.explained_variance_ratio_)*100:.2f}%")

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_clean,
    x='PCA1',
    y='PCA2',
    hue='property_category',
    hue_order=category_order,
    palette={'Budget': '#27ae60', 'Mid-Range': '#2980b9', 'Luxury': '#8e44ad'},
    alpha=0.7,
    s=55
)
plt.title(f'PCA 2D Projection of Property Space (Total Variance: {sum(pca.explained_variance_ratio_)*100:.1f}%)', fontsize=14, fontweight='bold')
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)')
plt.show()"""))

    # Section 9
    cells.append(nbf.v4.new_markdown_cell("""---
## 9. Deep Learning with TensorFlow/Keras
We design and train a Deep Neural Network architecture:
- **Input Layer**: Transformed feature dimension
- **Dense(128, ReLU)** + BatchNormalization + Dropout(0.15)
- **Dense(64, ReLU)** + Dropout(0.1)
- **Dense(32, ReLU)**
- **Output Layer**: Dense(1, Linear)
"""))

    cells.append(nbf.v4.new_code_cell("""import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Scale target variable for stable gradient descent
target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

input_dim = X_train_trans.shape[1]

# Define Model Architecture
nn_model = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.15),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='linear')
])

nn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
    loss='mse',
    metrics=['mae']
)

es = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
lr_decay = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-5)

history = nn_model.fit(
    X_train_trans,
    y_train_scaled,
    validation_split=0.15,
    epochs=120,
    batch_size=32,
    callbacks=[es, lr_decay],
    verbose=0
)

# Evaluate Deep Learning Model
y_pred_nn_scaled = nn_model.predict(X_test_trans, verbose=0).flatten()
y_pred_nn = target_scaler.inverse_transform(y_pred_nn_scaled.reshape(-1, 1)).flatten()

dl_mae = mean_absolute_error(y_test, y_pred_nn)
dl_rmse = np.sqrt(mean_squared_error(y_test, y_pred_nn))
dl_r2 = r2_score(y_test, y_pred_nn)

print(f"Deep Neural Network Test Performance:")
print(f"MAE:  ₹{dl_mae:,.2f}")
print(f"RMSE: ₹{dl_rmse:,.2f}")
print(f"R²:   {dl_r2:.4f}")

# Plot Loss
plt.figure(figsize=(9, 5))
plt.plot(history.history['loss'], label='Training Loss (MSE)', color='#2980b9', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)', color='#e67e22', linewidth=2, linestyle='--')
plt.title('Neural Network Loss Curve (Train vs Validation)', fontsize=13, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Normalized Loss')
plt.legend()
plt.show()"""))

    # Section 10
    cells.append(nbf.v4.new_markdown_cell("""---
## 10. Model Selection & Joblib Serialization
We compare all models on the test set, select the highest-performing estimator, and serialize the trained pipeline to `models/house_price_model.pkl`.
"""))

    cells.append(nbf.v4.new_code_cell("""import joblib

# Comprehensive Leaderboard Comparison
all_eval = results.copy()
all_eval.append({
    'Model': 'Deep Neural Network (Keras)',
    'MAE (₹)': round(dl_mae, 2),
    'RMSE (₹)': round(dl_rmse, 2),
    'R² Score': round(dl_r2, 4),
    'CV R² Mean': 'N/A (Val Split)',
    'CV R² Std': 'N/A'
})

leaderboard_df = pd.DataFrame(all_eval).sort_values(by='R² Score', ascending=False)
display(leaderboard_df)

# Best Model Selection
best_model_name = leaderboard_df.iloc[0]['Model']
print(f"Selected Top Model: {best_model_name}")

# Build Full Production Pipeline
best_estimator = trained_models.get(best_model_name, trained_models['XGBoost'])
production_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', best_estimator)
])

production_pipeline.fit(X, y)

# Save Production Model
models_dir = os.path.join('..', 'models')
os.makedirs(models_dir, exist_ok=True)
joblib.dump(production_pipeline, os.path.join(models_dir, 'house_price_model.pkl'))
print(f"Successfully exported house_price_model.pkl to {models_dir}")"""))

    nb.cells = cells
    notebook_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks", "model_development.ipynb")
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook created at: {notebook_path}")

if __name__ == "__main__":
    build_notebook()
