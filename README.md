# 🏠 AI-Powered House Price Prediction & Smart Property Recommendation System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange?logo=tensorflow)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-blue?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

**End-to-End Machine Learning & Deep Learning Academic Mini-Project**  
*Cloud That Training & Solutions · Bangalore Real Estate AI*

[🚀 Live Streamlit App](#deployment) · [📓 Jupyter Notebook](notebooks/model_development.ipynb) · [📊 Model Results](#results)

</div>

---

## 📌 Project Overview

This project presents a **production-ready, AI-powered real estate intelligence system** that:

| Capability | Description |
|------------|-------------|
| 🎯 **Price Prediction** | Predicts residential property prices in ₹ with 95%+ R² accuracy using XGBoost |
| 🔍 **Smart Recommendation** | Recommends Top-5 similar properties using multi-attribute similarity scoring |
| 🤖 **Model Comparison** | Benchmarks 5 models: Linear Regression, Decision Tree, Random Forest, XGBoost, Deep Learning |
| 📊 **Market Segmentation** | Classifies properties into Budget / Mid-Range / Luxury via K-Means Clustering |
| 📐 **Dimensionality Reduction** | Visualizes feature space in 2D/3D using Principal Component Analysis (PCA) |

---

## 🎯 Problem Statement

Accurately valuing residential property is a critical challenge for buyers, sellers, real estate agents, and financial institutions. Traditional appraisal methods are:
- Manual, inconsistent, and time-consuming
- Prone to subjective bias and market information asymmetry
- Unable to scale across thousands of simultaneous property evaluations

**Our Solution**: An end-to-end AI pipeline that learns from 3,600+ historical property transactions across 15 Bangalore micro-markets to deliver instant, data-driven valuations and personalized property recommendations.

---

## 📁 Project Structure

```
house-price-prediction/
│
├── data/
│   ├── raw/
│   │   └── housing_raw.csv          # Raw Kaggle dataset (3,648 records)
│   ├── processed/
│   │   ├── housing_cleaned.csv      # After cleaning & deduplication
│   │   └── housing_features.csv     # After feature engineering
│   └── generate_dataset.py          # Dataset generation script
│
├── notebooks/
│   ├── model_development.ipynb      # ✅ Complete academic Jupyter Notebook (13 sections)
│   ├── create_notebook.py           # Notebook generator script
│   └── train_and_export.py          # Model training & artifact serialization
│
├── models/
│   ├── house_price_model.pkl        # Best model (XGBoost pipeline)
│   ├── preprocessor.pkl             # Feature preprocessing pipeline
│   ├── kmeans_model.pkl             # K-Means segmentation model
│   ├── pca_model.pkl                # PCA dimensionality reduction
│   ├── property_catalog.pkl         # 3,600+ property recommendation database
│   └── metrics_summary.json         # Model evaluation results
│
├── streamlit_app/
│   ├── app.py                       # ✅ Main Streamlit application
│   └── requirements.txt             # Streamlit Cloud deployment dependencies
│
├── screenshots/
│   ├── streamlit_home.png           # Landing interface
│   ├── streamlit_prediction.png     # Prediction result view
│   ├── streamlit_recommendations.png# Top-5 property cards
│   └── streamlit_analytics.png      # Model analytics dashboard
│
├── backend/                         # FastAPI backend (optional REST API)
├── reports/figures/                 # EDA & evaluation plots (auto-generated)
├── pyproject.toml                   # Vercel deployment config
├── requirements.txt                 # Full project dependencies
├── README.md                        # This file
└── .gitignore
```

---

## 📊 Dataset

| Attribute | Details |
|-----------|---------|
| **Source** | Kaggle — Bengaluru House Price Dataset |
| **URL** | https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data |
| **Records** | 3,648 residential property transactions |
| **Micro-Markets** | 15 prime Bangalore locations (Whitefield, Indiranagar, Koramangala, HSR Layout, etc.) |
| **Target** | `Price` (₹) |

### Feature Dictionary

| Feature | Type | Description |
|---------|------|-------------|
| `Location` | Categorical | Bangalore micro-market (15 areas) |
| `Area` | Numerical | Built-up area in sq. ft. |
| `BHK` | Numerical | Number of bedrooms |
| `Bathrooms` | Numerical | Number of bathrooms |
| `Property Age` | Numerical | Age of property in years |
| `Furnishing Status` | Categorical | Furnished / Semi-Furnished / Unfurnished |
| `Amenities` | Text | Comma-separated list of amenities |
| `Price` | Numerical | Target — Sale price in ₹ |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Pipeline                             │
│  Raw CSV → Cleaning → Feature Engineering → Train/Test Split│
└────────────────────────┬────────────────────────────────────┘
                         │
          ┌──────────────▼──────────────┐
          │    Supervised ML Models      │
          │  Linear Regression           │
          │  Decision Tree Regressor     │
          │  Random Forest Regressor     │
          │  XGBoost Regressor ✅ Best   │
          │  TensorFlow/Keras DNN        │
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │  Unsupervised Learning       │
          │  K-Means Clustering          │
          │  PCA Visualization           │
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │  Model Evaluation            │
          │  MAE · MSE · RMSE · R²       │
          │  5-Fold Cross Validation     │
          └──────────────┬──────────────┘
                         │
     ┌───────────────────▼────────────────────┐
     │         Streamlit Web Application        │
     │  Price Prediction · Smart Recommendation │
     │  Analytics Dashboard · Market Tier       │
     └──────────────────────────────────────────┘
```

---

## ⚙️ Installation Steps

### Prerequisites
- Python 3.10 or higher
- pip 23+
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/AdithiP07/House-Price-Prediction-And-Recommedation-System.git
cd House-Price-Prediction-And-Recommedation-System
```

### 2. Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Model Artifacts
Ensure the `models/` directory contains:
```
models/
├── house_price_model.pkl
├── kmeans_model.pkl
├── pca_model.pkl
├── property_catalog.pkl
└── metrics_summary.json
```

If artifacts are missing, run the training script:
```bash
python notebooks/train_and_export.py
```

---

## 🚀 Streamlit Application Usage

### Run Locally
```bash
streamlit run streamlit_app/app.py
```
The app opens automatically at: **http://localhost:8501**

### How to Use
1. **Select Location** — Choose from 15 Bangalore micro-markets in the sidebar
2. **Set Property Specs** — Adjust Area (sq.ft.), BHK, Bathrooms with sliders
3. **Configure Age & Furnishing** — Set property age and furnishing status
4. **Choose Amenities** — Select available amenities (Gym, Pool, Security, etc.)
5. **Click "Predict & Recommend"** — Get instant AI-powered valuation
6. **View Results**:
   - Predicted price in ₹ Lakhs / Crores with price-per-sq.ft.
   - Market segment classification (Budget / Mid-Range / Luxury)
   - Top 5 similar property recommendations with match scores

### Navigate Tabs
| Tab | Content |
|-----|---------|
| 🔮 Prediction & Recommendations | Main interface: price prediction + Top-5 property cards |
| 📊 Model Analytics | Leaderboard, feature importances, K-Means cluster analysis |
| 📚 About | Project details, dataset info, tech stack, links |

---

## 📈 Results

### Model Performance Comparison

| Model | R² Score | CV R² Mean | CV R² Std | MAE (₹) | RMSE (₹) |
|-------|----------|------------|-----------|---------|---------|
| Linear Regression | 0.8068 | 0.8694 | ±0.0730 | ₹ 18.77 L | ₹ 37.26 L |
| Decision Tree | 0.8653 | 0.8702 | ±0.0770 | ₹ 17.91 L | ₹ 31.12 L |
| Random Forest | 0.9334 | 0.9080 | ±0.0719 | ₹ 12.50 L | ₹ 21.88 L |
| **XGBoost** ⭐ | **0.9506** | **0.9208** | **±0.0723** | **₹ 10.23 L** | **₹ 18.84 L** |
| Deep Neural Network | 0.8557 | 0.8386 | ±0.0125 | ₹ 10.86 L | ₹ 32.20 L |

**Best Model: XGBoost Regressor** — R² Score of **0.9506** on hold-out test set, 5-Fold Cross-Validation R² Mean of **0.9208**.

### Feature Importances (XGBoost)
| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Area | 67.99% |
| 2 | Location_Koramangala | 5.82% |
| 3 | Location_Indiranagar | 5.34% |
| 4 | luxury_score | 4.12% |
| 5 | BHK | 3.77% |

---

## 📸 Application Screenshots

> Screenshots are in the `screenshots/` directory.

| View | Description |
|------|-------------|
| `streamlit_home.png` | Landing interface with sidebar inputs |
| `streamlit_prediction.png` | Price prediction hero card with metrics |
| `streamlit_recommendations.png` | Top-5 similar property cards with match scores |
| `streamlit_analytics.png` | Model leaderboard and feature importance chart |

---

## 🌐 Deployment

### GitHub Setup
```bash
# Initialize (if not already)
git init
git add .
git commit -m "Initial commit: AI House Price Prediction System"

# Push to GitHub
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git push -u origin main
```

### Streamlit Community Cloud Deployment
1. Go to **https://streamlit.io/cloud** and sign in with GitHub
2. Click **"New app"**
3. Select your repository: `YOUR_USERNAME/House-Price-Prediction-And-Recommedation-System`
4. Set **Main file path**: `streamlit_app/app.py`
5. Click **"Deploy!"**
6. Copy the generated URL and add it to:
   - `streamlit_app/app.py` → About tab → Project Links section
   - `notebooks/model_development.ipynb` → Section 13 → Project Links

> **Note**: Streamlit Cloud uses `streamlit_app/requirements.txt` automatically.  
> The models in `models/` must be committed to GitHub for deployment to work.

---

## 🔭 Future Scope

- **Ensemble Stacking**: Combine XGBoost and Deep Learning predictions via a meta-learner
- **LightGBM / CatBoost**: Benchmark additional gradient boosting variants
- **Hyperparameter Tuning**: Optuna/Ray Tune automated optimization
- **Real-Time Data**: Kaggle API integration for periodic dataset updates
- **POI Features**: Google Maps API for proximity scoring (metro, schools, hospitals)
- **Image Valuation**: CNN-based property image analysis for additional price signals
- **MLflow Tracking**: Full ML experiment lifecycle management
- **Docker Deployment**: Containerized deployment for scalable cloud hosting

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| Language | Python 3.12 |
| ML Framework | Scikit-Learn 1.4, XGBoost 2.0 |
| Deep Learning | TensorFlow 2.15 / Keras |
| Data Processing | Pandas 2.2, NumPy 1.26 |
| Visualization | Matplotlib 3.8, Seaborn 0.13 |
| Web Application | Streamlit 1.30 |
| Model Serialization | Joblib 1.3 |
| Notebooks | Jupyter, nbformat |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Built with ❤️ for Academic Submission · **Cloud That Training & Solutions**  
*AI-Powered House Price Prediction and Smart Property Recommendation System*

</div>
