# 🚀 Deployment Guide

Step-by-step instructions for pushing the project to GitHub and deploying the Streamlit app on Streamlit Community Cloud.

---

## Part 1: GitHub Setup & Push

### Step 1 — Install Git (if not installed)
Download from: https://git-scm.com/download/win  
Verify: `git --version`

### Step 2 — Configure Git Identity (one-time)
```bash
git config --global user.name  "Your Name"
git config --global user.email "your.email@example.com"
```

### Step 3 — Create GitHub Repository
1. Go to https://github.com → **New Repository**
2. Repository name: `House-Price-Prediction-And-Recommedation-System`
3. Set to **Public**
4. Do NOT initialize with README (we already have one)
5. Click **Create Repository**

### Step 4 — Initialize & Push from Local Machine

Open a terminal in the project root directory:
```bash
cd "c:\Users\AdithiP\OneDrive - CloudThat\Desktop\House Price Prediction And Recommedation System"

# Initialize git repository
git init

# Stage all files
git add .

# Create initial commit
git commit -m "feat: complete AI House Price Prediction & Recommendation System"

# Add your GitHub remote (replace YOUR_USERNAME)
git remote add origin https://github.com/AdithiP07/House-Price-Prediction-And-Recommedation-System.git

# Set default branch
git branch -M main

# Push to GitHub
git push -u origin main
```

> 💡 You will be prompted for your GitHub credentials.  
> Use a **Personal Access Token (PAT)** instead of your password:  
> GitHub → Settings → Developer Settings → Personal Access Tokens → Fine-grained tokens

### Step 5 — Verify Push
1. Go to your GitHub repository URL
2. Confirm all files are present, especially:
   - `streamlit_app/app.py`
   - `models/house_price_model.pkl`
   - `models/property_catalog.pkl`
   - `notebooks/model_development.ipynb`
   - `requirements.txt`
   - `README.md`

---

## Part 2: Streamlit Community Cloud Deployment

### Step 1 — Sign Up for Streamlit Cloud
1. Go to https://streamlit.io/cloud
2. Click **"Sign up"** → **"Continue with GitHub"**
3. Authorize Streamlit to access your repositories

### Step 2 — Deploy Your App
1. Click **"New app"** button (top-right)
2. Choose:
   - **Repository**: `AdithiP07/House-Price-Prediction-And-Recommedation-System`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app/app.py`
3. *(Optional)* Set App URL slug, e.g. `house-price-ai`
4. Click **"Deploy!"**

### Step 3 — Wait for Deployment
- Streamlit will install dependencies from `streamlit_app/requirements.txt`
- Deployment typically takes 2–5 minutes
- You'll see a live progress log on screen

### Step 4 — Get Your App URL
Once deployed, your app URL will look like:
```
https://YOUR-USERNAME-house-price-ai-app-XXXXX.streamlit.app
```

### Step 5 — Add URL to Notebook and App
1. Open `notebooks/model_development.ipynb` → **Section 13** → replace:
   ```
   *(Paste your Streamlit Cloud URL here after deployment)*
   ```
   with your actual URL.

2. Open `streamlit_app/app.py` → **Tab 3 (About)** → replace:
   ```python
   > **Hosted Application URL**: *(deploy and paste URL here)*
   ```
   with your actual URL.

3. Commit and push the update:
   ```bash
   git add .
   git commit -m "docs: add Streamlit Cloud deployment URL"
   git push
   ```

---

## Part 3: Testing the Deployed Application

### Test Checklist
- [ ] App loads without errors at the Streamlit Cloud URL
- [ ] Sidebar inputs are interactive (Location dropdown, Area slider, BHK select, etc.)
- [ ] Clicking "Predict & Recommend" returns a price prediction
- [ ] Price hero card displays ₹ value, price/sq.ft, and market tier
- [ ] Top 5 Similar Properties cards are rendered
- [ ] Model Analytics tab shows leaderboard and feature importance chart
- [ ] About tab displays project details and links

### Troubleshooting
| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'xgboost'` | Check `streamlit_app/requirements.txt` includes `xgboost>=2.0.0` |
| Model not found error | Ensure `models/*.pkl` files are committed to GitHub (not in `.gitignore`) |
| App crashes on prediction | Check console logs; verify `house_price_model.pkl` was trained with same sklearn version |
| Slow loading | Model files are large; first cold start may take 30–60 seconds |

---

## Part 4: Taking Screenshots

### Using the Browser (Recommended)

1. **Launch Streamlit locally**:
   ```bash
   streamlit run streamlit_app/app.py
   ```

2. Open `http://localhost:8501` in Chrome or Edge

3. **Capture each view**:
   - **Home/Landing**: `screenshots/streamlit_home.png` — Sidebar + landing placeholder
   - **Prediction Result**: `screenshots/streamlit_prediction.png` — After clicking Predict
   - **Recommendations**: `screenshots/streamlit_recommendations.png` — Scroll to Top-5 cards
   - **Analytics Tab**: `screenshots/streamlit_analytics.png` — Click 📊 Model Analytics tab

4. Use `Ctrl+Shift+S` (Edge/Chrome) or screenshot tool to capture full-page screenshots

5. Save to `screenshots/` directory and commit:
   ```bash
   git add screenshots/
   git commit -m "docs: add application screenshots"
   git push
   ```

---

## Quick Reference

```
📂 Repository  : https://github.com/AdithiP07/House-Price-Prediction-And-Recommedation-System
🚀 Streamlit   : https://YOUR-APP.streamlit.app
📓 Notebook    : notebooks/model_development.ipynb
🤖 Best Model  : models/house_price_model.pkl (XGBoost, R²=0.9506)
```
