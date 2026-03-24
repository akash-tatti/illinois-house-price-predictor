# 🏠 Illinois House Price Predictor
### End-to-End Machine Learning Web App built with Streamlit

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange)
![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-purple)

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
python -m pip install -r requirements.txt

# 2. Run the app
python -m streamlit run app.py

# 3. Open in browser → http://localhost:8501
```

---

## 📌 What This App Does

A full data science project predicting home prices across **20 real Illinois regions** — from Chicago's North Side neighborhoods to downstate cities like Springfield and Rockford. Built as a portfolio project for Data Analyst / Data Scientist roles.

---

## 🗺️ Illinois Regions Covered

| Category | Regions |
|---|---|
| **Chicago Neighborhoods** | Lincoln Park, River North, Wicker Park, Logan Square, Hyde Park, South Loop, Pilsen |
| **Inner Suburbs** | Evanston, Naperville, Oak Park, Schaumburg, Arlington Heights |
| **Outer / Downstate** | Joliet, Aurora, Elgin, Rockford, Springfield, Peoria, Champaign, Bloomington |

---

## 📊 App Pages

### 1. Overview & EDA
- Price distribution with median and mean markers
- Median home price bar chart ranked by region
- Interactive geographic map of Illinois colored by price
- Square footage vs price scatter plot
- Price by bedrooms box plot
- Full feature correlation heatmap

### 2. Model Training
- 5 models trained and compared side-by-side
- Model comparison table (R², MAE, RMSE, CV R²)
- Actual vs Predicted chart — selectable by model
- Top 12 feature importances (Gradient Boosting)
- Residual analysis charts (scatter + distribution)

### 3. Live Price Predictor
- Select any of 20 Illinois regions
- Adjust 10 home features with sliders
- Instant price estimate with confidence range
- Side-by-side comparison of all 5 model predictions

### 4. How It Works
- Step-by-step pipeline explanation
- Ready-to-copy resume bullet points

---

## 🤖 Models

| Model | Type | Notes |
|---|---|---|
| Linear Regression | Baseline | Simple, interpretable |
| Ridge Regression | Regularized linear | Handles correlated features |
| Lasso Regression | Regularized linear | Auto feature selection |
| Random Forest | Ensemble (120 trees) | Captures non-linear patterns |
| Gradient Boosting | Ensemble (150 trees) | Usually best performer |

---

## 🏡 Features Used

| Feature | Description |
|---|---|
| `sqft` | Square footage of the home |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `house_age` | Derived from year built (2024 - year) |
| `garage_spaces` | 0, 1, or 2 car garage |
| `has_basement` | Yes/No — very common in Midwest homes |
| `school_rating` | District school rating (1–10) |
| `property_tax` | Tax rate % — Illinois avg ~2.2% (one of US's highest) |
| `lot_size` | Lot size in square feet |
| `latitude / longitude` | Real coordinates per city |
| `is_chicago` | Binary flag: Chicago neighborhood vs. rest of IL |
| `region` | One-hot encoded — 20 Illinois regions |

---

## 📦 Tech Stack

```
Python 3.8+
Streamlit       — web app framework
Plotly          — interactive charts & maps
Pandas          — data manipulation
NumPy           — numerical computing
Scikit-learn    — machine learning models
```

---

## 📁 Project Structure

```
House Price Predictor/
│
├── app.py              ← main Streamlit application
├── requirements.txt    ← Python dependencies
└── README.md           ← this file
```

---

## 📄 requirements.txt

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
scikit-learn>=1.3.0
```

---

## 💼 Resume Bullet Points

> Copy and paste these directly into your resume:

- Built an end-to-end ML web app in **Streamlit** predicting home prices across 20 Illinois regions including Chicago neighborhoods, suburbs (Naperville, Evanston, Oak Park), and downstate cities (Springfield, Rockford, Peoria)

- Engineered Illinois-specific features including property tax rate, basement presence, school district rating, and urban/suburban classification to improve model accuracy

- Trained and compared **5 regression models** (Linear, Ridge, Lasso, Random Forest, Gradient Boosting) with 5-fold cross-validation and full diagnostic reporting

- Built interactive **Plotly** dashboards: geographic Illinois price map, regional comparisons, correlation matrix, feature importances, and residual analysis

- Deployed a live prediction tool allowing users to estimate home values for any Illinois region with real-time comparison across all 5 trained models

---

## 🧠 Key Illinois Market Insights (from EDA)

- **Lincoln Park & River North** are the most expensive regions — median prices 3–4× higher than downstate
- **Naperville** is the priciest suburb, driven by top school ratings and large lot sizes
- **Property tax** is a significant price depressor — IL's ~2.2% rate is among the highest in the US
- **Basements** add ~$20K in value and are present in ~70% of Illinois homes
- **Rockford and Peoria** offer the lowest price points — ideal for affordability comparisons

---

*Built by Akash | Portfolio Project | Data Analyst*
