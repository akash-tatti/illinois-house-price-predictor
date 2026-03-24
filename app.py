"""
============================================================
  ILLINOIS HOUSE PRICE PREDICTOR — Streamlit Web App
  Built by Akash Tatti
  Stack: Streamlit · Plotly · Pandas · Scikit-learn
============================================================
  Run with:  python -m streamlit run app.py
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Illinois House Price Predictor | Akash Tatti",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
  }
  #MainMenu, footer, header { visibility: hidden; }
  .stApp { background: #0d1117; }

  [data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #21262d;
  }

  [data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 16px !important;
  }
  [data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace;
    color: #58a6ff;
    font-size: 26px !important;
  }
  [data-testid="stMetricLabel"] { color: #8b949e; font-size: 13px; }

  .hero {
    background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
    border: 1px solid #21262d;
    border-radius: 16px;
    padding: 32px 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, #1f6feb33 0%, transparent 70%);
    border-radius: 50%;
  }
  .hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 28px;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 8px 0;
  }
  .hero-sub { color: #8b949e; font-size: 15px; margin: 0; }
  .tag {
    display: inline-block;
    background: #1f6feb22;
    border: 1px solid #1f6feb55;
    color: #58a6ff;
    border-radius: 20px;
    font-size: 12px;
    font-family: 'Space Mono', monospace;
    padding: 3px 12px;
    margin: 8px 4px 0 0;
  }
  .section-title {
    font-family: 'Space Mono', monospace;
    font-size: 20px;
    font-weight: 700;
    color: #58a6ff;
    border-bottom: 2px solid #21262d;
    padding-bottom: 10px;
    margin-bottom: 20px;
  }
  .pred-box {
    background: linear-gradient(135deg, #1f6feb22, #388bfd11);
    border: 1.5px solid #1f6feb66;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
  }
  .pred-label {
    font-size: 12px; color: #8b949e;
    text-transform: uppercase; letter-spacing: 1px;
    font-family: 'Space Mono', monospace;
  }
  .pred-value {
    font-family: 'Space Mono', monospace;
    font-size: 46px; font-weight: 700;
    color: #58a6ff; line-height: 1.1; margin: 8px 0 0 0;
  }
  .pred-range { color: #3fb950; font-size: 14px; margin-top: 8px; }
  .info-card {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 10px; padding: 16px 20px; margin-bottom: 12px;
  }
  .info-card h4 { margin: 0 0 6px 0; color: #c9d1d9; font-size: 14px; }
  .info-card p  { margin: 0; color: #8b949e; font-size: 13px; line-height: 1.6; }

  /* Author card in sidebar */
  .author-card {
    background: linear-gradient(135deg, #1f6feb15, #1f6feb08);
    border: 1px solid #1f6feb33;
    border-radius: 12px;
    padding: 14px 16px;
    margin-top: 16px;
    text-align: center;
  }
  .author-avatar {
    width: 52px; height: 52px;
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
    margin: 0 auto 10px auto;
  }
  .author-name {
    font-family: 'Space Mono', monospace;
    font-size: 14px; font-weight: 700;
    color: #ffffff; margin-bottom: 2px;
  }
  .author-title {
    font-size: 11px; color: #8b949e;
  }

  /* Footer */
  .footer {
    margin-top: 48px;
    padding: 24px 0 8px 0;
    border-top: 1px solid #21262d;
    text-align: center;
    color: #8b949e;
    font-size: 13px;
  }
  .footer span { color: #58a6ff; font-weight: 600; }

  /* Watermark badge */
  .built-by {
    display: inline-block;
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 12px;
    color: #8b949e;
    font-family: 'Space Mono', monospace;
    margin-top: 10px;
  }
  .built-by span { color: #58a6ff; }

  .stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color: white; border: none; border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 14px; padding: 10px 28px;
    font-weight: 700; width: 100%;
  }
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#161b22", plot_bgcolor="#161b22",
    font=dict(family="DM Sans", color="#c9d1d9", size=12),
    xaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d", tickfont=dict(color="#8b949e")),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d", tickfont=dict(color="#8b949e")),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(bgcolor="#0d1117", bordercolor="#21262d", borderwidth=1),
)
COLORS = ["#58a6ff", "#3fb950", "#f0883e", "#d2a8ff", "#ff7b72", "#ffa657"]

# ── Illinois dataset ──────────────────────────────────────────────────────────
@st.cache_data
def load_illinois_data():
    np.random.seed(42)
    n = 2000

    regions = {
        "Lincoln Park":      (41.921, -87.644, 620_000),
        "River North":       (41.893, -87.634, 580_000),
        "Wicker Park":       (41.908, -87.677, 490_000),
        "Logan Square":      (41.921, -87.707, 430_000),
        "Hyde Park":         (41.795, -87.596, 340_000),
        "South Loop":        (41.862, -87.627, 450_000),
        "Pilsen":            (41.855, -87.658, 310_000),
        "Evanston":          (42.045, -87.688, 480_000),
        "Oak Park":          (41.885, -87.794, 420_000),
        "Naperville":        (41.785, -88.147, 460_000),
        "Schaumburg":        (42.033, -88.083, 360_000),
        "Arlington Heights": (42.088, -87.980, 390_000),
        "Joliet":            (41.525, -88.082, 220_000),
        "Rockford":          (42.271, -89.094, 160_000),
        "Springfield":       (39.798, -89.654, 175_000),
        "Peoria":            (40.693, -89.589, 155_000),
        "Champaign":         (40.116, -88.243, 195_000),
        "Bloomington":       (40.484, -88.994, 210_000),
        "Aurora":            (41.760, -88.320, 270_000),
        "Elgin":             (42.037, -88.281, 255_000),
    }

    region_names = list(regions.keys())
    region_probs = np.array([
        0.08, 0.07, 0.07, 0.06, 0.05, 0.06, 0.05,
        0.06, 0.05, 0.07, 0.05, 0.05,
        0.04, 0.03, 0.03, 0.03, 0.03, 0.03, 0.04, 0.04
    ])
    region_probs /= region_probs.sum()

    chosen  = np.random.choice(region_names, size=n, p=region_probs)
    lats    = np.array([regions[r][0] for r in chosen]) + np.random.normal(0, 0.015, n)
    lons    = np.array([regions[r][1] for r in chosen]) + np.random.normal(0, 0.015, n)
    base    = np.array([regions[r][2] for r in chosen], dtype=float)

    sqft        = np.random.randint(500, 5000, n)
    bedrooms    = np.random.choice([1, 2, 3, 4, 5], n, p=[0.10, 0.25, 0.38, 0.20, 0.07])
    bathrooms   = np.clip(bedrooms - 1 + np.random.choice([0, 1], n, p=[0.45, 0.55]), 1, 5)
    garage      = np.random.choice([0, 1, 2], n, p=[0.25, 0.40, 0.35])
    year_built  = np.random.randint(1900, 2024, n)
    school_rate = np.clip(np.random.normal(6.8, 1.5, n), 1, 10).round(1)
    tax_rate    = np.clip(np.random.normal(2.2, 0.5, n), 0.8, 4.5).round(2)
    basement    = np.random.choice([0, 1], n, p=[0.30, 0.70])
    lot_size    = np.clip(np.random.exponential(6500, n), 1000, 40000).round(0)

    price = (
        base
        + sqft        * np.random.uniform(80, 140, n)
        + bedrooms    * 7_000
        + bathrooms   * 10_000
        + (2024 - year_built) * -500
        + garage      * 15_000
        + basement    * 20_000
        + school_rate * 8_000
        - tax_rate    * 5_000
        + lot_size    * 2.5
        + np.random.normal(0, 20_000, n)
    )
    price = np.clip(price, 60_000, 1_500_000).round(-2)

    df = pd.DataFrame({
        "price":         price,
        "sqft":          sqft,
        "bedrooms":      bedrooms,
        "bathrooms":     bathrooms.astype(int),
        "year_built":    year_built,
        "garage_spaces": garage,
        "has_basement":  basement,
        "school_rating": school_rate,
        "property_tax":  tax_rate,
        "lot_size":      lot_size,
        "latitude":      lats,
        "longitude":     lons,
        "region":        chosen,
    })

    q_low  = df["price"].quantile(0.01)
    q_high = df["price"].quantile(0.99)
    df = df[(df["price"] >= q_low) & (df["price"] <= q_high)].reset_index(drop=True)

    df["house_age"]      = 2024 - df["year_built"]
    df["price_per_sqft"] = df["price"] / df["sqft"]
    df["is_chicago"]     = df["region"].isin([
        "Lincoln Park", "River North", "Wicker Park",
        "Logan Square", "Hyde Park", "South Loop", "Pilsen"
    ]).astype(int)

    return df


@st.cache_resource
def train_models(df):
    encode_df    = pd.get_dummies(df[["region"]], drop_first=True)
    feature_cols = ["sqft", "bedrooms", "bathrooms", "house_age", "garage_spaces",
                    "has_basement", "school_rating", "property_tax", "lot_size",
                    "latitude", "longitude", "is_chicago"]
    X = pd.concat([df[feature_cols], encode_df], axis=1)
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    model_defs = {
        "Linear Regression": (LinearRegression(), True),
        "Ridge Regression":  (Ridge(alpha=10), True),
        "Lasso Regression":  (Lasso(alpha=100), True),
        "Random Forest":     (RandomForestRegressor(n_estimators=120, max_depth=14,
                                                     min_samples_leaf=3, random_state=42), False),
        "Gradient Boosting": (GradientBoostingRegressor(n_estimators=150, learning_rate=0.08,
                                                         max_depth=5, random_state=42), False),
    }

    results, trained = {}, {}
    for name, (model, use_scaled) in model_defs.items():
        Xtr = X_tr_s if use_scaled else X_train.values
        Xte = X_te_s if use_scaled else X_test.values
        model.fit(Xtr, y_train)
        preds = model.predict(Xte)
        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2   = r2_score(y_test, preds)
        cv   = cross_val_score(model, Xtr, y_train, cv=5, scoring="r2").mean()
        results[name] = dict(MAE=mae, RMSE=rmse, R2=r2, CV_R2=cv, preds=preds)
        trained[name] = (model, scaler if use_scaled else None, encode_df.columns.tolist())

    return trained, results, X_test, y_test, feature_cols, encode_df.columns.tolist()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # App title
    st.markdown("""
    <div style="padding:16px 0 8px 0;">
      <div style="font-family:'Space Mono',monospace;font-size:17px;font-weight:700;color:#58a6ff;">
        🏠 Illinois Housing
      </div>
      <div style="color:#8b949e;font-size:12px;margin-top:2px;">ML Price Predictor</div>
    </div>
    """, unsafe_allow_html=True)

    # Author card
    st.markdown("""
    <div class="author-card">
      <div class="author-avatar">👨‍💻</div>
      <div class="author-name">Akash Tatti</div>
      <div class="author-title">Data Analyst · ML Developer</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "📊  Overview & EDA",
        "🤖  Model Training",
        "🔮  Predict Price",
        "📖  How It Works",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""
    <div style="color:#8b949e;font-size:12px;line-height:1.8;">
      <b style="color:#c9d1d9;">Chicago Neighborhoods</b><br>
      Lincoln Park · River North<br>
      Wicker Park · Logan Square<br>
      Hyde Park · South Loop · Pilsen<br><br>
      <b style="color:#c9d1d9;">Suburbs</b><br>
      Evanston · Naperville · Oak Park<br>
      Schaumburg · Arlington Heights<br><br>
      <b style="color:#c9d1d9;">Downstate</b><br>
      Joliet · Aurora · Elgin<br>
      Rockford · Springfield · Peoria<br>
      Champaign · Bloomington
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="color:#8b949e;font-size:11px;text-align:center;line-height:1.7;">
      Built with ❤️ by <span style="color:#58a6ff;font-weight:600;">Akash Tatti</span><br>
      © 2025 · MIT License
    </div>
    """, unsafe_allow_html=True)


# ── Load data & train ─────────────────────────────────────────────────────────
with st.spinner("Building Illinois housing dataset..."):
    df = load_illinois_data()

with st.spinner("Training 5 ML models on Illinois data..."):
    trained_models, model_results, X_test, y_test, FEATURES, ENC_COLS = train_models(df)

best_name = max(model_results, key=lambda k: model_results[k]["R2"])

# ── Shared footer function ────────────────────────────────────────────────────
def render_footer():
    st.markdown("""
    <div class="footer">
      Illinois House Price Predictor &nbsp;·&nbsp;
      Built by <span>Akash Tatti</span> &nbsp;·&nbsp;
      Python · Streamlit · Scikit-learn · Plotly &nbsp;·&nbsp;
      © 2025 MIT License
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW & EDA
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊  Overview & EDA":
    st.markdown("""
    <div class="hero">
      <div class="hero-title">🏙️ Illinois Housing Market Analysis</div>
      <p class="hero-sub">Exploring home prices across Chicago neighborhoods, suburbs, and downstate Illinois cities.</p>
      <span class="tag">Chicago</span>
      <span class="tag">Naperville</span>
      <span class="tag">Evanston</span>
      <span class="tag">20 Regions</span>
      <span class="tag">2,000 Homes</span>
      <br>
      <div class="built-by">Built by <span>Akash Tatti</span></div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Homes",     f"{len(df):,}")
    c2.metric("Median Price",    f"${df['price'].median()/1000:.0f}K")
    c3.metric("Avg Sq Footage",  f"{df['sqft'].mean():,.0f} sqft")
    c4.metric("Regions Covered", "20")

    st.markdown("---")
    st.markdown('<div class="section-title">Price Distribution</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.histogram(df, x="price", nbins=70,
                           color_discrete_sequence=[COLORS[0]],
                           labels={"price": "Home Price (USD)"})
        fig.add_vline(x=df["price"].median(), line_dash="dash", line_color=COLORS[1],
                      annotation_text=f"Median ${df['price'].median()/1000:.0f}K",
                      annotation_font_color=COLORS[1])
        fig.add_vline(x=df["price"].mean(), line_dash="dash", line_color=COLORS[2],
                      annotation_text=f"Mean ${df['price'].mean()/1000:.0f}K",
                      annotation_font_color=COLORS[2])
        fig.update_layout(**PLOTLY_LAYOUT, title="Distribution of Illinois Home Prices",
                          bargap=0.03, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        stats = df["price"].describe()
        for label, key in [("Min", "min"), ("25th %ile", "25%"),
                            ("Median", "50%"), ("75th %ile", "75%"), ("Max", "max")]:
            st.metric(label, f"${stats[key]/1000:.0f}K")

    st.markdown("---")
    st.markdown('<div class="section-title">Median Price by Region</div>', unsafe_allow_html=True)

    reg_med = df.groupby("region")["price"].median().sort_values(ascending=True).reset_index()
    reg_med["color"] = [
        COLORS[1] if r in ["Lincoln Park", "River North", "Naperville", "Evanston", "Wicker Park"]
        else COLORS[0] for r in reg_med["region"]
    ]
    fig_reg = go.Figure(go.Bar(
        x=reg_med["price"] / 1000, y=reg_med["region"],
        orientation="h", marker_color=reg_med["color"],
        text=[f"${v:.0f}K" for v in reg_med["price"] / 1000],
        textposition="outside", textfont=dict(color="white", size=11),
    ))
    fig_reg.update_layout(**PLOTLY_LAYOUT, title="Median Home Price by Illinois Region",
                          xaxis_title="Median Price ($000s)", height=540, showlegend=False)
    st.plotly_chart(fig_reg, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">🗺️ Geographic Price Map — Illinois</div>', unsafe_allow_html=True)

    fig_map = px.scatter_mapbox(
        df, lat="latitude", lon="longitude",
        color="price", size="sqft",
        color_continuous_scale="Blues",
        size_max=10, zoom=5,
        center={"lat": 41.0, "lon": -89.2},
        hover_data={"price": "$,.0f", "sqft": True, "bedrooms": True,
                    "region": True, "latitude": False, "longitude": False},
        mapbox_style="carto-darkmatter",
        labels={"price": "Price ($)"},
    )
    fig_map.update_layout(
        paper_bgcolor="#161b22", margin=dict(l=0, r=0, t=10, b=0), height=520,
        coloraxis_colorbar=dict(title="Price ($)", tickformat="$,.0f",
                                tickfont=dict(color="#8b949e"),
                                title_font=dict(color="#c9d1d9")),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Feature Relationships</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        sample1 = df.sample(1000, random_state=1)
        m1, b1  = np.polyfit(sample1["sqft"], sample1["price"], 1)
        xs1     = np.linspace(sample1["sqft"].min(), sample1["sqft"].max(), 200)
        fig2    = go.Figure()
        fig2.add_trace(go.Scatter(x=sample1["sqft"], y=sample1["price"], mode="markers",
                                  marker=dict(color=COLORS[0], opacity=0.4, size=4), name="Homes"))
        fig2.add_trace(go.Scatter(x=xs1, y=m1 * xs1 + b1, mode="lines",
                                  line=dict(color=COLORS[2], width=2), name="Trend"))
        fig2.update_layout(**PLOTLY_LAYOUT, title="Square Footage vs Price",
                           xaxis_title="Square Footage", yaxis_title="Price ($)", showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        sample2 = df.sample(1000, random_state=2)
        m2, b2  = np.polyfit(sample2["school_rating"], sample2["price"], 1)
        xs2     = np.linspace(sample2["school_rating"].min(), sample2["school_rating"].max(), 200)
        fig3    = go.Figure()
        fig3.add_trace(go.Scatter(x=sample2["school_rating"], y=sample2["price"], mode="markers",
                                  marker=dict(color=COLORS[3], opacity=0.4, size=4), name="Homes"))
        fig3.add_trace(go.Scatter(x=xs2, y=m2 * xs2 + b2, mode="lines",
                                  line=dict(color=COLORS[2], width=2), name="Trend"))
        fig3.update_layout(**PLOTLY_LAYOUT, title="School Rating vs Price",
                           xaxis_title="School Rating (1–10)", yaxis_title="Price ($)", showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-title">Correlation Matrix</div>', unsafe_allow_html=True)
    num_cols = ["price", "sqft", "bedrooms", "bathrooms", "house_age",
                "garage_spaces", "school_rating", "property_tax", "lot_size"]
    corr = df[num_cols].corr().round(2)
    fig_c = px.imshow(corr, text_auto=True, aspect="auto",
                      color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig_c.update_layout(**PLOTLY_LAYOUT, title="Feature Correlation Matrix", height=420,
                        coloraxis_colorbar=dict(tickfont=dict(color="#8b949e"),
                                                title_font=dict(color="#c9d1d9")))
    fig_c.update_traces(textfont_size=11)
    st.plotly_chart(fig_c, use_container_width=True)

    render_footer()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Model Training":
    st.markdown("""
    <div class="hero">
      <div class="hero-title">🤖 Model Training & Evaluation</div>
      <p class="hero-sub">5 regression models trained on Illinois housing data and compared head-to-head.</p>
      <span class="tag">5 Models</span>
      <span class="tag">Cross-Validation</span>
      <span class="tag">80/20 Split</span>
      <br>
      <div class="built-by">Built by <span>Akash Tatti</span></div>
    </div>
    """, unsafe_allow_html=True)

    br = model_results[best_name]
    st.success(f"🏆 **Best Model: {best_name}** — R² = {br['R2']:.4f}  |  MAE = ${br['MAE']:,.0f}  |  CV R² = {br['CV_R2']:.4f}")

    st.markdown('<div class="section-title">Model Comparison Table</div>', unsafe_allow_html=True)
    comp = pd.DataFrame({
        "Model":    list(model_results.keys()),
        "R² Score": [f"{v['R2']:.4f}" for v in model_results.values()],
        "MAE ($)":  [f"${v['MAE']:,.0f}" for v in model_results.values()],
        "RMSE ($)": [f"${v['RMSE']:,.0f}" for v in model_results.values()],
        "CV R²":    [f"{v['CV_R2']:.4f}" for v in model_results.values()],
    }).set_index("Model")
    st.dataframe(comp, use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    names = list(model_results.keys())

    with col1:
        r2s = [v["R2"] * 100 for v in model_results.values()]
        fig_r2 = go.Figure(go.Bar(
            x=[n.replace(" ", "<br>") for n in names], y=r2s,
            marker_color=[COLORS[1] if n == best_name else COLORS[0] for n in names],
            text=[f"{v:.1f}%" for v in r2s], textposition="outside",
            textfont=dict(color="white", size=12),
        ))
        fig_r2.update_layout(**PLOTLY_LAYOUT, title="R² Score (%)",
                             yaxis_range=[0, 108], showlegend=False)
        st.plotly_chart(fig_r2, use_container_width=True)

    with col2:
        maes = [v["MAE"] / 1000 for v in model_results.values()]
        fig_mae = go.Figure(go.Bar(
            x=[n.replace(" ", "<br>") for n in names], y=maes,
            marker_color=[COLORS[1] if n == best_name else COLORS[2] for n in names],
            text=[f"${v:.1f}K" for v in maes], textposition="outside",
            textfont=dict(color="white", size=12),
        ))
        fig_mae.update_layout(**PLOTLY_LAYOUT, title="Mean Absolute Error ($000s)",
                              showlegend=False, yaxis_title="MAE ($000s)")
        st.plotly_chart(fig_mae, use_container_width=True)

    st.markdown('<div class="section-title">Actual vs Predicted</div>', unsafe_allow_html=True)
    sel   = st.selectbox("Choose model:", names, index=names.index(best_name))
    act   = y_test.values / 1000
    pred  = model_results[sel]["preds"] / 1000
    pct10 = (np.abs((pred - act) / act) <= 0.10).mean() * 100

    fig_ap = go.Figure()
    fig_ap.add_trace(go.Scatter(x=act, y=pred, mode="markers",
                                marker=dict(color=COLORS[0], opacity=0.35, size=5),
                                name="Predictions"))
    fig_ap.add_trace(go.Scatter(x=[act.min(), act.max()], y=[act.min(), act.max()],
                                mode="lines", line=dict(color=COLORS[4], width=2, dash="dash"),
                                name="Perfect Prediction"))
    fig_ap.update_layout(**PLOTLY_LAYOUT,
                         title=f"{sel} — {pct10:.1f}% of predictions within ±10% of actual",
                         xaxis_title="Actual Price ($000s)", yaxis_title="Predicted Price ($000s)")
    st.plotly_chart(fig_ap, use_container_width=True)

    st.markdown('<div class="section-title">Feature Importance — Gradient Boosting</div>', unsafe_allow_html=True)
    imp_model, _, _ = trained_models["Gradient Boosting"]
    feat_names = FEATURES + ENC_COLS
    imps = pd.Series(imp_model.feature_importances_, index=feat_names).nlargest(12).sort_values()
    fig_imp = go.Figure(go.Bar(
        x=imps.values, y=imps.index, orientation="h",
        marker_color=[COLORS[1] if i == imps.idxmax() else COLORS[0] for i in imps.index],
    ))
    fig_imp.update_layout(**PLOTLY_LAYOUT, title="Top 12 Most Important Features",
                          xaxis_title="Importance Score", height=420, showlegend=False)
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown('<div class="section-title">Residual Analysis</div>', unsafe_allow_html=True)
    residuals = pred - act
    col3, col4 = st.columns(2)
    with col3:
        fig_res = go.Figure(go.Scatter(x=pred, y=residuals, mode="markers",
                                       marker=dict(color=COLORS[3], opacity=0.35, size=4)))
        fig_res.add_hline(y=0, line_dash="dash", line_color=COLORS[4], line_width=2)
        fig_res.update_layout(**PLOTLY_LAYOUT, title="Residuals vs Predicted",
                              xaxis_title="Predicted ($000s)", yaxis_title="Residual ($000s)")
        st.plotly_chart(fig_res, use_container_width=True)
    with col4:
        fig_res2 = go.Figure(go.Histogram(x=residuals, nbinsx=50,
                                           marker_color=COLORS[2], opacity=0.85))
        fig_res2.add_vline(x=0, line_dash="dash", line_color=COLORS[4], line_width=2)
        fig_res2.update_layout(**PLOTLY_LAYOUT, title="Residual Distribution",
                               xaxis_title="Residual ($000s)", yaxis_title="Count")
        st.plotly_chart(fig_res2, use_container_width=True)

    render_footer()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PREDICT PRICE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮  Predict Price":
    st.markdown("""
    <div class="hero">
      <div class="hero-title">🔮 Live Illinois Price Predictor</div>
      <p class="hero-sub">Describe a home anywhere in Illinois and get an instant AI-powered price estimate.</p>
      <span class="tag">Real-time</span>
      <span class="tag">5 Models</span>
      <span class="tag">20 IL Regions</span>
      <br>
      <div class="built-by">Built by <span>Akash Tatti</span></div>
    </div>
    """, unsafe_allow_html=True)

    col_in, col_out = st.columns([1, 1], gap="large")

    REGION_COORDS = {
        "Lincoln Park":      (41.921, -87.644),
        "River North":       (41.893, -87.634),
        "Wicker Park":       (41.908, -87.677),
        "Logan Square":      (41.921, -87.707),
        "Hyde Park":         (41.795, -87.596),
        "South Loop":        (41.862, -87.627),
        "Pilsen":            (41.855, -87.658),
        "Evanston":          (42.045, -87.688),
        "Oak Park":          (41.885, -87.794),
        "Naperville":        (41.785, -88.147),
        "Schaumburg":        (42.033, -88.083),
        "Arlington Heights": (42.088, -87.980),
        "Joliet":            (41.525, -88.082),
        "Rockford":          (42.271, -89.094),
        "Springfield":       (39.798, -89.654),
        "Peoria":            (40.693, -89.589),
        "Champaign":         (40.116, -88.243),
        "Bloomington":       (40.484, -88.994),
        "Aurora":            (41.760, -88.320),
        "Elgin":             (42.037, -88.281),
    }
    CHICAGO_HOODS = {
        "Lincoln Park", "River North", "Wicker Park",
        "Logan Square", "Hyde Park", "South Loop", "Pilsen"
    }
    REGION_LIST = sorted(df["region"].unique().tolist())

    with col_in:
        st.markdown("#### 🏡 Home Details")
        pred_model  = st.selectbox("Model", list(trained_models.keys()),
                                    index=list(trained_models.keys()).index(best_name))
        region_sel  = st.selectbox("Region / City", REGION_LIST,
                                    index=REGION_LIST.index("Naperville"))
        sqft_val    = st.slider("Square Footage", 500, 5000, 1800)
        beds_val    = st.slider("Bedrooms", 1, 5, 3)
        baths_val   = st.slider("Bathrooms", 1, 5, 2)
        year_val    = st.slider("Year Built", 1900, 2024, 1995)
        garage_val  = st.slider("Garage Spaces", 0, 2, 1)
        basement    = st.selectbox("Basement", ["Yes", "No"])
        school_val  = st.slider("School Rating (1–10)", 1.0, 10.0, 7.0, 0.1)
        tax_val     = st.slider("Property Tax Rate (%)", 0.8, 4.5, 2.2, 0.1,
                                 help="Illinois avg ~2.2% — one of the highest in the US")
        lot_val     = st.slider("Lot Size (sqft)", 1000, 20000, 6500)

    with col_out:
        st.markdown("#### 💰 Prediction")

        lat_v, lon_v = REGION_COORDS.get(region_sel, (41.8, -88.0))
        house_age_v  = 2024 - year_val
        is_chicago_v = int(region_sel in CHICAGO_HOODS)
        basement_v   = 1 if basement == "Yes" else 0

        base_row = [sqft_val, beds_val, baths_val, house_age_v, garage_val,
                    basement_v, school_val, tax_val, lot_val, lat_v, lon_v, is_chicago_v]
        enc_row  = [1 if col == f"region_{region_sel}" else 0 for col in ENC_COLS]
        full_row = np.array([base_row + enc_row])

        model_obj, scaler_obj, _ = trained_models[pred_model]
        inp      = scaler_obj.transform(full_row) if scaler_obj else full_row
        pred_val = model_obj.predict(inp)[0]
        mae_val  = model_results[pred_model]["MAE"]

        st.markdown(f"""
        <div class="pred-box">
          <div class="pred-label">Estimated Home Value — {region_sel}</div>
          <div class="pred-value">${pred_val:,.0f}</div>
          <div class="pred-range">
            Likely range: ${max(0, pred_val - mae_val):,.0f} – ${pred_val + mae_val:,.0f}
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        all_p = []
        for mn, (mo, ms, _) in trained_models.items():
            i2 = ms.transform(full_row) if ms else full_row
            all_p.append({"Model": mn, "Price": mo.predict(i2)[0]})
        all_pdf = pd.DataFrame(all_p)

        fig_cmp = go.Figure(go.Bar(
            x=all_pdf["Model"].str.replace(" ", "<br>"),
            y=all_pdf["Price"] / 1000,
            marker_color=[COLORS[1] if n == pred_model else COLORS[0]
                          for n in all_pdf["Model"]],
            text=[f"${v:.0f}K" for v in all_pdf["Price"] / 1000],
            textposition="outside", textfont=dict(color="white", size=12),
        ))
        fig_cmp.update_layout(**PLOTLY_LAYOUT, title="All Model Estimates ($000s)",
                              yaxis_title="Estimate ($000s)", showlegend=False, height=290)
        st.plotly_chart(fig_cmp, use_container_width=True)

        r2p = model_results[pred_model]["R2"] * 100
        st.markdown(f"""
        <div class="info-card">
          <h4>📊 Model Confidence</h4>
          <p><b>{pred_model}</b> — R² {r2p:.1f}%, avg prediction error ±${mae_val:,.0f}</p>
        </div>
        <div class="info-card">
          <h4>🌽 Illinois Context</h4>
          <p>Illinois has one of the highest property tax rates in the US (~2.2% avg).
          Chicago's North Side and suburbs like Naperville and Evanston command
          2–4× the price of downstate cities like Rockford or Peoria.</p>
        </div>
        """, unsafe_allow_html=True)

    render_footer()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — HOW IT WORKS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📖  How It Works":
    st.markdown("""
    <div class="hero">
      <div class="hero-title">📖 How It Works</div>
      <p class="hero-sub">The full ML pipeline — from Illinois housing data to live predictions.</p>
      <br>
      <div class="built-by">Built by <span>Akash Tatti</span></div>
    </div>
    """, unsafe_allow_html=True)

    steps = [
        ("1. Illinois-Specific Dataset",
         "Covers 20 real IL regions: Chicago neighborhoods (Lincoln Park, Wicker Park, Logan Square), inner suburbs (Evanston, Naperville, Oak Park), and downstate cities (Springfield, Rockford, Peoria, Champaign). Prices reflect real Illinois market patterns."),
        ("2. Illinois-Specific Features",
         "Includes property tax rate (IL averages ~2.2% — one of the US's highest), basement presence (very common in Midwest homes), school district rating, and a Chicago vs. downstate flag."),
        ("3. Exploratory Data Analysis",
         "Price distributions, regional bar charts, and an interactive Illinois map reveal that North Side Chicago and Naperville homes trade at 2–4× the price of Rockford or Peoria."),
        ("4. Feature Engineering",
         "year_built is converted to house_age. The 20 regions are one-hot encoded. An is_chicago flag captures the urban premium across many features."),
        ("5. Train-Test Split",
         "80% of homes train the models; 20% are held out for honest evaluation. The model never sees test homes during training — this prevents overfitting."),
        ("6. 5 Models Trained",
         "Linear Regression (baseline), Ridge (handles correlated features), Lasso (auto-selects features), Random Forest (120 trees), Gradient Boosting (sequential error correction — usually wins)."),
        ("7. Metrics Explained",
         "R² = % of price variance explained. MAE = average dollar error. RMSE = penalizes large errors. CV R² = stability across 5 data folds."),
        ("8. Live Prediction",
         "Your slider inputs go through the same pipeline (feature engineering → encoding → scaling) and into the trained model for instant price estimates with a confidence range."),
    ]

    for title, body in steps:
        st.markdown(f"""
        <div class="info-card">
          <h4>{title}</h4>
          <p>{body}</p>
        </div>
        """, unsafe_allow_html=True)

    # st.markdown("---")
    # st.markdown('<div class="section-title">Resume Bullet Points</div>', unsafe_allow_html=True)
    # st.markdown("""
    # <div class="info-card">
    #   <p>
    #   • Built an end-to-end ML web app in <b>Streamlit</b> predicting home prices across
    #     20 Illinois regions including Chicago neighborhoods, suburbs (Naperville, Evanston, Oak Park),
    #     and downstate cities (Springfield, Rockford, Peoria)<br><br>
    #   • Engineered Illinois-specific features: property tax rate, basement presence, school district
    #     rating, and urban/suburban classification to improve model accuracy<br><br>
    #   • Trained and compared <b>5 regression models</b> (Linear, Ridge, Lasso, Random Forest,
    #     Gradient Boosting) with 5-fold cross-validation and full diagnostic reporting<br><br>
    #   • Built interactive <b>Plotly</b> dashboards: geographic IL price map, regional comparisons,
    #     correlation matrix, feature importances, and residual analysis<br><br>
    #   • Deployed a live prediction tool allowing users to estimate home values for any Illinois
    #     region with real-time comparison across all 5 trained models
    #   </p>
    # </div>
    # """, unsafe_allow_html=True)

    # render_footer()