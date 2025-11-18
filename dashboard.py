##############################################
#  ELEPHANT CONFLICT EARLY WARNING SYSTEM UI
#        FULL STREAMLIT DASHBOARD
##############################################

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import st_folium
import shap
import joblib
import matplotlib.pyplot as plt
import rasterio
from scipy.ndimage import gaussian_filter
from datetime import date, datetime
from dotenv import load_dotenv
import os

from deep_translator import GoogleTranslator

def auto_translate(message, source_lang="auto"):
    """Translate message automatically into English, Sinhala, Tamil."""
    
    def translate(text, target):
        try:
            return GoogleTranslator(source=source_lang, target=target).translate(text)
        except:
            return f"[Translation to {target} failed]"
    
    return {
        "English": translate(message, "en"),
        "Sinhala": translate(message, "si"),
        "Tamil": translate(message, "ta")
    }


# ------------------ APP CONFIG ------------------

st.set_page_config(
    page_title="Elephant Conflict EWS",
    page_icon="🐘",
    layout="wide"
)

API_URL = "http://localhost:8000"
load_dotenv()
APP_API_KEY = os.getenv("APP_API_KEY")
API_HEADERS = {}

from app import locations
from app.database import SessionLocal


###################################################
#                API HELPERS
###################################################

def get_risk_heatmap():
    try:
        res = requests.get(f"{API_URL}/risk-heatmap/", headers=API_HEADERS)
        res.raise_for_status()
        return res.json()
    except:
        return []


def get_analytics():
    try:
        res = requests.get(f"{API_URL}/analytics/", headers=API_HEADERS)
        res.raise_for_status()
        return res.json()
    except:
        return None


###################################################
#                TERRITORY ANALYZER
###################################################

from sqlalchemy.orm import Session

class HerdTerritoryAnalyzer:
    def __init__(self, db_session):
        self.db = db_session

    def cluster_conflict_zones(self):
        from app.models import ConflictIncident
        incidents = self.db.query(ConflictIncident).all()
        if not incidents:
            return pd.DataFrame(), pd.DataFrame()

        data = [{
            "latitude": i.latitude,
            "longitude": i.longitude,
            "timestamp": i.timestamp,
            "elephant_count": i.elephant_count or 1
        } for i in incidents]

        df = pd.DataFrame(data)

        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler

        coords = df[["latitude", "longitude"]].values
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coords)

        if len(coords) > 1:
            model = DBSCAN(eps=0.5, min_samples=3)
            df["territory_id"] = model.fit_predict(coords_scaled)
        else:
            df["territory_id"] = 0

        stats = df[df["territory_id"] >= 0].groupby("territory_id").agg(
            center_lat=("latitude", "mean"),
            lat_spread=("latitude", "std"),
            center_lon=("longitude", "mean"),
            lon_spread=("longitude", "std"),
            total_elephants=("elephant_count", "sum"),
            conflict_count=("elephant_count", "count")
        ).reset_index()

        return df, stats

    def analyze_temporal_patterns(self, df):
        if df.empty:
            return pd.DataFrame(), []

        df["month"] = df["timestamp"].dt.month
        df["season"] = df["month"].apply(
            lambda m: "Winter" if m in [12,1,2] else
                      "Spring" if m in [3,4,5] else
                      "Summer" if m in [6,7,8,9] else
                      "Autumn"
        )

        seasonal = df.groupby(["territory_id","season"]).size().reset_index(name="conflicts")

        peak = []
        for tid in seasonal["territory_id"].unique():
            temp = seasonal[seasonal["territory_id"] == tid]
            max_row = temp.loc[temp["conflicts"].idxmax()]
            peak.append({
                "territory_id": tid,
                "peak_season": max_row["season"],
                "peak_conflicts": max_row["conflicts"]
            })

        return seasonal, peak

    def predict_territory_expansion(self, df):
        if df.empty:
            return pd.DataFrame()

        expansions = []
        for tid in df["territory_id"].unique():
            tdf = df[df["territory_id"] == tid]
            if len(tdf) > 5:
                early = tdf.nsmallest(5, "timestamp")
                late = tdf.nlargest(5, "timestamp")
                dlat = late["latitude"].mean() - early["latitude"].mean()
                dlon = late["longitude"].mean() - early["longitude"].mean()
                dist = np.sqrt(dlat**2 + dlon**2) * 111
                direction = (
                    "North" if dlat > 0 else "South"
                    if abs(dlat) > abs(dlon)
                    else "East" if dlon > 0 else "West"
                )
                expansions.append({
                    "territory_id": tid,
                    "distance_shifted_km": round(dist,2),
                    "direction": direction,
                    "years_tracked": 2
                })
        return pd.DataFrame(expansions)


###################################################
#                TERRITORY UI PAGE
###################################################

def show_territory_analysis():
    st.header("🐘 Elephant Territory Analysis")

    db = SessionLocal()

    try:
        analyzer = HerdTerritoryAnalyzer(db)
        df, stats = analyzer.cluster_conflict_zones()

        if df.empty:
            st.warning("No conflict data.")
            return

        st.subheader("🗺 Territory Map")
        m = folium.Map(location=[7.8, 80.7], zoom_start=8)

        colors = ["red","blue","green","purple","orange","yellow"]
        for _, row in stats.iterrows():
            cid = int(row["territory_id"])
            color = colors[cid % len(colors)]

            folium.Circle(
                location=[row["center_lat"], row["center_lon"]],
                radius=row["lat_spread"]*111000,
                color=color,
                fill=True,
                popup=f"T{cid} — {int(row['conflict_count'])} conflicts"
            ).add_to(m)

        st_folium(m, width=1000, height=500)

        st.subheader("📊 Territory Statistics")
        st.dataframe(stats, use_container_width=True)

        st.subheader("🌦 Seasonal Activity Patterns")
        seasonal, peaks = analyzer.analyze_temporal_patterns(df)
        if not seasonal.empty:
            pivot = seasonal.pivot(index="territory_id", columns="season", values="conflicts").fillna(0)
            fig = px.imshow(pivot, color_continuous_scale="Reds",
                            title="Seasonal Activity Heatmap")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Peak Seasons")
            st.dataframe(pd.DataFrame(peaks), use_container_width=True)

        st.subheader("⚠️ Territory Expansion Alerts")
        exp = analyzer.predict_territory_expansion(df)
        if exp.empty:
            st.info("No movement detected.")
        else:
            st.dataframe(exp, use_container_width=True)

    finally:
        db.close()


###################################################
#                TERRAIN MODELING UI PAGE
###################################################

import plotly.graph_objects as go

def show_terrain_modeling():
    st.header("🗻 Terrain Modeling & Elephant Corridors")

    elevation_path = os.path.join("app", "data", "sri_lanka_elevation.tif")

    # ---- LOAD DEM SAFELY ----
    try:
        src = rasterio.open(elevation_path)
        elevation = src.read(1)
    except Exception:
        st.error(f"Missing elevation file. Expected at: {elevation_path}")
        return

    # ---- DOWNSAMPLE TO AVOID >200MB STREAM ----
    # Target resolution: ~800 x 800
    max_dim = 800
    scale = max(1, elevation.shape[0] // max_dim)

    elevation_small = elevation[::scale, ::scale]

    # ---- SLOPE ----
    gy, gx = np.gradient(elevation_small)
    slope = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2)))

    # ---- SMOOTH ----
    smooth = gaussian_filter(slope, 3)
    valleys = smooth < 12   # corridor threshold

    # ------------------------------
    # 🌍 Elevation Model
    # ------------------------------
    st.subheader("🌍 Elevation Model (Downsampled)")
    st.plotly_chart(
        px.imshow(elevation_small, color_continuous_scale="earth"),
        use_container_width=True
    )

    # ------------------------------
    # 🌀 CONTOUR LINES (using go.Contour)
    # ------------------------------
    st.subheader("🌀 Elevation Contours")
    fig_contour = go.Figure(
        data=go.Contour(
            z=elevation_small,
            colorscale="earth",
            contours=dict(
                showlines=True,
                coloring="heatmap"
            )
        )
    )
    st.plotly_chart(fig_contour, use_container_width=True)

    # ------------------------------
    # ⛰ RAW SLOPE
    # ------------------------------
    st.subheader("⛰ Slope Map")
    st.plotly_chart(
        px.imshow(slope, color_continuous_scale="viridis"),
        use_container_width=True
    )

    # ------------------------------
    # 🔄 SMOOTHED SLOPE
    # ------------------------------
    st.subheader("🔄 Smoothed Slope (Noise Removed)")
    st.plotly_chart(
        px.imshow(smooth, color_continuous_scale="viridis"),
        use_container_width=True
    )

    # ------------------------------
    # 🐘 ELEPHANT CORRIDORS
    # ------------------------------
    st.subheader("🐘 Elephant Corridors (Valleys)")
    st.plotly_chart(
        px.imshow(valleys, color_continuous_scale=["black", "yellow"]),
        use_container_width=True
    )

    # ------------------------------
    # 🌋 3D ELEVATION MODEL
    # ------------------------------
    st.subheader("🌋 3D Elevation Surface")
    fig3d = go.Figure(
        data=[go.Surface(z=elevation_small, colorscale="earth")]
    )
    fig3d.update_layout(
        height=600,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Elevation"
        )
    )
    st.plotly_chart(fig3d, use_container_width=True)



###################################################
#                SHAP-FREE EXPLAINABILITY
###################################################

def show_explainability_dashboard():
    st.header("🔍 Data-Driven Explainability (No Model Needed)")

    st.info("""
    This module extracts explainability insights directly from
    your conflict dataset — **without requiring an ML model**.

    Included:
    • Correlation heatmap  
    • Sensitivity-based feature importance  
    • Partial-dependency style plots  
    • Mutual Information driver analysis  
    """)

    # ===================================================
    # Load conflict data
    # ===================================================
    db = SessionLocal()
    from app.models import ConflictIncident
    incidents = db.query(ConflictIncident).all()
    db.close()

    if not incidents:
        st.warning("No conflict data available.")
        return

    # Build dataframe
    df = pd.DataFrame([{
        "latitude": i.latitude,
        "longitude": i.longitude,
        "elephant_count": i.elephant_count,
        "incident_type": i.incident_type,
        "district": i.district,
        "timestamp": i.timestamp
    } for i in incidents])

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month

    # ===================================================
    # 1️⃣ Correlation Heatmap
    # ===================================================
    st.subheader("📌 Correlation Map")

    numeric_df = df[["latitude", "longitude", "elephant_count", "hour", "month"]]
    corr = numeric_df.corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu",
        title="Correlation Between Key Numerical Features"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ===================================================
    # 2️⃣ Sensitivity-Based (Variance) Feature Importance
    # ===================================================
    st.subheader("🔥 Feature Importance (Variance-Based)")

    importance = numeric_df.var().sort_values(ascending=False)
    imp_df = pd.DataFrame({
        "Feature": importance.index,
        "Importance": importance.values
    })

    fig = px.bar(
        imp_df,
        x="Feature",
        y="Importance",
        color="Importance",
        color_continuous_scale="Blues",
        title="Which Features Vary the Most? (Sensitivity)"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ===================================================
    # 3️⃣ Partial Dependency Style Plots
    # ===================================================
    st.subheader("📈 Behavioral Relationships (Lowess Smoothed)")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(
            df,
            x="elephant_count",
            y="month",
            trendline="lowess",
            title="Elephant Count vs Month (Seasonality)"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(
            df,
            x="hour",
            y="elephant_count",
            trendline="lowess",
            title="Time of Day vs Elephant Group Size"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ===================================================
    # 4️⃣ Mutual Information (Feature → Incident Type)
    # ===================================================
    st.subheader("🧠 Conflict Driver Analysis (Mutual Information)")

    from sklearn.feature_selection import mutual_info_classif

    mi_df = df.copy()
    mi_df["incident_label"] = mi_df["incident_type"].astype("category").cat.codes

    features = ["elephant_count", "latitude", "longitude", "hour", "month"]
    X = mi_df[features]
    y = mi_df["incident_label"]

    mi_scores = mutual_info_classif(X, y, discrete_features=False)

    mi_plot = pd.DataFrame({
        "Feature": features,
        "Importance": mi_scores
    }).sort_values("Importance", ascending=False)

    fig = px.bar(
        mi_plot,
        x="Feature",
        y="Importance",
        color="Importance",
        color_continuous_scale="Plasma",
        title="Which Features Influence the Incident Type Most?"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Mutual Information estimates nonlinear dependency between features and incident types.")





###################################################
#                ECONOMIC IMPACT UI
###################################################

def get_5day_forecast():
    return [
        {'location':'Hambantota','latitude':6.124,'longitude':81.119,'risk_level':'High','elephant_count_est':4},
        {'location':'Monaragala','latitude':6.872,'longitude':81.350,'risk_level':'Medium','elephant_count_est':2},
        {'location':'Ampara','latitude':7.297,'longitude':81.675,'risk_level':'Critical','elephant_count_est':6},
        {'location':'Polonnaruwa','latitude':7.940,'longitude':81.000,'risk_level':'Low','elephant_count_est':1},
    ]


def show_economic_impact_dashboard(return_data=False):
    st.header("💰 Economic Impact & Resource Allocation")

    forecast = get_5day_forecast()

    # Basic severity scoring
    preds = []
    for item in forecast:
        lvl = item["risk_level"]
        base = {"Low":50000,"Medium":150000,"High":300000,"Critical":500000}.get(lvl,50000)
        dmg = base * (1 + (item["elephant_count_est"]-1)*0.3)
        preds.append({
            **item,
            "severity_level": "Critical" if lvl=="Critical" else "High" if lvl=="High" else "Medium",
            "estimated_damage_rs": int(dmg),
            "confidence": 0.7 if lvl in ["High","Critical"] else 0.5
        })

    df = pd.DataFrame(preds)
    total_risk = sum(df["estimated_damage_rs"] * df["confidence"])
    high_risk = len(df[df["severity_level"].isin(["High","Critical"])])

    c1,c2,c3 = st.columns(3)
    c1.metric("Total Expected Loss", f"Rs. {total_risk:,.0f}")
    c2.metric("High Risk Locations", high_risk)
    c3.metric("Avg Elephants", df["elephant_count_est"].mean().round(1))

    st.subheader("🗺 Resource Allocation Map")
    m = folium.Map(location=[7.8,80.7],zoom_start=8)

    for _, r in df.iterrows():
        color = "red" if r["severity_level"]=="Critical" else \
                "orange" if r["severity_level"]=="High" else "yellow"
        folium.Marker(
            [r["latitude"],r["longitude"]],
            popup=f"{r['location']}<br>Damage: Rs {r['estimated_damage_rs']:,}",
            icon=folium.Icon(color=color)
        ).add_to(m)

    st_folium(m, width=1000, height=500)

    st.subheader("💥 Economic Risk vs Severity Matrix")
    df["sev_num"] = df["severity_level"].map({"Low":1,"Medium":2,"High":3,"Critical":4})
    fig = px.scatter(
        df, x="sev_num", y="estimated_damage_rs",
        color="severity_level", size="estimated_damage_rs",
        hover_name="location",
        title="Economic Impact vs Severity Level"
    )
    st.plotly_chart(fig, use_container_width=True)

    # 👇 Only return data IF the tab requests it
    if return_data:
        return df


###################################################
#                MANUAL PREDICT UI
###################################################

def show_manual_predict():
    st.subheader("🔮 Manual Elephant Conflict Prediction")

    with st.form("manual_predict_form"):
        location = st.text_input("Location Name")
        latitude = st.number_input("Latitude", format="%.6f")
        longitude = st.number_input("Longitude", format="%.6f")
        elephant_count = st.number_input("Elephant Count", min_value=0, max_value=50, step=1)
        rainfall = st.number_input("Rainfall (mm)", min_value=0, step=1)
        crop_type = st.selectbox("Crop Type", ["Paddy", "Maize", "Sugarcane", "Banana", "Other"])

        submitted = st.form_submit_button("Predict Risk")

    if submitted:
        payload = {
            "location": location,
            "latitude": latitude,
            "longitude": longitude,
            "elephant_count": elephant_count,
            "rainfall": rainfall,
            "crop_type": crop_type
        }

        try:
            res = requests.post(f"{API_URL}/predict", json=payload)
            if res.status_code == 200:
                pred = res.json()
                st.success(f"Predicted Risk Level: **{pred['risk_level']}**")
                st.metric("📈 Risk Score", round(pred["risk_score"], 2))

            else:
                st.error(f"Prediction failed: {res.text}")

        except Exception as e:
            st.error(f"Error connecting to prediction API: {e}")

###################################################
#                REPORT SIGHTING UI
###################################################

def show_report_sighting():
    st.subheader("📝 Report Real-Time Elephant Sighting")

    with st.form("report_sighting_form"):
        location = st.text_input("Location / Village Name")
        latitude = st.number_input("Latitude", format="%.6f")
        longitude = st.number_input("Longitude", format="%.6f")
        elephants = st.number_input("Number of Elephants", min_value=1, max_value=50, step=1)

        behaviour = st.selectbox(
            "Elephant Behaviour",
            ["Calm", "Aggressive", "Crop-Raid", "Near Settlements"]
        )

        notes = st.text_area("Additional Notes (optional)")

        submitted = st.form_submit_button("Submit Sighting")

    if submitted:
        payload = {
            "location": location,
            "latitude": latitude,
            "longitude": longitude,
            "elephant_count": elephants,
            "behaviour": behaviour,
            "notes": notes
        }

        try:
            res = requests.post(f"{API_URL}/report-sighting", json=payload)

            if res.status_code == 200:
                st.success("✅ Sighting reported successfully!")
                st.info("Thank you! This helps improve conflict monitoring accuracy.")

            else:
                st.error(f"Error: {res.text}")

        except Exception as e:
            st.error(f"Failed to send report: {e}")


###################################################
#                MAIN APP UI
###################################################

st.title("🐘 Elephant Conflict Early Warning System")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📍 Live Map",
    "🔮 5-Day Forecast",
    "📊 Analytics",
    "🐘 Territory Analysis",
    "🗻 Terrain Modeling",
    "🔍 Explainability",
    "💰 Economic Impact",
    "⚙️ Manual Predict",
    "📝 Report Sighting"
])

###################################################
#         🔥 NEW: Causal Inference Module
###################################################

def estimate_causal_effect(df):
    import numpy as np
    import plotly.express as px
    from sklearn.linear_model import LogisticRegression

    st.subheader("🧠 Causal Inference on Economic Damage")

    df = df.copy()
    df["treatment"] = (df["elephant_count_est"] >= 3).astype(int)

    # Outcome
    Y = df["estimated_damage_rs"]
    T = df["treatment"]

    # Propensity score model
    ps_model = LogisticRegression()
    ps_model.fit(df[["elephant_count_est"]], T)
    df["ps"] = ps_model.predict_proba(df[["elephant_count_est"]])[:, 1]

    # Inverse Propensity Weighting
    df["weight"] = T / df["ps"] + (1 - T) / (1 - df["ps"])

    ate = (df["weight"] * Y).sum() / df["weight"].sum()

    st.metric("📌 Estimated Causal Effect (ATE)",
              f"+ Rs {ate:,.0f} additional damage")

    st.info("""
    **Interpretation:**  
    High elephant presence *causes* an increase in expected economic loss.  
    We adjust for imbalance using IPW (Inverse Propensity Weighting).
    """)

    st.subheader("📈 Simplified Causal Graph")
    st.write("""
    Elephant Count ─────▶ Damage  
           ▲  
           │  
        Rainfall  
    """)

    # Causal scatter plot
    fig = px.scatter(
        df,
        x="elephant_count_est",
        y="estimated_damage_rs",
        color="treatment",
        size="estimated_damage_rs",
        hover_name="location",
        title="Causal Relationship: Elephant Count → Damage"
    )

    st.plotly_chart(fig, use_container_width=True)


###################################################
#                TAB CONTENTS
###################################################

# -------- TAB 1 ----------
with tab1:
    st.header("📍 Live Conflict Risk Map")

    # Pull latest conflict data directly from database
    db = SessionLocal()
    from app.models import ConflictIncident
    incidents = db.query(ConflictIncident).all()
    db.close()

    if not incidents:
        st.warning("No conflict data available.")
    else:
        df = pd.DataFrame([{
            "timestamp": i.timestamp,
            "location": i.location,
            "latitude": i.latitude,
            "longitude": i.longitude,
            "district": i.district,
            "incident_type": i.incident_type,
            "elephant_count": i.elephant_count
        } for i in incidents])

        # Derive risk score (simple heat formula)
        df["risk_score"] = (
            df["elephant_count"] * 1.5 +
            df["incident_type"].map({
                "crop_raid": 3,
                "property_damage": 4,
                "human_injury": 6,
                "elephant_death": 8
            }).fillna(2)
        )

        # Convert score → level
        def risk_level(x):
            if x >= 8:
                return "Critical"
            elif x >= 5:
                return "High"
            elif x >= 3:
                return "Medium"
            else:
                return "Low"

        df["risk_level"] = df["risk_score"].apply(risk_level)

        # Create map
        m = folium.Map(location=[7.8, 80.7], zoom_start=8)

        for _, row in df.iterrows():
            color = (
                "red" if row["risk_level"] == "Critical" else
                "orange" if row["risk_level"] == "High" else
                "yellow" if row["risk_level"] == "Medium" else
                "green"
            )

            popup = f"""
            <b>Location:</b> {row['location']}<br>
            <b>Type:</b> {row['incident_type']}<br>
            <b>Elephants:</b> {row['elephant_count']}<br>
            <b>Risk:</b> {row['risk_level']}
            """

            folium.CircleMarker(
                [row["latitude"], row["longitude"]],
                radius=10,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=popup
            ).add_to(m)

        st_folium(m, width="100%", height=550)



# -------- TAB 2 ----------
with tab2:
    st.header("🔮 5-Day Forecast & Alerts")
    st.info("Demo mode with automatic translation between English, Sinhala and Tamil.")

    # Demo forecast data
    forecast = [
        {"day": "Day 1", "location": "Hambantota", "rainfall_mm": 12, "risk": "Medium", "elephant_count": 8},
        {"day": "Day 2", "location": "Hambantota", "rainfall_mm": 23, "risk": "High", "elephant_count": 15},
        {"day": "Day 3", "location": "Hambantota", "rainfall_mm": 5,  "risk": "Low", "elephant_count": 3},
        {"day": "Day 4", "location": "Hambantota", "rainfall_mm": 30, "risk": "High", "elephant_count": 12},
        {"day": "Day 5", "location": "Hambantota", "rainfall_mm": 18, "risk": "Medium", "elephant_count": 7},
    ]

    df = pd.DataFrame(forecast)
    st.dataframe(df, use_container_width=True)

    st.subheader("📩 Multi-Language Alerts (Auto Generated)")

    for _, row in df.iterrows():

        # Base message (one template only — auto-translates)
        english_message = f"""
Day: {row['day']}
Location: {row['location']}
Rainfall: {row['rainfall_mm']} mm
Elephants Detected: {row['elephant_count']}
Risk Level: {row['risk']}

Please take necessary precautions.
"""

        translations = auto_translate(english_message)

        with st.expander(f"{row['day']} — {row['risk']} Risk — {row['location']}", expanded=row["risk"] != "Low"):

            st.write("### 🌾 Farmer Message (Simplified Tone)")

            farmer_template = f"""
FARMER ALERT:
{english_message}

Action:
• Protect crops
• Avoid night travel
• Report sightings immediately
"""

            farmer_translated = auto_translate(farmer_template)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.text_area("English (Farmer)", farmer_translated["English"], height=150)

            with col2:
                st.text_area("Sinhala (Farmer)", farmer_translated["Sinhala"], height=150)

            with col3:
                st.text_area("Tamil (Farmer)", farmer_translated["Tamil"], height=150)

            # OFFICIAL MESSAGE
            st.write("### 🛡 wildlife Official Alert (Formal Tone)")

            official_template = f"""
WILDLIFE DEPARTMENT ALERT:
A {row['risk']} risk of elephant conflict has been detected.

Location: {row['location']}
Rainfall: {row['rainfall_mm']} mm
Elephant Group Size: {row['elephant_count']}
Day: {row['day']}

Recommended Actions:
• Deploy patrol units
• Monitor migratory corridors
• Update command center
"""

            official_translated = auto_translate(official_template)

            col4, col5, col6 = st.columns(3)

            with col4:
                st.text_area("English (Official)", official_translated["English"], height=150)

            with col5:
                st.text_area("Sinhala (Official)", official_translated["Sinhala"], height=150)

            with col6:
                st.text_area("Tamil (Official)", official_translated["Tamil"], height=150)

    st.caption("💡 Auto-translation is generated locally with deep-translator — no external paid API, no SMS errors.")





# -------- TAB 3 ----------
with tab3:
    st.header("📊 Analytics")

    db = SessionLocal()
    from app.models import ConflictIncident

    incidents = db.query(ConflictIncident).all()
    db.close()

    if not incidents:
        st.warning("No conflict data available.")
    else:
        # Build dataframe
        df = pd.DataFrame([{
            "timestamp": i.timestamp,
            "location": i.location,
            "latitude": i.latitude,
            "longitude": i.longitude,
            "district": i.district,
            "incident_type": i.incident_type,
            "elephant_count": i.elephant_count
        } for i in incidents])

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["month"] = df["timestamp"].dt.month
        df["year"] = df["timestamp"].dt.year
        df["hour"] = df["timestamp"].dt.hour

        # ======================
        # 1️⃣ INCIDENT TYPE SHARE
        # ======================
        st.subheader("Incident Type Distribution")
        fig = px.pie(
            df,
            names="incident_type",
            title="Share of Incident Types",
            hole=0.3
        )
        st.plotly_chart(fig, use_container_width=True)

        # ======================
        # 2️⃣ MONTHLY TREND
        # ======================
        st.subheader("Monthly Conflict Trend")

        trend = df.groupby(["year", "month"]).size().reset_index(name="count")
        trend["date"] = pd.to_datetime(trend[["year", "month"]].assign(day=1))

        fig = px.line(
            trend,
            x="date",
            y="count",
            markers=True,
            title="Conflict Frequency Over Time"
        )
        st.plotly_chart(fig, use_container_width=True)

        # ======================
        # 3️⃣ DISTRICT HEATMAP
        # ======================
        st.subheader("District Hotspot Heatmap")

        district_counts = df.groupby("district").size().reset_index(name="count")

        fig = px.bar(
            district_counts.sort_values("count", ascending=False),
            x="district",
            y="count",
            title="Top Conflict Districts",
            color="count",
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig, use_container_width=True)

        # ======================
        # 4️⃣ PEAK HOURS
        # ======================
        st.subheader("Peak Conflict Hours")

        hourly = df.groupby("hour").size().reset_index(name="count")

        fig = px.bar(
            hourly,
            x="hour",
            y="count",
            title="Conflicts by Hour of Day"
        )
        st.plotly_chart(fig, use_container_width=True)

        # ======================
        # 5️⃣ ELEPHANT GROUP SIZE
        # ======================
        st.subheader("Elephant Group Size Distribution")

        fig = px.box(
            df,
            y="elephant_count",
            title="Elephant Group Size During Conflicts"
        )
        st.plotly_chart(fig, use_container_width=True)

        # ======================
        # 6️⃣ INCIDENT–DISTRICT CORRELATION
        # ======================
        st.subheader("Incident Type vs District (Correlation)")

        pivot = pd.pivot_table(
            df,
            index="district",
            columns="incident_type",
            values="elephant_count",
            aggfunc="count",
            fill_value=0
        )

        fig = px.imshow(
            pivot,
            text_auto=True,
            color_continuous_scale="RdBu",
            title="District vs Incident Type Heatmap"
        )
        st.plotly_chart(fig, use_container_width=True)

        # ======================
        # 7️⃣ CONFLICT DRIVER ANALYSIS (Mutual Information)
        # ======================
        from sklearn.feature_selection import mutual_info_classif

        st.subheader("🧠 Conflict Driver Analysis (Mutual Information)")

        mi_df = df.copy()
        mi_df["incident_label"] = mi_df["incident_type"].astype("category").cat.codes

        features = ["elephant_count", "latitude", "longitude", "hour", "month"]
        X = mi_df[features]
        y = mi_df["incident_label"]

        mi_scores = mutual_info_classif(X, y, discrete_features=False)

        mi_plot = pd.DataFrame({
            "Feature": features,
            "Importance": mi_scores
        }).sort_values("Importance", ascending=False)

        fig = px.bar(
            mi_plot,
            x="Feature",
            y="Importance",
            color="Importance",
            color_continuous_scale="Plasma",
            title="Which Factors Drive Conflict Type?"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Mutual Information reveals nonlinear influence of each feature on conflict type.")




# -------- TAB 4 ----------
with tab4:
    show_territory_analysis()


# -------- TAB 5 ----------
with tab5:
    show_terrain_modeling()


# -------- TAB 6 ----------
with tab6:
    show_explainability_dashboard()

# -------- TAB 7 ----------
with tab7:
    st.header("💰 Economic Impact & Causal Insights")

    # 1️⃣ Economic model
    df = show_economic_impact_dashboard(return_data=True)

    # 2️⃣ Causal inference
    if df is not None:
        estimate_causal_effect(df)


# -------- TAB 8 ----------
with tab8:
    st.header("Manual Predict")
    show_manual_predict()

# -------- TAB 9 ----------
with tab9:
    st.header("Report Sighting")
    show_report_sighting()



