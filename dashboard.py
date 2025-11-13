# dashboard.py
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
from datetime import date
from app import locations
import os
from dotenv import load_dotenv

# --- Config ---
st.set_page_config(
    page_title="Elephant Conflict EWS Dashboard",
    page_icon="🐘",
    layout="wide"
)

# --- API Base URL ---
API_URL = "http://localhost:8000"

# --- Day 5: API Key for Security ---
load_dotenv()
APP_API_KEY = os.getenv("APP_API_KEY")

# Check if API key is loaded
if not APP_API_KEY:
    st.error("FATAL ERROR: APP_API_KEY not found in .env file. Dashboard cannot start.")
    st.stop()
    
API_HEADERS = {"X-API-Key": APP_API_KEY}

# --- Helper Functions (Now with Headers) ---
def get_analytics():
    try:
        response = requests.get(f"{API_URL}/analytics/", headers=API_HEADERS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching analytics: {e}")
        return None

def get_risk_heatmap():
    try:
        response = requests.get(f"{API_URL}/risk-heatmap/", headers=API_HEADERS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching risk heatmap: {e}")
        return []

# --- Main App ---
st.title("🐘 Elephant Conflict Early Warning System")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📍 Live Risk Map",
    "🔮 5-Day Forecast",
    "📊 Analytics Dashboard",
    "⚙️ Manual Predict",
    "📝 Report Sighting"
])

# --- TAB 1: Live Risk Map ---
with tab1:
    st.header("Live Conflict Risk Map")
    st.markdown("This map shows real-time predicted conflict risk levels. Data is updated based on the latest environmental data.")
    
    if st.button("Refresh Map"):
        st.cache_data.clear()
    
    heatmap_data = get_risk_heatmap()
    
    if not heatmap_data:
        st.warning("Could not fetch heatmap data. Is the API server running and is the API key valid?")
    else:
        m = folium.Map(location=[7.8731, 80.7718], zoom_start=8)
        
        def get_color(risk_level):
            if risk_level == "High": return "red"
            elif risk_level == "Medium": return "orange"
            else: return "green"
        
        for item in heatmap_data:
            loc = item.get("location", "Unknown")
            risk = item.get("risk_level", "Low")
            lat = item.get("latitude")
            lon = item.get("longitude")
            
            if lat is None or lon is None:
                coords = locations.get_coords(loc)
                if coords:
                    lat, lon = coords
                else:
                    st.warning(f"Missing coordinates for {loc}. Skipping.")
                    continue

            folium.CircleMarker(
                location=[lat, lon],
                radius=15,
                popup=f"<strong>{loc}</strong><br>Risk: {risk}",
                color=get_color(risk),
                fill=True,
                fill_color=get_color(risk),
                fill_opacity=0.7
            ).add_to(m)
        
        st_folium(m, width="100%", height=500)
        
        with st.expander("Show Raw Heatmap Data"):
            st.json(heatmap_data)

# --- TAB 2: 5-Day Forecast ---
with tab2:
    st.header("🔮 5-Day Risk Forecast")
    st.markdown("Select a location to see the predicted conflict risk for the next 5 days, based on the weather forecast.")
    
    location_names = locations.get_location_names()
    selected_location = st.selectbox("Select Location", location_names)
    
    if st.button(f"Get 5-Day Forecast for {selected_location}"):
        with st.spinner("Fetching forecast and predicting risk..."):
            try:
                # --- ADDED HEADERS ---
                response = requests.get(f"{API_URL}/predict-forecast/{selected_location}", headers=API_HEADERS)
                response.raise_for_status()
                forecast_data = response.json()
                
                if not forecast_data:
                    st.warning("No forecast data returned from API.")
                else:
                    st.subheader(f"Risk Forecast: {selected_location}")
                    df = pd.DataFrame(forecast_data)
                    df['date'] = pd.to_datetime(df['date'])
                    risk_map = {"Low": 1, "Medium": 2, "High": 3}
                    df['risk_score'] = df['risk_level'].map(risk_map)
                    
                    fig = px.bar(
                        df, 
                        x='date', 
                        y='risk_score',
                        color='risk_level',
                        text='risk_level',
                        title=f"5-Day Risk Forecast for {selected_location}",
                        color_discrete_map={"Low": "#2ca02c", "Medium": "#ff7f0e", "High": "#d62728"},
                        category_orders={"risk_level": ["Low", "Medium", "High"]}
                    )
                    fig.update_yaxes(title="Risk Level", tickvals=[1, 2, 3], ticktext=["Low", "Medium", "High"])
                    fig.update_xaxes(title="Date", dtick="D1", tickformat="%a, %b %d")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("Show Raw Forecast Data"):
                        st.dataframe(df)
            except requests.exceptions.RequestException as e:
                st.error(f"Error fetching forecast: {e}")
                if e.response:
                    st.error(f"Details: {e.response.json()}")

# --- TAB 3: Analytics Dashboard ---
with tab3:
    st.header("📊 Historical Conflict Analytics")
    analytics_data = get_analytics()
    if not analytics_data:
        st.error("Could not load analytics data. API server may be down or API key is invalid.")
    elif analytics_data.get("error"):
        st.warning(analytics_data.get("error"))
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Incidents by Type")
            df_type = pd.DataFrame(analytics_data.get("by_type", []))
            if not df_type.empty:
                fig_type = px.pie(df_type, names='type', values='count', hole=0.3)
                st.plotly_chart(fig_type, use_container_width=True)
            else:
                st.warning("No 'by_type' data.")
        with col2:
            st.subheader("Incidents by Location")
            df_loc = pd.DataFrame(analytics_data.get("by_location", []))
            if not df_loc.empty:
                fig_loc = px.bar(df_loc, x='location', y='count', color='location')
                st.plotly_chart(fig_loc, use_container_width=True)
            else:
                st.warning("No 'by_location' data.")
        st.subheader("Incidents Over Time")
        df_time = pd.DataFrame(analytics_data.get("over_time", []))
        if not df_time.empty:
            fig_time = px.line(df_time, x='month_year', y='count', markers=True)
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.warning("No 'over_time' data.")

# --- TAB 4: Manual Predict ---
with tab4:
    st.header("⚙️ Manual Predict")
    st.markdown("Manually trigger a prediction for a specific set of conditions.")
    
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            location = st.selectbox("Select Location", locations.get_location_names())
            prediction_date = st.date_input("Date", date.today())
        with col2:
            rainfall_mm = st.number_input("Rainfall (mm)", min_value=0.0, value=10.0)
            vegetation_index = st.number_input("Vegetation Index (NDVI)", min_value=0.0, max_value=1.0, value=0.5)
        
        submitted = st.form_submit_button("Predict Risk")
        
    if submitted:
        payload = {
            "date": prediction_date.isoformat(),
            "location": location,
            "rainfall_mm": rainfall_mm,
            "vegetation_index": vegetation_index
        }
        
        try:
            # --- ADDED HEADERS ---
            response = requests.post(f"{API_URL}/predict-risk/", json=payload, headers=API_HEADERS)
            response.raise_for_status()
            result = response.json()
            
            st.subheader("Prediction Result")
            risk_level = result.get("risk_level")
            
            if risk_level == "High":
                st.error(f"**Risk Level: {risk_level}**")
                st.warning(f"Alert Status: {result.get('alert_status')}")
            elif risk_level == "Medium":
                st.warning(f"**Risk Level: {risk_level}**")
            else:
                st.success(f"**Risk Level: {risk_level}**")
            
            with st.expander("Show Full API Response"):
                st.json(result)
                
        except requests.exceptions.RequestException as e:
            st.error(f"Error calling prediction API: {e}")
            st.json(e.response.json() if e.response else "No response from server.")

# --- TAB 5: Report Sighting (Day 3) ---
with tab5:
    st.header("📝 Report an Elephant Sighting")
    st.markdown("Use this form to submit a report if you have seen an elephant. This helps improve our data.")
    
    with st.form("report_form"):
        location = st.selectbox("Select Location", locations.get_location_names(), key="report_loc")
        elephant_count = st.number_input("Number of Elephants", min_value=1, value=1, step=1)
        description = st.text_area("Description (optional)", placeholder="e.g., 'A small herd near the main road.'")
        
        report_submitted = st.form_submit_button("Submit Report")
        
    if report_submitted:
        payload = {
            "location": location,
            "elephant_count": elephant_count,
            "description": description
        }
        
        try:
            # --- ADDED HEADERS ---
            response = requests.post(f"{API_URL}/report-sighting/", json=payload, headers=API_HEADERS)
            response.raise_for_status()
            st.success("Report submitted. Thank you!")
        except requests.exceptions.RequestException as e:
            st.error(f"Error submitting report: {e}")


