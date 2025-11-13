# dashboard.py
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
from datetime import date, datetime, timedelta
from app import locations
import os
from dotenv import load_dotenv
import numpy as np
from sqlalchemy import create_engine, func
from sqlalchemy.orm import Session

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

# --- NEW: Territory Analysis Class ---
class HerdTerritoryAnalyzer:
    def __init__(self, db_session):
        self.db = db_session
        
    def cluster_conflict_zones(self):
        """Cluster conflict incidents to identify elephant territories"""
        # Get all conflict incidents
        from app.models import ConflictIncident
        incidents = self.db.query(ConflictIncident).all()
        
        if not incidents:
            return pd.DataFrame(), pd.DataFrame()
            
        # Create DataFrame
        data = []
        for incident in incidents:
            data.append({
                'latitude': incident.latitude,
                'longitude': incident.longitude,
                'timestamp': incident.timestamp,
                'district': incident.district,
                'elephant_count': incident.elephant_count or 1
            })
        
        df = pd.DataFrame(data)
        
        # Simple clustering based on geographic proximity
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
        
        coords = df[['latitude', 'longitude']].values
        if len(coords) > 1:
            # Normalize coordinates
            scaler = StandardScaler()
            coords_scaled = scaler.fit_transform(coords)
            
            # DBSCAN clustering
            clustering = DBSCAN(eps=0.5, min_samples=3).fit(coords_scaled)
            df['territory_id'] = clustering.labels_
        else:
            df['territory_id'] = 0
            
        # Calculate territory statistics
        territory_stats = df[df['territory_id'] >= 0].groupby('territory_id').agg({
            'latitude': ['mean', 'std'],
            'longitude': ['mean', 'std'],
            'elephant_count': 'sum'
        }).round(4)
        
        territory_stats.columns = ['center_lat', 'lat_spread', 'center_lon', 'lon_spread', 'total_elephants']
        territory_stats['conflict_count'] = df[df['territory_id'] >= 0].groupby('territory_id').size()
        territory_stats = territory_stats.reset_index()
        
        return df, territory_stats
    
    def analyze_temporal_patterns(self, territories_df):
        """Analyze seasonal patterns in each territory"""
        if territories_df.empty:
            return pd.DataFrame(), []
            
        # Add season column
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8, 9]:
                return 'Summer'
            else:
                return 'Autumn'
        
        territories_df['month'] = territories_df['timestamp'].dt.month
        territories_df['season'] = territories_df['month'].apply(get_season)
        
        # Seasonal conflicts by territory
        seasonal_data = territories_df.groupby(['territory_id', 'season']).size().reset_index(name='conflicts')
        
        # Identify peak seasons for each territory
        peak_seasons = []
        for territory_id in seasonal_data['territory_id'].unique():
            territory_data = seasonal_data[seasonal_data['territory_id'] == territory_id]
            peak_season = territory_data.loc[territory_data['conflicts'].idxmax()]
            peak_seasons.append({
                'territory_id': territory_id,
                'peak_season': peak_season['season'],
                'peak_conflicts': peak_season['conflicts']
            })
            
        return seasonal_data, peak_seasons
    
    def predict_territory_expansion(self, territories_df):
        """Predict territory expansion based on historical movement"""
        if territories_df.empty:
            return pd.DataFrame()
            
        # Simplified expansion prediction
        expansion_data = []
        for territory_id in territories_df['territory_id'].unique():
            territory_data = territories_df[territories_df['territory_id'] == territory_id]
            
            if len(territory_data) > 5:  # Need sufficient data
                # Calculate centroid movement over time
                early_data = territory_data.nsmallest(5, 'timestamp')
                late_data = territory_data.nlargest(5, 'timestamp')
                
                early_center_lat = early_data['latitude'].mean()
                early_center_lon = early_data['longitude'].mean()
                late_center_lat = late_data['latitude'].mean()
                late_center_lon = late_data['longitude'].mean()
                
                # Calculate distance and direction
                lat_diff = late_center_lat - early_center_lat
                lon_diff = late_center_lon - early_center_lon
                distance_km = np.sqrt(lat_diff**2 + lon_diff**2) * 111  # Approx km per degree
                
                direction = "North" if lat_diff > 0 else "South"
                if abs(lon_diff) > abs(lat_diff):
                    direction = "East" if lon_diff > 0 else "West"
                
                expansion_data.append({
                    'territory_id': territory_id,
                    'distance_shifted_km': round(distance_km, 2),
                    'direction': direction,
                    'years_tracked': 2  # Simplified
                })
        
        return pd.DataFrame(expansion_data)

# --- NEW: Economic Impact Class ---
class ConflictSeverityPredictor:
    def __init__(self):
        self.model = None
        
    def load_model(self, model_path):
        """Load trained severity model"""
        try:
            import joblib
            self.model = joblib.load(model_path)
        except:
            st.warning("Could not load severity model. Using rule-based predictions.")
            self.model = "rule_based"
    
    def predict_severity(self, forecast_data):
        """Predict conflict severity and economic impact"""
        predictions = []
        
        for location_data in forecast_data:
            # Rule-based severity prediction
            risk_level = location_data.get('risk_level', 'Low')
            elephant_count = location_data.get('elephant_count_est', 3)
            
            # Economic impact estimation (Sri Lanka Rupees)
            base_damage = {
                'Low': 50000,
                'Medium': 150000,
                'High': 300000,
                'Critical': 500000
            }.get(risk_level, 50000)
            
            # Adjust for elephant count
            damage_multiplier = 1 + (elephant_count - 1) * 0.3
            estimated_damage = base_damage * damage_multiplier
            
            # Confidence based on historical data
            confidence = 0.7 if risk_level in ['High', 'Critical'] else 0.5
            
            predictions.append({
                'location': location_data.get('location', 'Unknown'),
                'latitude': location_data.get('latitude'),
                'longitude': location_data.get('longitude'),
                'risk_level': risk_level,
                'severity_level': 'Critical' if risk_level == 'Critical' else 'High' if risk_level == 'High' else 'Medium',
                'estimated_damage_rs': int(estimated_damage),
                'elephant_count_est': elephant_count,
                'confidence': confidence
            })
        
        return predictions
    
    def prioritize_response_resources(self, predictions):
        """Prioritize locations for resource allocation"""
        df = pd.DataFrame(predictions)
        
        if df.empty:
            return df
            
        # Priority scoring
        def calculate_priority(row):
            severity_score = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}[row['severity_level']]
            damage_score = min(row['estimated_damage_rs'] / 100000, 10)  # Normalize
            confidence_score = row['confidence'] * 2
            
            total_score = severity_score + damage_score + confidence_score
            
            if total_score >= 8:
                return f"IMMEDIATE - Score: {total_score:.1f}"
            elif total_score >= 6:
                return f"URGENT - Score: {total_score:.1f}"
            elif total_score >= 4:
                return f"MODERATE - Score: {total_score:.1f}"
            else:
                return f"LOW - Score: {total_score:.1f}"
        
        df['response_recommendation'] = df.apply(calculate_priority, axis=1)
        df = df.sort_values('estimated_damage_rs', ascending=False)
        
        return df

def calculate_patrol_units(predictions):
    """Calculate required patrol units based on predictions"""
    high_risk_count = len([p for p in predictions if p['severity_level'] in ['High', 'Critical']])
    return max(1, high_risk_count // 2)  # 1 patrol unit per 2 high-risk locations

def get_5day_forecast():
    """Mock function to get 5-day forecast data"""
    # This would integrate with your actual forecast API
    return [
        {'location': 'Hambantota', 'latitude': 6.124, 'longitude': 81.119, 'risk_level': 'High', 'elephant_count_est': 4},
        {'location': 'Monaragala', 'latitude': 6.872, 'longitude': 81.350, 'risk_level': 'Medium', 'elephant_count_est': 2},
        {'location': 'Ampara', 'latitude': 7.297, 'longitude': 81.675, 'risk_level': 'Critical', 'elephant_count_est': 6},
        {'location': 'Polonnaruwa', 'latitude': 7.940, 'longitude': 81.000, 'risk_level': 'Low', 'elephant_count_est': 1},
    ]

# --- NEW: Territory Analysis Tab ---
def show_territory_analysis():
    st.header("🐘 Elephant Territory Analysis")
    
    # Initialize database session
    from app.database import SessionLocal
    db = SessionLocal()
    
    try:
        analyzer = HerdTerritoryAnalyzer(db)
        territories, stats = analyzer.cluster_conflict_zones()
        
        if territories.empty:
            st.warning("No conflict data available for territory analysis.")
            return
            
        # 1. Territory Map
        st.subheader("Elephant Territory Map")
        
        m = folium.Map(location=[7.8731, 80.7718], zoom_start=8)
        
        # Color palette for territories
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightblue', 'darkgreen']
        
        for idx, row in stats.iterrows():
            territory_id = int(row['territory_id'])
            color = colors[territory_id % len(colors)]
            
            folium.Circle(
                location=[row['center_lat'], row['center_lon']],
                radius=row['lat_spread'] * 111000,  # Convert to meters
                color=color,
                fill=True,
                popup=f"Territory {territory_id}<br>{int(row['conflict_count'])} conflicts<br>{int(row['total_elephants'])} elephants estimated"
            ).add_to(m)
            
            # Add territory label
            folium.Marker(
                [row['center_lat'], row['center_lon']],
                popup=f"Territory {territory_id} Center",
                icon=folium.DivIcon(html=f'<div style="font-weight: bold; color: {color}">T{territory_id}</div>')
            ).add_to(m)
        
        st_folium(m, width=1000, height=500)
        
        # 2. Territory Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Territories Identified", len(stats))
        with col2:
            st.metric("Total Conflicts Analyzed", len(territories))
        with col3:
            st.metric("Average Conflicts per Territory", f"{stats['conflict_count'].mean():.1f}")
        
        # 3. Seasonal Activity Heatmap
        st.subheader("Seasonal Activity Patterns")
        
        temporal, peaks = analyzer.analyze_temporal_patterns(territories)
        
        if not temporal.empty:
            pivot = temporal.pivot(index='territory_id', columns='season', values='conflicts').fillna(0)
            
            fig = px.imshow(pivot, 
                            labels=dict(x="Season", y="Territory ID", color="Conflicts"),
                            x=pivot.columns,
                            y=pivot.index.astype(str),
                            color_continuous_scale='Reds',
                            title="Conflict Frequency by Territory and Season")
            st.plotly_chart(fig, use_container_width=True)
            
            # Peak seasons table
            st.subheader("Peak Conflict Seasons by Territory")
            peaks_df = pd.DataFrame(peaks)
            st.dataframe(peaks_df, use_container_width=True)
        
        # 4. Expansion Alerts
        st.subheader("⚠️ Territory Expansion Alerts")
        
        expansion = analyzer.predict_territory_expansion(territories)
        
        if not expansion.empty:
            for idx, row in expansion.iterrows():
                if row['distance_shifted_km'] > 5:  # Alert if shifted >5km
                    st.warning(f"🚨 **Territory {int(row['territory_id'])}** has shifted **{row['distance_shifted_km']} km {row['direction']}** over {int(row['years_tracked'])} years")
                else:
                    st.info(f"📍 Territory {int(row['territory_id'])}: Stable position (shifted {row['distance_shifted_km']} km)")
        else:
            st.info("No significant territory expansion detected.")
            
    except Exception as e:
        st.error(f"Error in territory analysis: {str(e)}")
    finally:
        db.close()

# --- NEW: Economic Impact Tab ---
def show_economic_impact_dashboard():
    st.header("💰 Economic Impact & Resource Allocation")
    
    predictor = ConflictSeverityPredictor()
    
    # Try to load model, fallback to rule-based
    try:
        predictor.load_model('models/severity_model.pkl')
    except:
        st.info("Using rule-based economic impact predictions")
    
    # Get predictions for next 5 days
    forecast_data = get_5day_forecast()
    predictions = predictor.predict_severity(forecast_data)
    
    if not predictions:
        st.warning("No prediction data available for economic analysis.")
        return
    
    # 1. Total Economic Risk
    total_risk = sum([p['estimated_damage_rs'] * p['confidence'] for p in predictions])
    high_risk_locations = len([p for p in predictions if p['severity_level'] in ['High', 'Critical']])
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Economic Risk (5 days)", f"Rs. {total_risk:,.0f}")
    col2.metric("High-Risk Locations", high_risk_locations)
    col3.metric("Patrol Units Needed", calculate_patrol_units(predictions))
    
    # 2. Resource Allocation Map
    st.subheader("Resource Allocation Map")
    
    allocation_df = predictor.prioritize_response_resources(predictions)
    
    m = folium.Map(location=[7.8731, 80.7718], zoom_start=8)
    
    for idx, row in allocation_df.iterrows():
        priority_level = row['response_recommendation'].split('-')[0].strip()
        color = {
            'IMMEDIATE': 'red',
            'URGENT': 'orange',
            'MODERATE': 'yellow',
            'LOW': 'green'
        }.get(priority_level, 'blue')
        
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"""
            <strong>{row['location']}</strong><br>
            Damage: Rs. {row['estimated_damage_rs']:,}<br>
            Severity: {row['severity_level']}<br>
            <b>{row['response_recommendation']}</b>
            """,
            icon=folium.Icon(color=color, icon='exclamation-triangle', prefix='fa')
        ).add_to(m)
    
    st_folium(m, width=1000, height=500)
    
    # 3. Priority Allocation Table
    st.subheader("Priority Response Allocation")
    st.dataframe(allocation_df[['location', 'estimated_damage_rs', 'severity_level', 'response_recommendation']], 
                 use_container_width=True)
    
    # 4. ROI Calculator
    st.subheader("📊 ROI Calculator")
    
    total_potential_loss = allocation_df['estimated_damage_rs'].sum()
    prevention_cost = st.slider("Prevention Budget (Rs.)", 100000, 5000000, 1000000, 100000)
    
    # Effectiveness based on budget allocation
    effectiveness = min(prevention_cost / total_potential_loss * 2, 0.8)  # Cap at 80% effectiveness
    
    prevented_loss = total_potential_loss * effectiveness
    net_benefit = prevented_loss - prevention_cost
    roi = (net_benefit / prevention_cost) * 100 if prevention_cost > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Potential Loss Prevented", f"Rs. {prevented_loss:,.0f}")
    col2.metric("Net Benefit", f"Rs. {net_benefit:,.0f}")
    col3.metric("Return on Investment", f"{roi:.1f}%")
    
    # Effectiveness gauge
    st.subheader("Prevention Effectiveness")
    fig = px.pie(values=[effectiveness, 1-effectiveness], 
                 names=['Damage Prevented', 'Remaining Risk'],
                 title=f"Budget Effectiveness: {effectiveness:.1%}")
    st.plotly_chart(fig, use_container_width=True)

# --- Updated Main App with New Tabs ---
st.title("🐘 Elephant Conflict Early Warning System")

# Updated tabs with new features
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📍 Live Risk Map",
    "🔮 5-Day Forecast", 
    "📊 Analytics Dashboard",
    "🐘 Territory Analysis",  # NEW
    "💰 Economic Impact",     # NEW
    "⚙️ Manual Predict",
    "📝 Report Sighting"
])

# --- TAB 1: Live Risk Map (UNCHANGED) ---
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

# --- TAB 2: 5-Day Forecast (UNCHANGED) ---
with tab2:
    st.header("🔮 5-Day Risk Forecast")
    st.markdown("Select a location to see the predicted conflict risk for the next 5 days, based on the weather forecast.")
    
    location_names = locations.get_location_names()
    selected_location = st.selectbox("Select Location", location_names, key="forecast_loc")
    
    if st.button(f"Get 5-Day Forecast for {selected_location}"):
        with st.spinner("Fetching forecast and predicting risk..."):
            try:
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

# --- TAB 3: Analytics Dashboard (UNCHANGED) ---
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

# --- NEW TAB 4: Territory Analysis ---
with tab4:
    show_territory_analysis()

# --- NEW TAB 5: Economic Impact ---  
with tab5:
    show_economic_impact_dashboard()

# --- TAB 6: Manual Predict (UNCHANGED) ---
with tab6:
    st.header("⚙️ Manual Predict")
    st.markdown("Manually trigger a prediction for a specific set of conditions.")
    
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            location = st.selectbox("Select Location", locations.get_location_names(), key="predict_loc")
            prediction_date = st.date_input("Date", date.today(), key="predict_date")
        with col2:
            rainfall_mm = st.number_input("Rainfall (mm)", min_value=0.0, value=10.0, key="rainfall")
            vegetation_index = st.number_input("Vegetation Index (NDVI)", min_value=0.0, max_value=1.0, value=0.5, key="vegetation")
        
        submitted = st.form_submit_button("Predict Risk")
        
    if submitted:
        payload = {
            "date": prediction_date.isoformat(),
            "location": location,
            "rainfall_mm": rainfall_mm,
            "vegetation_index": vegetation_index
        }
        
        try:
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

# --- TAB 7: Report Sighting (UNCHANGED) ---
with tab7:
    st.header("📝 Report an Elephant Sighting")
    st.markdown("Use this form to submit a report if you have seen an elephant. This helps improve our data.")
    
    with st.form("report_form"):
        location = st.selectbox("Select Location", locations.get_location_names(), key="report_loc")
        elephant_count = st.number_input("Number of Elephants", min_value=1, value=1, step=1, key="elephant_count")
        description = st.text_area("Description (optional)", placeholder="e.g., 'A small herd near the main road.'", key="description")
        
        report_submitted = st.form_submit_button("Submit Report")
        
    if report_submitted:
        payload = {
            "location": location,
            "elephant_count": elephant_count,
            "description": description
        }
        
        try:
            response = requests.post(f"{API_URL}/report-sighting/", json=payload, headers=API_HEADERS)
            response.raise_for_status()
            st.success("Report submitted. Thank you!")
        except requests.exceptions.RequestException as e:
            st.error(f"Error submitting report: {e}")