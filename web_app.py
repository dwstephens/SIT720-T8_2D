import streamlit as st
import pandas as pd
import math
import json
import joblib
import folium
from datetime import date, datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from streamlit_folium import st_folium


# -----------------------------
# Custom Transformer
# -----------------------------
class DateFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, start_date="2025-01-01"):
        if isinstance(start_date, str):
            self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        else:
            self.start_date = start_date

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        date_series = pd.to_datetime(X['date'])
        month_number = date_series.dt.month
        features = pd.DataFrame(index=X.index)
        features['month_sin'] = month_number.apply(lambda m: math.sin(2 * math.pi * m / 12))
        features['month_cos'] = month_number.apply(lambda m: math.cos(2 * math.pi * m / 12))
        features['elapsed_days'] = (date_series - self.start_date).dt.days
        features['day'] = date_series.dt.day_name()
        return features


# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("property_pipeline.joblib")

# Load the pipeline from the joblib file
loaded_pipeline = load_model()


# -----------------------------
# Column Setup
# -----------------------------
desired_order = [
    'bedrooms', 'bathrooms', 'car_spaces', 'land_size', 'primary_sch_count',
    'secondary_sch_count', 'childcare_count', 'bus_stop_distance',
    'train_stop_distance', 'park_distance', 'shopping_centre_distance',
    'month_sin', 'month_cos', 'elapsed_days', 'suburb', 'property_type',
    'day'
]

transformed_columns = [
    'month_sin', 'month_cos', 'elapsed_days', 'day',
    'bedrooms', 'bathrooms', 'car_spaces', 'land_size', 'primary_sch_count',
    'secondary_sch_count', 'childcare_count', 'bus_stop_distance',
    'train_stop_distance', 'park_distance', 'shopping_centre_distance',
    'suburb', 'property_type'
]

# -----------------------------
# Pipeline Setup
# -----------------------------
feature_engineer = ColumnTransformer(
    transformers=[
        ('date_features', DateFeatureEngineer(start_date="2025-01-01"), ['date'])
    ],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('preprocessing', feature_engineer)
])


# -----------------------------
# Sidebar Inputs
# -----------------------------

# Sidebar input
st.sidebar.header("Enter Property Details")

# Top section (single column)
top = st.sidebar.container()
selected_date = top.date_input("Date",                      
    min_value=date(2025, 1, 1),
    max_value=date(2025, 12, 31)
)

# Main property info
top.header("Main Property Info")
suburb = top.selectbox("Suburb", ["Berwick", "Officer", "Pakenham"])
property_type = top.selectbox("Property Type", ["House", "Townhouse"])
bedrooms = top.slider("Bedrooms", 2, 6, 3)
bathrooms = top.slider("Bathrooms", 1, 5, 2)
car_spaces = top.slider("Car Spaces", 1, 6, 1)
land_size = top.slider("Land Size (sqm)", 90, 2200, 100)

# Middle section (two columns)
middle = st.sidebar.container()
middle.header("Nearby Education")

m_col1, m_col2, m_col3 = middle.columns(3)

with m_col1:
    primary_sch_count = st.slider("Primary Schools", 0, 5, 0)

with m_col2:
    secondary_sch_count = st.slider("Secondary Schools", 0, 2, 1)

with m_col3:
    childcare_count = st.slider("Childcare", 2, 24, 5)

# Bottom section (two columns)
bottom = st.sidebar.container()
bottom.header("Nearby Amenities")

b_col1, b_col2 = bottom.columns(2)

with b_col1:
    bus_stop_distance = st.slider("Bus Stop (km)", 0.0, 1.3, 0.5)
    park_distance = st.slider("Park (km)", 0.0, 1.3, 0.5)

with b_col2:
    train_stop_distance = st.slider("Train Station (km)", 0.0, 3.2, 1.0)
    shopping_centre_distance = st.slider("Shopping (km)", 0.0, 2.5, 1.0)


# -----------------------------
# Main Layout
# -----------------------------

st.markdown("""
    <div style='text-align: center;'>
        <h1>Berwick, Officer, Pakenham</h1>
        <h2>Property Price Prediction</h2>
        <hr style='margin-top: 0;'>
    </div>
""", unsafe_allow_html=True)

left_col, right_col = st.columns([1, 2])

with left_col:
    st.header("Prediction")
    if st.button("Predict"):
        # 
        input_data = {
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "car_spaces": car_spaces,
            "land_size": land_size,
            "primary_sch_count": primary_sch_count,
            "secondary_sch_count": secondary_sch_count,
            "childcare_count": childcare_count,
            "bus_stop_distance": bus_stop_distance,
            "train_stop_distance": train_stop_distance,
            "park_distance": park_distance,
            "shopping_centre_distance": shopping_centre_distance,
            "date": selected_date.strftime("%Y-%m-%d"),
            "suburb": suburb,
            "property_type": property_type
        }

        input_df = pd.DataFrame([input_data])
        transformed = pipeline.fit_transform(input_df)
        transformed_df = pd.DataFrame(transformed, columns=transformed_columns)
        final_df = transformed_df[desired_order]

        prediction = loaded_pipeline.predict(final_df)[0]
        #st.write(f"**Predicted Price:** ${prediction:,.0f}")
        st.markdown(f"<h2 style='color:limegreen; font-weight:bold'>Predicted Price: ${prediction:,.0f}</h2>", unsafe_allow_html=True)

    else:
        st.write("")

with right_col:
    #st.header("Map")

    # Load GeoJSON boundaries
    with open("data/berwick_officer_pakenham_boundaries.geojson", "r", encoding="utf-8") as f:
        suburb_geojson = json.load(f)

    # Map setup
    map_centre = [-38.05916,  145.40947]
    m = folium.Map(location=map_centre, control_scale=True)

    # Add the GeoJSON data to the map
    geojson_layer = folium.GeoJson(
        suburb_geojson,
        name="Suburb Boundaries",
        style_function=lambda feature: {
            "fillColor": "blue",
            "color": "blue",
            "weight": 2,
            "fillOpacity": 0.2,
        },
        tooltip=folium.GeoJsonTooltip(fields=["name"], aliases=["Suburb"])
    ).add_to(m)

    # Fit the map to the bounds of the GeoJSON layer
    m.fit_bounds(geojson_layer.get_bounds())

    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=True,
        control=True,
        show=False 
    ).add_to(m)

    # Add layer control to toggle layers
    folium.LayerControl().add_to(m)

    st_folium(m, height=600)

# Footer section
with st.container():
    st.markdown("---")
    st.markdown("**Disclaimer:** This is a demo app. Predictions are for illustrative purposes only.")
    st.markdown("**Created by Darrin William Stephens** for Deakin SIT720 Machine Learning Task 8.2D")
    st.markdown("Copyright 2025")
