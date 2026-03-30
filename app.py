from flask import Flask, request, jsonify
import requests
import joblib
import pandas as pd
from datetime import datetime
import os
app = Flask(__name__)

# =========================
# LOAD ML FILES
# =========================
model = joblib.load("model_multihazard.pkl")
feature_columns = joblib.load("feature_columns_multihazard.pkl")
target_columns = joblib.load("target_columns_multihazard.pkl")

# =========================
# APIS
# =========================
GEOCODING_API = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_API = "https://api.open-meteo.com/v1/forecast"


# =========================
# 1) GEOCODING HELPERS
# =========================
def try_geocode(place_query):
    params = {
        "name": place_query,
        "count": 1,
        "language": "en",
        "format": "json",
        "countryCode": "IN"
    }

    response = requests.get(GEOCODING_API, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    results = data.get("results")
    if not results:
        return None

    place = results[0]
    return {
        "name": place.get("name"),
        "latitude": place.get("latitude"),
        "longitude": place.get("longitude"),
        "country": place.get("country"),
        "admin1": place.get("admin1"),
        "admin2": place.get("admin2")
    }


def get_coordinates(state=None, district=None, area=None):
    queries = []

    if area and district and state:
        queries.append(f"{area}, {district}, {state}")
    if district and state:
        queries.append(f"{district}, {state}")
    if area and state:
        queries.append(f"{area}, {state}")
    if district:
        queries.append(district)
    if state:
        queries.append(state)

    for query in queries:
        result = try_geocode(query)
        if result:
            result["matched_query"] = query
            return result

    return None


# =========================
# 2) WEATHER FROM OPEN-METEO
# =========================
def get_weather(lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,precipitation,rain,surface_pressure,wind_speed_10m",
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,rain,surface_pressure,wind_speed_10m",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,wind_speed_10m_max",
        "forecast_days": 7,
        "timezone": "auto"
    }

    response = requests.get(FORECAST_API, params=params, timeout=15)
    response.raise_for_status()
    return response.json()


# =========================
# 3) EXTRACT WEATHER FOR SELECTED DATE
# =========================
def get_selected_day_weather(weather_data, selected_date):
    current = weather_data.get("current", {})
    daily = weather_data.get("daily", {})

    daily_dates = daily.get("time", [])
    rain_sum = daily.get("rain_sum", [])
    precipitation_sum = daily.get("precipitation_sum", [])
    temp_max = daily.get("temperature_2m_max", [])
    temp_min = daily.get("temperature_2m_min", [])
    wind_max = daily.get("wind_speed_10m_max", [])

    selected_day = {
        "Rainfall (mm)": current.get("rain", 0) or current.get("precipitation", 0) or 0,
        "Temperature (°C)": current.get("temperature_2m", 0),
        "Humidity (%)": current.get("relative_humidity_2m", 0),
        "Forecast Temp Max": 0,
        "Forecast Temp Min": 0,
        "Forecast Wind Max": 0
    }

    if selected_date in daily_dates:
        idx = daily_dates.index(selected_date)

        selected_day["Rainfall (mm)"] = (
            rain_sum[idx] if idx < len(rain_sum) else
            precipitation_sum[idx] if idx < len(precipitation_sum) else
            0
        )

        selected_day["Forecast Temp Max"] = temp_max[idx] if idx < len(temp_max) else 0
        selected_day["Forecast Temp Min"] = temp_min[idx] if idx < len(temp_min) else 0
        selected_day["Forecast Wind Max"] = wind_max[idx] if idx < len(wind_max) else 0

        if idx < len(temp_max) and idx < len(temp_min):
            selected_day["Temperature (°C)"] = (temp_max[idx] + temp_min[idx]) / 2

    return selected_day


# =========================
# 4) LAND COVER LOGIC
# =========================
def get_land_cover_simple(location_name):
    name = location_name.lower()

    major_urban = ["chennai", "mumbai", "delhi", "bangalore", "kolkata", "hyderabad", "pune"]
    coastal_water = ["chennai", "mumbai", "visakhapatnam", "kochi", "nagapattinam", "cuddalore", "puri"]

    land_cover = {
        "Land Cover_Agricultural": 0,
        "Land Cover_Desert": 0,
        "Land Cover_Forest": 0,
        "Land Cover_Urban": 0,
        "Land Cover_Water Body": 0
    }

    if any(city in name for city in major_urban):
        land_cover["Land Cover_Urban"] = 1
    else:
        land_cover["Land Cover_Agricultural"] = 1

    if any(place in name for place in coastal_water):
        land_cover["Land Cover_Water Body"] = 1

    return land_cover


# =========================
# 5) POPULATION ESTIMATION
# =========================
def get_population_estimate(location_name):
    name = location_name.lower()

    high_density = ["chennai", "mumbai", "delhi", "bangalore", "kolkata", "hyderabad"]
    medium_density = ["coimbatore", "madurai", "trichy", "salem", "tiruppur"]

    if any(x in name for x in high_density):
        return 12000
    elif any(x in name for x in medium_density):
        return 6000
    else:
        return 1500


# =========================
# 6) SEASON FLAGS FROM DATE
# =========================
def get_season_flags_from_date(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    month = dt.month

    return {
        "year": dt.year,
        "is_monsoon": 1 if month in [6, 7, 8, 9] else 0,
        "is_summer": 1 if month in [3, 4, 5] else 0,
        "is_cyclone_season": 1 if month in [4, 5, 10, 11, 12] else 0,
    }


# =========================
# 7) DEFAULT FEATURE VALUES
# =========================
def get_default_feature_values(location, weather, selected_date):
    elevation_val = weather.get("elevation", 0)
    location_name = location.get("matched_query", location.get("name", ""))
    day_weather = get_selected_day_weather(weather, selected_date)

    defaults = {
        "Latitude": location.get("latitude", 0),
        "Longitude": location.get("longitude", 0),
        "Rainfall (mm)": day_weather.get("Rainfall (mm)", 0),
        "Temperature (°C)": day_weather.get("Temperature (°C)", 0),
        "Humidity (%)": day_weather.get("Humidity (%)", 0),
        "River Discharge (m³/s)": 300,
        "Water Level (m)": 5,
        "Elevation (m)": elevation_val,
        "Population Density": get_population_estimate(location_name),
        "Infrastructure": 1,
        "Historical Floods": 1,

        "Land Cover_Agricultural": 0,
        "Land Cover_Desert": 0,
        "Land Cover_Forest": 0,
        "Land Cover_Urban": 0,
        "Land Cover_Water Body": 0,

        "Soil Type_Clay": 1,
        "Soil Type_Loam": 0,
        "Soil Type_Peat": 0,
        "Soil Type_Sandy": 0,
        "Soil Type_Silt": 0,
    }

    defaults.update(get_land_cover_simple(location_name))
    defaults.update(get_season_flags_from_date(selected_date))

    return defaults


# =========================
# 8) PREPARE MODEL INPUT
# =========================
def prepare_model_input(location, weather, selected_date):
    feature_data = get_default_feature_values(location, weather, selected_date)

    df = pd.DataFrame([feature_data])

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]
    return df, feature_data


# =========================
# 9) RISK LEVEL HELPER
# =========================
def get_risk_level(prob_percent):
    if prob_percent >= 70:
        return "High"
    elif prob_percent >= 50:
        return "Moderate"
    else:
        return "Low"


# =========================
# 10) EXTRA FEATURE: AUTO 7-DAY ALERT SYSTEM
# =========================
def predict_next_7_days(location, weather_data):
    daily = weather_data.get("daily", {})
    dates = daily.get("time", [])

    alerts = []

    hazard_names = {
        "flood_label": "Flood",
        "cyclone_label": "Cyclone",
        "landslide_label": "Landslide",
        "heatwave_label": "Heatwave"
    }

    today = datetime.now().date()

    for date in dates:
        model_input, _ = prepare_model_input(location, weather_data, date)
        predictions = model.predict(model_input)[0]

        event_date = datetime.strptime(date, "%Y-%m-%d").date()
        days_remaining = (event_date - today).days

        day_predictions = {}
        day_confidence = {}
        day_risk_levels = {}

        for i, target in enumerate(target_columns):
            hazard_name = hazard_names.get(target, target)
            pred_value = int(predictions[i])

            probs = model.estimators_[i].predict_proba(model_input)[0]
            positive_prob = round(float(probs[1]) * 100, 2)

            day_predictions[hazard_name] = "Risk" if pred_value == 1 else "No Risk"
            day_confidence[hazard_name] = positive_prob
            day_risk_levels[hazard_name] = get_risk_level(positive_prob)

        max_risk_disaster = max(day_confidence, key=day_confidence.get)
        max_risk_confidence = day_confidence[max_risk_disaster]
        overall_risk_level = get_risk_level(max_risk_confidence)

        if max_risk_confidence >= 70 and days_remaining >= 0:
            if days_remaining == 0:
                alert_message = f"{max_risk_disaster} risk expected today"
            elif days_remaining == 1:
                alert_message = f"{max_risk_disaster} risk expected tomorrow"
            else:
                alert_message = f"{max_risk_disaster} risk expected in {days_remaining} days"

            alerts.append({
                "date": date,
                "days_remaining": days_remaining,
                "predictions": day_predictions,
                "confidence": day_confidence,
                "hazard_risk_levels": day_risk_levels,
                "max_risk_disaster": max_risk_disaster,
                "max_risk_confidence": max_risk_confidence,
                "overall_risk_level": overall_risk_level,
                "alert_message": alert_message
            })

    return alerts


# =========================
# 11) HOME
# =========================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Multi-Hazard Prediction API is running",
        "sample_route": "/predict?state=Tamil Nadu&district=Chennai&area=Anna Nagar&date=2026-04-05",
        "hazards": ["Flood", "Cyclone", "Landslide", "Heatwave"],
        "extra_feature": "Auto 7-Day Alert System enabled"
    })


# =========================
# 12) WEATHER ENDPOINT
# =========================
@app.route("/weather", methods=["GET"])
def weather():
    state = request.args.get("state")
    district = request.args.get("district")
    area = request.args.get("area")
    date = request.args.get("date")

    if not date:
        date = datetime.now().strftime("%Y-%m-%d")

    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return jsonify({"error": "Date must be in YYYY-MM-DD format"}), 400

    if not any([state, district, area]):
        return jsonify({"error": "Please provide at least state, district, or area"}), 400

    try:
        location = get_coordinates(state=state, district=district, area=area)

        if not location:
            return jsonify({
                "error": "Location not found",
                "tried": [
                    f"{area}, {district}, {state}" if area and district and state else None,
                    f"{district}, {state}" if district and state else None,
                    f"{area}, {state}" if area and state else None,
                    district if district else None,
                    state if state else None
                ]
            }), 404

        weather_data = get_weather(location["latitude"], location["longitude"])
        selected_day = get_selected_day_weather(weather_data, date)

        return jsonify({
            "requested_location": {
                "state": state,
                "district": district,
                "area": area,
                "date": date
            },
            "resolved_location": location,
            "selected_day_weather": selected_day,
            "raw_weather": weather_data
        })

    except requests.exceptions.HTTPError as e:
        return jsonify({
            "error": "Weather API HTTP error",
            "details": str(e)
        }), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# 13) PREDICT ENDPOINT
# =========================
@app.route("/predict", methods=["GET"])
def predict():
    state = request.args.get("state")
    district = request.args.get("district")
    area = request.args.get("area")
    date = request.args.get("date")

    if not date:
        date = datetime.now().strftime("%Y-%m-%d")

    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return jsonify({"error": "Date must be in YYYY-MM-DD format"}), 400

    if not any([state, district, area]):
        return jsonify({"error": "Please provide at least state, district, or area"}), 400

    try:
        location = get_coordinates(state=state, district=district, area=area)

        if not location:
            return jsonify({
                "error": "Location not found",
                "tried": [
                    f"{area}, {district}, {state}" if area and district and state else None,
                    f"{district}, {state}" if district and state else None,
                    f"{area}, {state}" if area and state else None,
                    district if district else None,
                    state if state else None
                ]
            }), 404

        weather_data = get_weather(location["latitude"], location["longitude"])
        model_input, used_features = prepare_model_input(location, weather_data, date)

        predictions = model.predict(model_input)[0]

        hazard_names = {
            "flood_label": "Flood",
            "cyclone_label": "Cyclone",
            "landslide_label": "Landslide",
            "heatwave_label": "Heatwave"
        }

        result_predictions = {}
        result_confidence = {}
        result_risk_levels = {}

        for i, target in enumerate(target_columns):
            hazard_name = hazard_names.get(target, target)
            pred_value = int(predictions[i])

            probs = model.estimators_[i].predict_proba(model_input)[0]
            positive_prob = round(float(probs[1]) * 100, 2)

            result_predictions[hazard_name] = "Risk" if pred_value == 1 else "No Risk"
            result_confidence[hazard_name] = positive_prob
            result_risk_levels[hazard_name] = get_risk_level(positive_prob)

        max_risk_disaster = max(result_confidence, key=result_confidence.get)
        max_risk_value = result_confidence[max_risk_disaster]
        overall_risk_level = get_risk_level(max_risk_value)

        # EXTRA FEATURE: auto 7-day alerts
        future_alerts = predict_next_7_days(location, weather_data)

        result = {
            "requested_location": {
                "state": state,
                "district": district,
                "area": area,
                "date": date
            },
            "resolved_location": location,
            "predictions": result_predictions,
            "confidence": result_confidence,
            "hazard_risk_levels": result_risk_levels,
            "max_risk_disaster": max_risk_disaster,
            "max_risk_confidence": max_risk_value,
            "overall_risk_level": overall_risk_level,
            "future_alerts": future_alerts,
            "used_features": used_features
        }

        return jsonify(result)

    except requests.exceptions.HTTPError as e:
        return jsonify({
            "error": "Weather API HTTP error",
            "details": str(e)
        }), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)