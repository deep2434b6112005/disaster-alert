from flask import Flask, request, jsonify
import requests
import joblib
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__)

# =========================
# LOAD FLOOD ML FILES
# =========================
model = joblib.load("flood_model.pkl")
feature_columns = joblib.load("flood_feature_columns.pkl")

# =========================
# APIS
# =========================
REVERSE_GEOCODE_API = "https://nominatim.openstreetmap.org/reverse"
FORECAST_API = "https://api.open-meteo.com/v1/forecast"


# =========================
# 1) LOCATION HELPERS
# =========================
def get_location_from_coordinates(lat, lon):
    params = {
        "lat": lat,
        "lon": lon,
        "format": "jsonv2",
        "accept-language": "en"
    }

    headers = {
        "User-Agent": "hazardnet-india/1.0"
    }

    try:
        response = requests.get(
            REVERSE_GEOCODE_API,
            params=params,
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        address = data.get("address", {})

        name = (
            address.get("suburb")
            or address.get("city_district")
            or address.get("city")
            or address.get("town")
            or address.get("village")
            or data.get("name")
            or "Selected Location"
        )

        admin2 = (
            address.get("city")
            or address.get("county")
            or address.get("state_district")
            or ""
        )

        admin1 = address.get("state", "")
        country = address.get("country", "India")

        return {
            "name": name,
            "latitude": lat,
            "longitude": lon,
            "country": country,
            "admin1": admin1,
            "admin2": admin2
        }

    except Exception:
        return {
            "name": "Selected Location",
            "latitude": lat,
            "longitude": lon,
            "country": "India",
            "admin1": "",
            "admin2": ""
        }


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
    hourly = weather_data.get("hourly", {})

    daily_dates = daily.get("time", [])
    rain_sum = daily.get("rain_sum", [])
    precipitation_sum = daily.get("precipitation_sum", [])
    temp_max = daily.get("temperature_2m_max", [])
    temp_min = daily.get("temperature_2m_min", [])
    wind_max = daily.get("wind_speed_10m_max", [])

    hourly_times = hourly.get("time", [])
    hourly_humidity = hourly.get("relative_humidity_2m", [])
    hourly_pressure = hourly.get("surface_pressure", [])
    hourly_wind = hourly.get("wind_speed_10m", [])

    selected_day = {
        "Rainfall (mm)": 0,
        "Temperature (°C)": current.get("temperature_2m", 0),
        "Humidity (%)": current.get("relative_humidity_2m", 0),
        "Forecast Temp Max": 0,
        "Forecast Temp Min": 0,
        "Forecast Wind Max": 0,
        "Surface Pressure": current.get("surface_pressure", 1013),
        "Wind Speed": current.get("wind_speed_10m", 0),
    }

    if selected_date in daily_dates:
        idx = daily_dates.index(selected_date)

        if idx < len(rain_sum) and rain_sum[idx] is not None:
            selected_day["Rainfall (mm)"] = rain_sum[idx]
        elif idx < len(precipitation_sum) and precipitation_sum[idx] is not None:
            selected_day["Rainfall (mm)"] = precipitation_sum[idx]

        selected_day["Forecast Temp Max"] = temp_max[idx] if idx < len(temp_max) and temp_max[idx] is not None else 0
        selected_day["Forecast Temp Min"] = temp_min[idx] if idx < len(temp_min) and temp_min[idx] is not None else 0
        selected_day["Forecast Wind Max"] = wind_max[idx] if idx < len(wind_max) and wind_max[idx] is not None else 0

        if idx < len(temp_max) and idx < len(temp_min):
            max_t = temp_max[idx] if temp_max[idx] is not None else 0
            min_t = temp_min[idx] if temp_min[idx] is not None else 0
            selected_day["Temperature (°C)"] = (max_t + min_t) / 2

    hum_vals = []
    pressure_vals = []
    wind_vals = []

    for i, t in enumerate(hourly_times):
        if t.startswith(selected_date):
            if i < len(hourly_humidity) and hourly_humidity[i] is not None:
                hum_vals.append(hourly_humidity[i])
            if i < len(hourly_pressure) and hourly_pressure[i] is not None:
                pressure_vals.append(hourly_pressure[i])
            if i < len(hourly_wind) and hourly_wind[i] is not None:
                wind_vals.append(hourly_wind[i])

    if hum_vals:
        selected_day["Humidity (%)"] = round(sum(hum_vals) / len(hum_vals), 2)
    if pressure_vals:
        selected_day["Surface Pressure"] = round(sum(pressure_vals) / len(pressure_vals), 2)
    if wind_vals:
        selected_day["Wind Speed"] = round(max(wind_vals), 2)

    return selected_day


# =========================
# 4) LAND COVER LOGIC
# =========================
def get_land_cover_simple(location_name):
    name = location_name.lower()

    major_urban = [
        "chennai", "mumbai", "delhi", "bangalore", "kolkata",
        "hyderabad", "pune", "patna", "new delhi"
    ]
    coastal_water = [
        "chennai", "mumbai", "visakhapatnam", "kochi",
        "nagapattinam", "cuddalore", "puri"
    ]
    forest_places = [
        "nilgiris", "ooty", "kodaikanal", "wayanad",
        "munnar", "coorg", "shimla"
    ]

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

    if any(place in name for place in forest_places):
        land_cover["Land Cover_Forest"] = 1

    return land_cover


# =========================
# 5) POPULATION ESTIMATION
# =========================
def get_population_estimate(location_name):
    name = location_name.lower()

    high_density = [
        "chennai", "mumbai", "delhi", "new delhi",
        "bangalore", "kolkata", "hyderabad", "patna"
    ]
    medium_density = [
        "coimbatore", "madurai", "trichy", "salem",
        "tiruppur", "pune", "gaya", "muzaffarpur"
    ]

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
    location_name = location.get("name", "Selected Location")
    day_weather = get_selected_day_weather(weather, selected_date)

    defaults = {
        "Latitude": location.get("latitude", 0),
        "Longitude": location.get("longitude", 0),
        "Rainfall (mm)": day_weather.get("Rainfall (mm)", 0),
        "Temperature (°C)": day_weather.get("Temperature (°C)", 0),
        "Humidity (%)": day_weather.get("Humidity (%)", 0),
        "River Discharge (m³/s)": 0,
        "Water Level (m)": 0,
        "Elevation (m)": elevation_val,
        "Population Density": get_population_estimate(location_name),
        "Infrastructure": 0,
        "Historical Floods": 0,

        "Land Cover_Agricultural": 0,
        "Land Cover_Desert": 0,
        "Land Cover_Forest": 0,
        "Land Cover_Urban": 0,
        "Land Cover_Water Body": 0,

        "Soil Type_Clay": 0,
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
def get_risk_level(score):
    if score >= 70:
        return "High"
    elif score >= 45:
        return "Medium"
    else:
        return "Low"


# =========================
# 10) LOGIC HELPERS
# =========================
def is_coastal_location(location_name):
    name = location_name.lower()
    coastal_keywords = [
        "chennai", "nagapattinam", "cuddalore", "thoothukudi", "tuticorin",
        "visakhapatnam", "kochi", "mumbai", "puri", "pondicherry", "karaikal"
    ]
    return any(word in name for word in coastal_keywords)


def detect_cyclone_logic(features, location_name=""):
    wind = features.get("Forecast Wind Max", 0)
    pressure = features.get("Surface Pressure", 1013)
    rainfall = features.get("Rainfall (mm)", 0)
    cyclone_season = features.get("is_cyclone_season", 0)
    coastal = is_coastal_location(location_name)

    score = 0

    if coastal:
        score += 15
    if cyclone_season == 1:
        score += 10

    if wind >= 60:
        score += 40
    elif wind >= 45:
        score += 30
    elif wind >= 30:
        score += 15

    if pressure <= 995:
        score += 20
    elif pressure <= 1005:
        score += 10

    if rainfall >= 80:
        score += 15
    elif rainfall >= 40:
        score += 10

    prediction = "Risk" if score >= 50 else "No Risk"
    return prediction, float(score), get_risk_level(score)


def detect_heatwave_logic(features):
    temp = features.get("Temperature (°C)", 0)
    humidity = features.get("Humidity (%)", 0)
    summer = features.get("is_summer", 0)

    score = 0

    if temp >= 42:
        score += 50
    elif temp >= 38:
        score += 35
    elif temp >= 35:
        score += 20

    if humidity <= 35:
        score += 20
    elif humidity <= 50:
        score += 10

    if summer == 1:
        score += 20

    prediction = "Risk" if score >= 45 else "No Risk"
    return prediction, float(score), get_risk_level(score)


def detect_landslide_logic(features, location_name=""):
    rainfall = features.get("Rainfall (mm)", 0)
    elevation = features.get("Elevation (m)", 0)

    forest = features.get("Land Cover_Forest", 0)
    clay = features.get("Soil Type_Clay", 0)
    silt = features.get("Soil Type_Silt", 0)
    peat = features.get("Soil Type_Peat", 0)

    score = 0

    if rainfall >= 100:
        score += 35
    elif rainfall >= 60:
        score += 25
    elif rainfall >= 30:
        score += 10

    if elevation >= 800:
        score += 25
    elif elevation >= 300:
        score += 15

    if forest == 1:
        score += 15

    if clay == 1 or silt == 1 or peat == 1:
        score += 15

    prediction = "Risk" if score >= 45 else "No Risk"
    return prediction, float(score), get_risk_level(score)


# =========================
# 11) FLOOD SANITY RULE
# =========================
def apply_flood_sanity_rule(flood_pred, flood_prob, used_features):
    rainfall = used_features.get("Rainfall (mm)", 0)
    water_level = used_features.get("Water Level (m)", 0)
    river_discharge = used_features.get("River Discharge (m³/s)", 0)

    if rainfall < 5 and water_level <= 0 and river_discharge <= 0:
        flood_pred = 0
        flood_prob = min(float(flood_prob), 25.0)

    return flood_pred, round(float(flood_prob), 1)


# =========================
# 12) 7-DAY ALERTS
# =========================
def make_alert_message(overall, disaster):
    if overall == "High":
        return f"High risk of {disaster} detected. Take immediate precautions."
    elif overall == "Medium":
        return f"Moderate {disaster} risk in this area. Stay alert."
    return "Conditions are currently stable. Stay prepared."


def predict_next_7_days(location, weather_data):
    daily = weather_data.get("daily", {})
    dates = daily.get("time", [])

    alerts = []
    today = datetime.now().date()

    for date in dates:
        model_input, used_features = prepare_model_input(location, weather_data, date)

        flood_pred = int(model.predict(model_input)[0])
        flood_prob = round(float(model.predict_proba(model_input)[0][1]) * 100, 1)
        flood_pred, flood_prob = apply_flood_sanity_rule(flood_pred, flood_prob, used_features)

        location_name = location.get("name", "Selected Location")
        cyclone_prediction, cyclone_score, cyclone_risk = detect_cyclone_logic(used_features, location_name)
        heatwave_prediction, heatwave_score, heatwave_risk = detect_heatwave_logic(used_features)
        landslide_prediction, landslide_score, landslide_risk = detect_landslide_logic(used_features, location_name)

        day_predictions = {
            "Cyclone": cyclone_prediction,
            "Flood": "Risk" if flood_pred == 1 else "No Risk",
            "Heatwave": heatwave_prediction,
            "Landslide": landslide_prediction
        }

        day_confidence = {
            "Cyclone": cyclone_score,
            "Flood": flood_prob,
            "Heatwave": heatwave_score,
            "Landslide": landslide_score
        }

        day_risk_levels = {
            "Cyclone": cyclone_risk,
            "Flood": get_risk_level(flood_prob),
            "Heatwave": heatwave_risk,
            "Landslide": landslide_risk
        }

        max_risk_disaster = max(day_confidence, key=day_confidence.get)
        max_risk_confidence = day_confidence[max_risk_disaster]
        overall_risk_level = get_risk_level(max_risk_confidence)

        event_date = datetime.strptime(date, "%Y-%m-%d").date()
        days_remaining = (event_date - today).days

        if days_remaining >= 0 and max_risk_confidence >= 45:
            alerts.append({
                "alert_message": make_alert_message(overall_risk_level, max_risk_disaster),
                "confidence": day_confidence,
                "date": date,
                "days_remaining": days_remaining,
                "hazard_risk_levels": day_risk_levels,
                "max_risk_confidence": max_risk_confidence,
                "max_risk_disaster": max_risk_disaster,
                "overall_risk_level": overall_risk_level,
                "predictions": day_predictions
            })

    return alerts


# =========================
# 13) ROOT
# =========================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "message": "HazardNet India API is running",
        "predict_method": "POST",
        "sample_body": {
            "latitude": 13.0827,
            "longitude": 80.2707
        }
    })


# =========================
# 14) WEATHER ENDPOINT
# =========================
@app.route("/weather", methods=["GET"])
def weather():
    lat = request.args.get("latitude")
    lon = request.args.get("longitude")
    date = request.args.get("date")

    if not date:
        date = datetime.now().strftime("%Y-%m-%d")

    if lat is None or lon is None:
        return jsonify({"error": "latitude and longitude are required"}), 400

    try:
        lat = float(lat)
        lon = float(lon)
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return jsonify({"error": "Invalid latitude, longitude, or date"}), 400

    try:
        location = get_location_from_coordinates(lat, lon)
        weather_data = get_weather(lat, lon)
        selected_day = get_selected_day_weather(weather_data, date)

        return jsonify({
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
# 15) PREDICT ENDPOINT
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}

    latitude = data.get("latitude")
    longitude = data.get("longitude")
    date = data.get("date")

    if not date:
        date = datetime.now().strftime("%Y-%m-%d")

    if latitude is None or longitude is None:
        return jsonify({
            "error": "latitude and longitude are required"
        }), 400

    try:
        latitude = float(latitude)
        longitude = float(longitude)
        datetime.strptime(date, "%Y-%m-%d")
    except (TypeError, ValueError):
        return jsonify({
            "error": "latitude and longitude must be numeric and date must be YYYY-MM-DD"
        }), 400

    try:
        location = get_location_from_coordinates(latitude, longitude)
        weather_data = get_weather(latitude, longitude)
        selected_day_weather = get_selected_day_weather(weather_data, date)
        model_input, used_features = prepare_model_input(location, weather_data, date)

        flood_pred = int(model.predict(model_input)[0])
        flood_prob = round(float(model.predict_proba(model_input)[0][1]) * 100, 1)
        flood_pred, flood_prob = apply_flood_sanity_rule(flood_pred, flood_prob, used_features)

        location_name = location.get("name", "Selected Location")
        cyclone_prediction, cyclone_score, cyclone_risk = detect_cyclone_logic(used_features, location_name)
        heatwave_prediction, heatwave_score, heatwave_risk = detect_heatwave_logic(used_features)
        landslide_prediction, landslide_score, landslide_risk = detect_landslide_logic(used_features, location_name)

        predictions = {
            "Cyclone": cyclone_prediction,
            "Flood": "Risk" if flood_pred == 1 else "No Risk",
            "Heatwave": heatwave_prediction,
            "Landslide": landslide_prediction
        }

        confidence = {
            "Cyclone": cyclone_score,
            "Flood": flood_prob,
            "Heatwave": heatwave_score,
            "Landslide": landslide_score
        }

        hazard_risk_levels = {
            "Cyclone": cyclone_risk,
            "Flood": get_risk_level(flood_prob),
            "Heatwave": heatwave_risk,
            "Landslide": landslide_risk
        }

        max_risk_disaster = max(confidence, key=confidence.get)
        max_risk_confidence = confidence[max_risk_disaster]
        overall_risk_level = get_risk_level(max_risk_confidence)

        future_alerts = predict_next_7_days(location, weather_data)

        response = {
            "confidence": confidence,
            "future_alerts": future_alerts,
            "hazard_risk_levels": hazard_risk_levels,
            "max_risk_confidence": max_risk_confidence,
            "max_risk_disaster": max_risk_disaster,
            "overall_risk_level": overall_risk_level,
            "predictions": predictions,
            "resolved_location": {
                "name": location.get("name", "Selected Location"),
                "admin2": location.get("admin2", ""),
                "admin1": location.get("admin1", ""),
                "country": location.get("country", "India"),
                "latitude": latitude,
                "longitude": longitude
            },
            "selected_day_weather": selected_day_weather
        }

        return jsonify(response), 200

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
