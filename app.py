from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import os

app = Flask(__name__)

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

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}

    latitude = data.get("latitude")
    longitude = data.get("longitude")

    if latitude is None or longitude is None:
        return jsonify({
            "error": "latitude and longitude are required"
        }), 400

    try:
        latitude = float(latitude)
        longitude = float(longitude)
    except (TypeError, ValueError):
        return jsonify({
            "error": "latitude and longitude must be numeric"
        }), 400

    confidence = {
        "Cyclone": 35.4,
        "Flood": 78.6,
        "Heatwave": 20.2,
        "Landslide": 44.8
    }

    predictions = {
        "Cyclone": "No Risk",
        "Flood": "Risk",
        "Heatwave": "No Risk",
        "Landslide": "No Risk"
    }

    hazard_risk_levels = {
        "Cyclone": "Low",
        "Flood": "High",
        "Heatwave": "Low",
        "Landslide": "Medium"
    }

    max_risk_disaster = "Flood"
    max_risk_confidence = 78.6
    overall_risk_level = "High"

    future_alerts = []
    for i in range(1, 8):
        date_str = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")

        future_alerts.append({
            "alert_message": "Moderate Flood risk in your area. Stay alert.",
            "confidence": {
                "Cyclone": 32.5,
                "Flood": 74.2,
                "Heatwave": 28.1,
                "Landslide": 41.7
            },
            "date": date_str,
            "days_remaining": i,
            "hazard_risk_levels": {
                "Cyclone": "Low",
                "Flood": "High",
                "Heatwave": "Low",
                "Landslide": "Medium"
            },
            "max_risk_confidence": 74.2,
            "max_risk_disaster": "Flood",
            "overall_risk_level": "High",
            "predictions": {
                "Cyclone": "No Risk",
                "Flood": "Risk",
                "Heatwave": "No Risk",
                "Landslide": "No Risk"
            }
        })

    response = {
        "confidence": confidence,
        "future_alerts": future_alerts,
        "hazard_risk_levels": hazard_risk_levels,
        "max_risk_confidence": max_risk_confidence,
        "max_risk_disaster": max_risk_disaster,
        "overall_risk_level": overall_risk_level,
        "predictions": predictions,
        "resolved_location": {
            "name": "Demo Location",
            "admin2": "Demo District",
            "admin1": "Demo State",
            "country": "India",
            "latitude": latitude,
            "longitude": longitude
        }
    }

    return jsonify(response), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
