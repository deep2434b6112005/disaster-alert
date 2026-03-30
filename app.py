from flask import Flask, request, jsonify
import random
from datetime import datetime, timedelta

app = Flask(__name__)

def make_confidence():
    return {
        "Cyclone": round(random.uniform(10, 95), 1),
        "Flood": round(random.uniform(10, 95), 1),
        "Heatwave": round(random.uniform(10, 95), 1),
        "Landslide": round(random.uniform(10, 95), 1),
    }

def confidence_to_risk(val):
    if val >= 65:
        return "High"
    elif val >= 40:
        return "Medium"
    else:
        return "Low"

def confidence_to_prediction(val):
    return "Risk" if val >= 50 else "No Risk"

def make_alert_message(overall, disaster):
    if overall == "High":
        return f"High risk of {disaster} detected. Take immediate precautions."
    elif overall == "Medium":
        return f"Moderate {disaster} risk in your area. Stay alert."
    else:
        return "Conditions are currently stable. Stay prepared."

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
        return jsonify({"error": "latitude and longitude are required"}), 400

    confidence = make_confidence()

    risk_levels = {k: confidence_to_risk(v) for k, v in confidence.items()}
    predictions = {k: confidence_to_prediction(v) for k, v in confidence.items()}

    max_disaster = max(confidence, key=confidence.get)
    max_conf = confidence[max_disaster]
    overall = confidence_to_risk(max_conf)

    future_alerts = []
    for i in range(1, 8):
        fc = make_confidence()
        fr = {k: confidence_to_risk(v) for k, v in fc.items()}
        fp = {k: confidence_to_prediction(v) for k, v in fc.items()}
        fmax_d = max(fc, key=fc.get)
        fmax_c = fc[fmax_d]
        foverall = confidence_to_risk(fmax_c)
        date_str = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")

        future_alerts.append({
            "alert_message": make_alert_message(foverall, fmax_d),
            "confidence": fc,
            "date": date_str,
            "days_remaining": i,
            "hazard_risk_levels": fr,
            "max_risk_confidence": fmax_c,
            "max_risk_disaster": fmax_d,
            "overall_risk_level": foverall,
            "predictions": fp,
        })

    response = {
        "confidence": confidence,
        "future_alerts": future_alerts,
        "hazard_risk_levels": risk_levels,
        "max_risk_confidence": max_conf,
        "max_risk_disaster": max_disaster,
        "overall_risk_level": overall,
        "predictions": predictions,
        "resolved_location": {
            "name": "Demo Location",
            "admin2": "Demo District",
            "admin1": "Demo State",
            "country": "India",
            "latitude": latitude,
            "longitude": longitude,
        }
    }

    return jsonify(response), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
