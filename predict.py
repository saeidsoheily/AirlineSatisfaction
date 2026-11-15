import pickle
import pandas as pd
import numpy as np
import xgboost as xgb

from flask import Flask, request, jsonify

# Initialization
model_filename = 'model_xgboost.bin'
with open(model_filename, 'rb') as f_in:
    (dv, model) = pickle.load(f_in)

rating_cols = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Cleanliness"
]

digital_cols = [
    "Inflight wifi service",
    "Ease of Online booking",
    "Online boarding"
]

comfort_cols = [
    "Seat comfort",
    "Leg room service",
    "Food and drink",
    "Cleanliness",
    "Inflight entertainment"
]

staff_cols = [
    "On-board service",
    "Checkin service",
    "Baggage handling"
]

not_rated_cols = [
    "Inflight wifi service",
    "Ease of Online booking",
    "Online boarding",
    "Departure/Arrival time convenient"
]

def feature_engineering(X):
    X = X.copy()

    # Service ratings
    # Zero means Not Rated, not dissatisfaction
    ratings = X[rating_cols].replace(0, np.nan)

    X["service_mean"] = ratings.mean(axis=1)
    X["service_min"] = ratings.min(axis=1)
    X["service_std"] = ratings.std(axis=1).fillna(0)

    X["low_rating_count"] = ratings.le(2).sum(axis=1)
    X["high_rating_count"] = ratings.ge(4).sum(axis=1)
    X["rated_service_count"] = ratings.notna().sum(axis=1)

    # Group-specific scores
    X["digital_service_mean"] = ratings[digital_cols].mean(axis=1)
    X["comfort_mean"] = ratings[comfort_cols].mean(axis=1)
    X["staff_service_mean"] = ratings[staff_cols].mean(axis=1)

    # Difference between experience dimensions
    X["comfort_vs_digital"] = (
        X["comfort_mean"] - X["digital_service_mean"]
    )

    # Not-rated features
    for col in not_rated_cols:
        X[f"{col}_not_rated"] = X[col].eq(0).astype("int8")

    # C. Delay features
    dep = X["Departure Delay in Minutes"]
    arr = X["Arrival Delay in Minutes"]

    X["arrival_delay_missing"] = arr.isna().astype("int8")

    # Original chosen imputation: zero
    arr = arr.fillna(0)
    X["Arrival Delay in Minutes"] = arr

    X["total_delay"] = dep + arr
    X["delay_difference"] = arr - dep

    X["delay_recovered"] = (
        (dep > arr) & (dep > 0)
    ).astype("int8")

    X["delayed_over_30"] = (
        (dep > 30) | (arr > 30)
    ).astype("int8")

    X["delayed_over_60"] = (
        (dep > 60) | (arr > 60)
    ).astype("int8")

    # D. Passenger profile interactions
    X["class_travel"] = (
        X["Class"].astype(str)
        + "_"
        + X["Type of Travel"].astype(str)
    )

    X["class_customer"] = (
        X["Class"].astype(str)
        + "_"
        + X["Customer Type"].astype(str)
    )

    # E. Fill derived missing aggregates
    derived_cols = [
        "service_mean",
        "service_min",
        "digital_service_mean",
        "comfort_mean",
        "staff_service_mean",
        "comfort_vs_digital"
    ]

    X[derived_cols] = X[derived_cols].fillna(-1)
    return X


app = Flask('Model Prediction')

@app.route('/predict', methods=['POST'])
def predict():
    passenger = request.get_json()

    # Convert passenger dictionary to DataFrame
    passenger_df = pd.DataFrame([passenger])

    # Apply the SAME feature engineering
    passenger_fe = feature_engineering(passenger_df)

    # Convert engineered features to dictionary
    passenger_dict = passenger_fe.to_dict(orient="records")

    # Use the saved DictVectorizer
    X = dv.transform(passenger_dict)

    # Convert to DMatrix
    dtest = xgb.DMatrix(
        X,
        feature_names=dv.get_feature_names_out().tolist()
    )

    # Predict
    probability = float(model.predict(dtest)[0])

    return {
        "satisfaction_probability": probability,
        "satisfied": probability >= 0.5
    }

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9000)
