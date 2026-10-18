import json
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

MIN_CAR_YEAR = 1886
MAX_CAR_YEAR = 2026
FEATURE_COLS = [
    'brand', 'year', 'car_age', 'km_driven', 'fuel',
    'seller_type', 'transmission', 'owner',
]
CAT_COLS = ['brand', 'fuel', 'seller_type', 'transmission', 'owner']


def load_assets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'output')
    model = load_model(os.path.join(output_dir, 'car_cnn_model.keras'))
    scaler = joblib.load(os.path.join(output_dir, 'scaler.pkl'))
    encoders = joblib.load(os.path.join(output_dir, 'label_encoders.pkl'))
    return model, scaler, encoders


def main():
    try:
        input_data = sys.stdin.read()
        if not input_data.strip():
            raise ValueError('Input is empty.')

        data = json.loads(input_data)
        if not isinstance(data, dict):
            raise ValueError('Input must be a JSON object.')

        name = data.get('name')
        if not isinstance(name, str) or not name.strip():
            raise ValueError('Car name is required.')
        brand = name.strip().split()[0]

        year = int(data.get('year'))
        if year < MIN_CAR_YEAR or year > MAX_CAR_YEAR:
            raise ValueError(f'Year must be between {MIN_CAR_YEAR} and {MAX_CAR_YEAR}.')

        km_driven = float(data.get('km_driven'))
        if not np.isfinite(km_driven) or km_driven < 0:
            raise ValueError('Mileage must be a finite number greater than or equal to 0.')

        df = pd.DataFrame([{
            'brand': brand,
            'year': year,
            # Keep this in sync with classification/preprocess.py used to train the model.
            'car_age': 2026 - year,
            'km_driven': km_driven,
            'fuel': data.get('fuel', 'Petrol'),
            'seller_type': data.get('seller_type', 'Individual'),
            'transmission': data.get('transmission', 'Manual'),
            'owner': data.get('owner', 'First Owner'),
        }])

        model, scaler, encoders = load_assets()
        missing_encoders = [col for col in CAT_COLS if col not in encoders]
        if missing_encoders:
            raise ValueError(f'Missing label encoders for: {", ".join(missing_encoders)}.')

        for col in CAT_COLS:
            encoder = encoders[col]
            value = str(df.at[0, col])
            if value not in encoder.classes_:
                raise ValueError(f'Unsupported {col} value: {value}.')
            df[col] = encoder.transform([value])[0]

        scaled = scaler.transform(df[FEATURE_COLS])
        model_input = np.reshape(scaled, (scaled.shape[0], scaled.shape[1], 1))
        prediction = np.asarray(model.predict(model_input, verbose=0))
        if prediction.size < 1:
            raise ValueError('Model returned an empty prediction.')

        predicted_price = float(prediction.reshape(-1)[0])
        if not np.isfinite(predicted_price):
            raise ValueError('Model returned a non-finite prediction.')

        print(json.dumps({'success': True, 'predicted_price': predicted_price}, allow_nan=False))
    except Exception as error:
        print(json.dumps({'success': False, 'error': str(error)}), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
