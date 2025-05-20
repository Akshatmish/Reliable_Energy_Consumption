from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import sqlite3
from datetime import datetime
import pickle

# Initialize Flask app with minimal workers to reduce memory usage
app = Flask(__name__, template_folder='templates', static_folder='static')

# Global variables for models
models = {
    'Global_active_power': {'lin': None, 'ridge': None, 'xgb': None},
    'Sub_metering_1': {'lin': None, 'ridge': None, 'xgb': None},
    'Sub_metering_2': {'lin': None, 'ridge': None, 'xgb': None},
    'Sub_metering_3': {'lin': None, 'ridge': None, 'xgb': None}
}

# Cache for the sampled dataset
cached_data = None

# Database setup for reviews
def init_db():
    conn = sqlite3.connect('reviews.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS reviews
                 (id INTEGER PRIMARY KEY, username TEXT, review TEXT, rating INTEGER, timestamp TEXT)''')
    conn.commit()
    conn.close()

# Load and cache data using chunks to reduce memory usage
def load_data(sample_size=10000, chunksize=100000):
    global cached_data
    if cached_data is not None:
        return cached_data

    try:
        # Read the dataset in chunks to avoid loading the entire file into memory
        chunks = pd.read_csv('household_power_consumption.txt', sep=';',
                             parse_dates={'datetime': ['Date', 'Time']},
                             infer_datetime_format=True,
                             low_memory=False,
                             chunksize=chunksize)

        # Process chunks and sample
        sampled_data = []
        total_rows = 0
        for chunk in chunks:
            chunk = chunk.apply(pd.to_numeric, errors='coerce')
            chunk['datetime'] = chunk['datetime'].astype('int64') // 10**9
            chunk = chunk.dropna()
            total_rows += len(chunk)
            
            # Sample proportionally from each chunk
            sample_fraction = min(sample_size / total_rows, 1.0) if total_rows > 0 else 1.0
            sampled_chunk = chunk.sample(frac=sample_fraction, random_state=42) if sample_fraction < 1 else chunk
            sampled_data.append(sampled_chunk)
            
            # Stop if we have enough samples
            if sum(len(df) for df in sampled_data) >= sample_size:
                break

        # Concatenate sampled chunks
        data = pd.concat(sampled_data, axis=0)
        if len(data) > sample_size:
            data = data.sample(n=sample_size, random_state=42)

        cached_data = data
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Train and save models if they don't exist (optimized for memory)
def train_and_save_models():
    data = load_data(sample_size=5000)  # Use a smaller sample for training
    if data is None:
        print("Cannot train models: Data loading failed.")
        return False

    features = ['datetime', 'Global_reactive_power', 'Voltage', 'Global_intensity']
    X = data[features]
    
    for target in models.keys():
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        # Train Linear Regression
        lin_model = LinearRegression()
        lin_model.fit(X_train, y_train)
        with open(f'{target}_lin.pkl', 'wb') as f:
            pickle.dump(lin_model, f)

        # Train Ridge Regression
        ridge_model = Ridge()
        ridge_model.fit(X_train, y_train)
        with open(f'{target}_ridge.pkl', 'wb') as f:
            pickle.dump(ridge_model, f)

        # Train XGBoost with further reduced complexity
        xgb_model = XGBRegressor(n_estimators=30, max_depth=2, random_state=42)  # Further reduced complexity
        xgb_model.fit(X_train, y_train)
        with open(f'{target}_xgb.pkl', 'wb') as f:
            pickle.dump(xgb_model, f)

    return True

# Load pre-trained models
def load_models():
    global models
    try:
        # Check if model files exist, if not, train and save them
        for target in models.keys():
            if not all(os.path.exists(f'{target}_{model}.pkl') for model in ['lin', 'ridge', 'xgb']):
                print(f"Model files for {target} not found. Training new models...")
                if not train_and_save_models():
                    return False

        # Load the models
        for target in models.keys():
            models[target]['lin'] = pickle.load(open(f'{target}_lin.pkl', 'rb'))
            models[target]['ridge'] = pickle.load(open(f'{target}_ridge.pkl', 'rb'))
            models[target]['xgb'] = pickle.load(open(f'{target}_xgb.pkl', 'rb'))
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

# Home page
@app.route('/')
def index():
    data = load_data()
    if data is None:
        return render_template('error.html', message="Failed to load data")
    
    sample_data = data.head().to_dict(orient='records')
    stats = {
        'total_records': len(data),
        'columns': list(data.columns),
        'mean_power': float(data['Global_active_power'].mean()),
        'max_power': float(data['Global_active_power'].max())
    }
    return render_template('index.html', sample_data=sample_data, stats=stats)

# Prediction page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            input_data = {
                'datetime': pd.Timestamp(request.form['datetime']).value // 10**9,
                'Global_reactive_power': float(request.form['global_reactive']),
                'Voltage': float(request.form['voltage']),
                'Global_intensity': float(request.form['global_intensity'])
            }
            input_df = pd.DataFrame([input_data])
            input_df = input_df[['datetime', 'Global_reactive_power', 'Voltage', 'Global_intensity']]
            
            predictions = {}
            for target in models.keys():
                if models[target]['lin'] is None:
                    return render_template('error.html', message="Models not initialized")
                predictions[target] = {
                    'lin': float(models[target]['lin'].predict(input_df)[0]),
                    'ridge': float(models[target]['ridge'].predict(input_df)[0]),
                    'xgb': float(models[target]['xgb'].predict(input_df)[0])
                }
            
            total_power = predictions['Global_active_power']['xgb']
            percentages = {
                'Sub_metering_1': float((predictions['Sub_metering_1']['xgb'] / total_power) * 100 if total_power > 0 else 0),
                'Sub_metering_2': float((predictions['Sub_metering_2']['xgb'] / total_power) * 100 if total_power > 0 else 0),
                'Sub_metering_3': float((predictions['Sub_metering_3']['xgb'] / total_power) * 100 if total_power > 0 else 0)
            }
            
            return render_template('predict.html', total_power=total_power, percentages=percentages, input_data=input_data, pd=pd)
        except Exception as e:
            return render_template('error.html', message=f"Prediction error: {str(e)}")
    return render_template('predict.html')

# Compare route
@app.route('/compare')
def compare():
    data = load_data()
    if data is None:
        return render_template('error.html', message="Failed to load data")
    
    features = ['datetime', 'Global_reactive_power', 'Voltage', 'Global_intensity']
    X = data[features]
    metrics = {}
    
    try:
        for target in models.keys():
            if models[target]['lin'] is None:
                return render_template('error.html', message="Models not initialized")
                
            y = data[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
            
            lin_pred = models[target]['lin'].predict(X_test)
            ridge_pred = models[target]['ridge'].predict(X_test)
            xgb_pred = models[target]['xgb'].predict(X_test)
            
            metrics[target] = {
                'models': ['Linear Regression', 'Ridge Regression', 'XGBoost'],
                'mae': [
                    float(mean_absolute_error(y_test, lin_pred)),
                    float(mean_absolute_error(y_test, ridge_pred)),
                    float(mean_absolute_error(y_test, xgb_pred))
                ],
                'r2': [
                    float(r2_score(y_test, lin_pred)),
                    float(r2_score(y_test, ridge_pred)),
                    float(r2_score(y_test, xgb_pred))
                ]
            }
        return render_template('compare.html', metrics=metrics)
    except Exception as e:
        return render_template('error.html', message=f"Comparison error: {str(e)}")

# Review page
@app.route('/reviews', methods=['GET', 'POST'])
def reviews():
    conn = sqlite3.connect('reviews.db')
    c = conn.cursor()
    
    if request.method == 'POST':
        username = request.form['username']
        review = request.form['review']
        rating = int(request.form['rating'])
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        c.execute("INSERT INTO reviews (username, review, rating, timestamp) VALUES (?, ?, ?, ?)",
                  (username, review, rating, timestamp))
        conn.commit()
    
    c.execute("SELECT * FROM reviews ORDER BY timestamp DESC")
    reviews = c.fetchall()
    conn.close()
    
    return render_template('reviews.html', reviews=reviews)

# Error template
@app.route('/error')
def error():
    message = request.args.get('message', 'An error occurred')
    return render_template('error.html', message=message)

if __name__ == '__main__':
    init_db()
    if load_models():
        # Run Flask with a single worker to minimize memory usage
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=False, processes=1)
    else:
        print("Failed to initialize application.")








