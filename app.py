from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import os
import sqlite3
from datetime import datetime
import pickle
import logging
import requests
import sys

# Configure logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
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
    try:
        conn = sqlite3.connect('reviews.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS reviews
                     (id INTEGER PRIMARY KEY, username TEXT, review TEXT, rating INTEGER, timestamp TEXT)''')
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

# Download files from cloud if needed
def download_file(url, destination):
    if not os.path.exists(destination):
        logger.info(f"Downloading {destination} from {url}")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"{destination} downloaded successfully")
        except Exception as e:
            logger.error(f"Error downloading {destination}: {str(e)}")
            raise

# Load and cache data using chunks to reduce memory usage
def load_data(sample_size=5000, chunksize=100000):
    global cached_data
    if cached_data is not None:
        logger.info("Returning cached data")
        return cached_data

    data_path = os.path.join(os.path.dirname(__file__), 'household_power_consumption.txt')
    # Download dataset if not present (uncomment and provide URL)
    # data_url = 'https://your-cloud-storage-url/household_power_consumption.txt'
    # download_file(data_url, data_path)

    if not os.path.exists(data_path):
        logger.error(f"Dataset not found at {data_path}")
        return None

    try:
        # Define date format for parsing
        date_format = '%d/%m/%Y %H:%M:%S'
        # Read the dataset in chunks with explicit date format
        chunks = pd.read_csv(
            data_path,
            sep=';',
            parse_dates={'datetime': ['Date', 'Time']},
            date_format=date_format,
            low_memory=False,
            chunksize=chunksize
        )

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
        logger.info(f"Loaded and cached dataset with shape: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return None

# Train and save models if they don't exist
def train_and_save_models():
    data = load_data(sample_size=5000)
    if data is None:
        logger.error("Cannot train models: Data loading failed.")
        return False

    features = ['datetime', 'Global_reactive_power', 'Voltage', 'Global_intensity']
    X = data[features]
    
    try:
        for target in models.keys():
            y = data[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

            # Train Linear Regression
            lin_model = LinearRegression()
            lin_model.fit(X_train, y_train)
            lin_path = os.path.join(os.path.dirname(__file__), f'{target}_lin.pkl')
            with open(lin_path, 'wb') as f:
                pickle.dump(lin_model, f)
            logger.info(f"Saved Linear Regression model for {target} to {lin_path}")

            # Train Ridge Regression
            ridge_model = Ridge()
            ridge_model.fit(X_train, y_train)
            ridge_path = os.path.join(os.path.dirname(__file__), f'{target}_ridge.pkl')
            with open(ridge_path, 'wb') as f:
                pickle.dump(ridge_model, f)
            logger.info(f"Saved Ridge Regression model for {target} to {ridge_path}")

            # Train XGBoost with reduced complexity
            xgb_model = XGBRegressor(n_estimators=30, max_depth=2, random_state=42)
            xgb_model.fit(X_train, y_train)
            xgb_path = os.path.join(os.path.dirname(__file__), f'{target}_xgb.pkl')
            with open(xgb_path, 'wb') as f:
                pickle.dump(xgb_model, f)
            logger.info(f"Saved XGBoost model for {target} to {xgb_path}")

        return True
    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        return False

# Load pre-trained models
def load_models():
    global models
    try:
        # Download model files if not present (uncomment and provide URLs)
        # model_urls = {
        #     'Global_active_power_lin.pkl': 'https://your-cloud-storage-url/Global_active_power_lin.pkl',
        #     'Global_active_power_ridge.pkl': 'https://your-cloud-storage-url/Global_active_power_ridge.pkl',
        #     'Global_active_power_xgb.pkl': 'https://your-cloud-storage-url/Global_active_power_xgb.pkl',
        #     'Sub_metering_1_lin.pkl': 'https://your-cloud-storage-url/Sub_metering_1_lin.pkl',
        #     'Sub_metering_1_ridge.pkl': 'https://your-cloud-storage-url/Sub_metering_1_ridge.pkl',
        #     'Sub_metering_1_xgb.pkl': 'https://your-cloud-storage-url/Sub_metering_1_xgb.pkl',
        #     'Sub_metering_2_lin.pkl': 'https://your-cloud-storage-url/Sub_metering_2_lin.pkl',
        #     'Sub_metering_2_ridge.pkl': 'https://your-cloud-storage-url/Sub_metering_2_ridge.pkl',
        #     'Sub_metering_2_xgb.pkl': 'https://your-cloud-storage-url/Sub_metering_2_xgb.pkl',
        #     'Sub_metering_3_lin.pkl': 'https://your-cloud-storage-url/Sub_metering_3_lin.pkl',
        #     'Sub_metering_3_ridge.pkl': 'https://your-cloud-storage-url/Sub_metering_3_ridge.pkl',
        #     'Sub_metering_3_xgb.pkl': 'https://your-cloud-storage-url/Sub_metering_3_xgb.pkl',
        # }
        # for file_name, url in model_urls.items():
        #     download_file(url, os.path.join(os.path.dirname(__file__), file_name))

        # Check if model files exist
        for target in models.keys():
            model_files = {
                'lin': os.path.join(os.path.dirname(__file__), f'{target}_lin.pkl'),
                'ridge': os.path.join(os.path.dirname(__file__), f'{target}_ridge.pkl'),
                'xgb': os.path.join(os.path.dirname(__file__), f'{target}_xgb.pkl')
            }
            if not all(os.path.exists(path) for path in model_files.values()):
                logger.info(f"Model files for {target} not found. Attempting to train new models...")
                if not train_and_save_models():
                    logger.error(f"Failed to train models for {target}")
                    return False

        # Load the models
        for target in models.keys():
            models[target]['lin'] = pickle.load(open(os.path.join(os.path.dirname(__file__), f'{target}_lin.pkl'), 'rb'))
            models[target]['ridge'] = pickle.load(open(os.path.join(os.path.dirname(__file__), f'{target}_ridge.pkl'), 'rb'))
            models[target]['xgb'] = pickle.load(open(os.path.join(os.path.dirname(__file__), f'{target}_xgb.pkl'), 'rb'))
            logger.info(f"Loaded models for {target}")
        logger.info("All models loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return False

# Initialize models at startup
try:
    init_db()
    if not load_models():
        logger.error("Application startup failed: Models not initialized")
        sys.exit(1)
except Exception as e:
    logger.error(f"Startup error: {str(e)}")
    sys.exit(1)

# Debug endpoint to check file and model status
@app.route('/debug')
def debug():
    model_files = {f'{target}_{model}.pkl' for target in models.keys() for model in ['lin', 'ridge', 'xgb']}
    file_status = {file: os.path.exists(os.path.join(os.path.dirname(__file__), file)) for file in model_files}
    file_status['dataset'] = os.path.exists(os.path.join(os.path.dirname(__file__), 'household_power_consumption.txt'))
    return jsonify({
        'model_initialized': {target: all(model is not None for model in models[target].values()) for target in models.keys()},
        'file_status': file_status
    })

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
                    logger.error(f"Model for {target} (lin) is not initialized")
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
            logger.error(f"Prediction error: {str(e)}")
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
                logger.error(f"Model for {target} (lin) is not initialized")
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
        logger.error(f"Comparison error: {str(e)}")
        return render_template('error.html', message=f"Comparison error: {str(e)}")

# Review page
@app.route('/reviews', methods=['GET', 'POST'])
def reviews():
    try:
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
    except Exception as e:
        logger.error(f"Error handling reviews: {str(e)}")
        return render_template('error.html', message=f"Reviews error: {str(e)}")

# Error template
@app.route('/error')
def error():
    message = request.args.get('message', 'An error occurred')
    return render_template('error.html', message=message)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=False, processes=1)
