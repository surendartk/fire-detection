import os
import random
import pickle
import pandas as pd
import xgboost as xgb
import requests
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)

# Define paths
UPLOAD_FOLDER = 'D:/sems/des/preapp/static/uploads'
DETECTION_FOLDER = 'D:/sems/des/preapp/static/detected'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DETECTION_FOLDER'] = DETECTION_FOLDER

# Load YOLO detector
model_path = "D:/sems/des/preapp/model.pt"
detector = YOLO(model=model_path, task='detect')

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

api_key = '932bf300eb9e17980c2120509abd0070'

# Function to get weather data from OpenWeatherMap API
def get_weather_data(api_key, city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print("Failed to retrieve weather data.")
        return None

# Function to predict estimated fire area
def predict_fire_area(api_key, city):
    weather_data = get_weather_data(api_key, city)

    if weather_data:
        temp = weather_data['main']['temp']
        rh = weather_data['main']['humidity']
        wind = weather_data['wind']['speed']
        pre = weather_data['rain']['1h'] if 'rain' in weather_data and '1h' in weather_data['rain'] else 0

        new_data = {
            'Region': 1,
            'Precipitation (km^2)': pre,
            'RelativeHumidity (%)': rh,
            'SoilWaterContent': random.uniform(0, 1),
            'SolarRadiation': random.uniform(0, 1000),
            'Temperature': temp,
            'WindSpeed': wind
        }

        new_df = pd.DataFrame([new_data])
        model_file = 'D:/sems/des/preapp/area_model.pkl'

        if not os.path.exists(model_file):
            print("Trained model file does not exist. Please train and save the model first.")
            return None

        with open(model_file, 'rb') as file:
            model = pickle.load(file)

        if model is None:
            print("Error: Failed to load the model.")
            return None

        new_dmatrix = xgb.DMatrix(data=new_df)
        predicted_area = model.predict(new_dmatrix)

        return predicted_area

    else:
        return None

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle image upload and detection
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image part in the request"

    file = request.files['image']

    if file.filename == '':
        return "No selected image file"

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(file_path)
        city = filename[:-4]
        file.save(file_path)
        results = detector(file_path)

        if not os.path.exists(app.config['DETECTION_FOLDER']):
            os.makedirs(app.config['DETECTION_FOLDER'])

        result = results[0]

        output_filename = os.path.join(
            app.config['DETECTION_FOLDER'], f"{filename}")

        print(output_filename)

        i_path = output_filename.replace("\\", "/")
        image_path = "/" + i_path.split("/static/")[-1]
        print(image_path)

        if os.path.exists(app.config['DETECTION_FOLDER']):
            result.save(output_filename)

            if len(result.boxes) > 0:
                fire_detected = True
                # Calculate estimated fire area when fire is detected
                estimated_area = predict_fire_area(api_key, city)
                if estimated_area is not None:
                    # Send weather data along with fire detection result to the prediction template
                    return render_template('predict.html', fire_detected=fire_detected, estimated_area=estimated_area, image_path=image_path, weather_data=get_weather_data(api_key, city))
                else:
                    return render_template('predict.html', fire_detected=fire_detected, image_path=image_path, error_message="Failed to calculate estimated fire area.")
            else:
                fire_detected = False
                return render_template('predict.html', fire_detected=fire_detected, image_path=image_path)

    return "Invalid file format"

if __name__ == '__main__':
    app.run(debug=True)
