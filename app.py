
from PIL import Image
import datetime
from flask import jsonify
from datetime import datetime
from jinja2.exceptions import UndefinedError
from flask import Flask, request, jsonify

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

UPLOAD_FOLDER = 'D:/sems/las/preapp/static/uploads'
DETECTION_FOLDER = 'D:/sems/las/preapp/static/detected'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DETECTION_FOLDER'] = DETECTION_FOLDER

model_path = 'D:/sems/las/preapp/best10.pt'
detector = YOLO(model=model_path, task='detect')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


api_key = '932bf300eb9e17980c2120509abd0070'


def get_weather_data(api_key, city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print("Failed to retrieve weather data.")
        return None


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
        model_file = 'D:/sems/las/preapp/area_model.pkl'

        if not os.path.exists(model_file):
            print(
                "Trained model file does not exist. Please train and save the model first.")
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


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/livedetect')
def homes():
    return render_template('live-detect.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image part in the request"

    file = request.files['image']

    if file.filename == '':
        return "No selected image file"

    city = request.form.get('city') or file.filename[:-4].replace('_', ' ')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        city = request.form.get('city') or file.filename[:-4].replace('_', ' ')
        file.save(file_path)
        results = detector(file_path)

        if not os.path.exists(app.config['DETECTION_FOLDER']):
            os.makedirs(app.config['DETECTION_FOLDER'])

        result = results[0]

        output_filename = os.path.join(
            app.config['DETECTION_FOLDER'], f"{filename}")

        i_path = output_filename.replace("\\", "/")
        image_path = "/" + i_path.split("/static/")[-1]

        if os.path.exists(app.config['DETECTION_FOLDER']):
            result.save(output_filename)

            if len(result.boxes) > 0:
                fire_detected = False
                smoke_detected = False

                for box in result.boxes:
                    if box.cls == 0:
                        fire_detected = True
                    elif box.cls == 1:
                        smoke_detected = True

                estimated_area = predict_fire_area(api_key, city)
                if estimated_area is not None:
                    weather_data = get_weather_data(api_key, city)
                    try:
                        return render_template('predict.html', fire_detected=fire_detected, smoke_detected=smoke_detected, estimated_area=estimated_area, image_path=image_path, weather_data=weather_data, city=city)
                    except UndefinedError as e:
                        error_message = str(e)
                        return render_template('predict.html', fire_detected=fire_detected, smoke_detected=smoke_detected, image_path=image_path, error_message=error_message, city=city)
                else:
                    error_message = "Failed to calculate estimated fire area."
                    return render_template('predict.html', fire_detected=fire_detected, smoke_detected=smoke_detected, image_path=image_path, error_message=error_message, city=city)
            else:
                fire_detected = False
                smoke_detected = False
                return render_template('predict.html', fire_detected=fire_detected, smoke_detected=smoke_detected, image_path=image_path, city=city)

    return "Invalid file format"


TEMP_FOLDER = 'D:/sems/las/preapp/static/temp'


def save_image(image_file):
    """Saves the uploaded image to a temporary file with a timestamp."""
    try:

        os.makedirs(TEMP_FOLDER, exist_ok=True)

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        filename = f'{timestamp}.jpg'

        image_path = os.path.join(TEMP_FOLDER, filename)
        image_file.save(image_path)

        return image_path

    except Exception as e:
        print(f"Error saving image: {e}")
        return None


@app.route('/detect-fire', methods=['POST'])
def detect_fire():
    if 'image' not in request.files:
        return jsonify({"message": "No image part in the request"}), 400

    image_file = request.files['image']
    image_path = save_image(image_file)

    if not image_path:
        return jsonify({"message": "Error saving image"}), 500

    try:

        image = image_path
        print(image_path)

        results = detector(image)

        fire_detected = False
        smoke_detected = False
        result = results[0]
        if len(result.boxes) > 0:
            fire_detected = False
            smoke_detected = False

            for box in result.boxes:
                if box.cls == 0:
                    fire_detected = True
                elif box.cls == 1:
                    smoke_detected = True

        message = prepare_message(fire_detected, smoke_detected)
        print(message)
        return jsonify({"fire_detected": fire_detected, "smoke_detected": smoke_detected, "message": message})

    except Exception as e:
        print(f"Error detecting fire: {e}")
        return jsonify({"message": "Internal server error"}), 500


def prepare_message(fire_detected, smoke_detected):
    """Prepares a message based on detection results."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if fire_detected and smoke_detected:
        message = f"fire and smoke detected at {now}"
    elif fire_detected:
        message = f"Fire detected at {now}"
    elif smoke_detected:
        message = f"Smoke detected at {now}"
    else:
        message = f"No fire or smoke detected at {now}"
    return message


if __name__ == '__main__':
    app.run(debug=True)
