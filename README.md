# 🔥 Live Fire Detection & Area Estimation Web App

This Flask-based web application detects fire and smoke from uploaded images using a pre-trained YOLOv8 model and estimates the affected area using weather data via an XGBoost regression model. It also fetches your live location and weather details using `geopy` and OpenWeatherMap API.

---

## 📦 requirements.txt


---

## 🚀 Features

- 🔍 Detects **fire** and **smoke** using a YOLOv8 model.
- 📍 Fetches your **live location** and **city**.
- 🌦️ Integrates with **OpenWeatherMap API** to fetch live weather data.
- 📏 Predicts estimated fire-affected **area using XGBoost**.
- 🖼️ Supports image upload and live preview of detections.
- 🧠 Utilizes trained ML models (`best10.pt`, `area_model.pkl`).

---

## 🧰 Tech Stack

- **Back-end**: Python, Flask  
- **ML/DL**: YOLOv8 (`ultralytics`), XGBoost, pandas  
- **Utilities**: geopy, geocoder, requests, datetime, pickle  
- **Front-end**: HTML (Jinja2 templates)  
- **APIs**: OpenWeatherMap  

---

## 📁 Folder Structure


---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/fire-detection-app.git
cd fire-detection-app


python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate



pip install -r requirements.txt


api_key = '*********************************'

python app.py

http://localhost:5000
