🐘 Elephant Conflict Early Warning System (EWS)

This project is a full-stack application that predicts human-elephant conflict risk in Sri Lanka. It uses a machine learning model, live weather forecast data, and sends real-time SMS alerts to warn communities of high-risk periods.

✨ Core Features (Work Complete: Day 1-5)

ML Risk Prediction: Trains a Random Forest model on historical and environmental data to predict "High," "Medium," or "Low" conflict risk. (Day 2)

5-Day Risk Forecast: Fetches a 5-day weather forecast from OpenWeatherMap and runs it through the ML model to predict risk for the coming days. (Day 4)

Interactive Dashboard: A multi-page Streamlit web app for visualizing data and interacting with the system. (Day 2)

Live Risk Map: A Folium map showing the latest risk predictions for all monitored locations. (Day 2)

Analytics Dashboard: A page with interactive Plotly charts showing historical conflict data by type, location, and time. (Day 2)

Real-Time SMS Alerts: Automatically sends a real SMS alert via Twilio to a verified phone number when a "High" risk event is predicted. (Day 3)

Farmer Sighting Reports: A dashboard tab allowing users to submit their own elephant sighting reports, which are saved directly to the database. (Day 3)

Secure API Backend: The FastAPI backend is secured with an API key. Only authenticated requests (like from the dashboard) are allowed to access the data or trigger predictions. (Day 5)

💻 Tech Stack

Backend: FastAPI, Uvicorn, SQLAlchemy

Frontend: Streamlit

Data Science: Pandas, Scikit-learn, Plotly, Folium

Database: SQLite

External APIs: Twilio (for SMS), OpenWeatherMap (for weather)

Security: python-dotenv, FastAPI APIKeyHeader

📁 Project Structure

elephant-conflict-alert/
│
├── app/
│   ├── __init__.py
│   ├── database.py         # Database connection
│   ├── data_loader.py      # Creates synthetic data
│   ├── dependencies.py     # API key security
│   ├── locations.py        # Location coordinates
│   ├── ml_predictor.py     # The ML model logic
│   ├── models.py           # SQLAlchemy database tables
│   ├── notifications.py    # Twilio SMS/email logic
│   ├── schemas.py          # Pydantic data validation
│   └── weather_fetcher.py  # OpenWeatherMap API logic
│
├── venv/                   # Your local virtual environment
│
├── .env                    # <-- All your secret API keys
├── .gitignore
├── dashboard.py            # The Streamlit dashboard UI
├── elephant_conflict.db    # Your local SQLite database
├── main.py                 # The FastAPI application
├── README.md               # This file
├── requirements.txt        # Python libraries
├── run_day1.py             # Script to initialize the database
└── run_day2.py             # Script to train the ML model


🚀 How to Run This Project

Follow these steps to set up and run the application on your local machine.

1. API Key Setup (.env file)

This project requires 3 external API keys to function.

Create a file named .env in the root folder (elephant-conflict-alert/).

Paste the following content into it, filling in your own secret keys.

# .env file

# 1. Twilio (for SMS Alerts)
# Get from twilio.com
TWILIO_ACCOUNT_SID="YOUR_LIVE_ACCOUNT_SID"
TWILIO_AUTH_TOKEN="YOUR_LIVE_AUTH_TOKEN"
TWILIO_PHONE_NUMBER="YOUR_TWILIO_TRIAL_PHONE_NUMBER"
YOUR_MOBILE_NUMBER="YOUR_PERSONAL_VERIFIED_PHONE_NUMBER_WITH_+COUNTRY_CODE"

# 2. OpenWeatherMap (for 5-Day Forecast)
# Get from openweathermap.org
OWM_API_KEY="YOUR_OPENWEATHERMAP_API_KEY"

# 3. App Security (This can be any secret password you want)
APP_API_KEY="my-super-secret-key-12345"


2. Installation

Create a virtual environment:

python -m venv venv


Activate it (in PowerShell):

.\venv\Scripts\Activate.ps1


Install all libraries:

pip install -r requirements.txt


Initialize the Database: Run the Day 1 script to create your elephant_conflict.db file and fill it with data.

python run_day1.py


3. Run the Application

This application requires two terminals to run at the same time.

In your FIRST terminal (PowerShell):

Activate the venv: .\venv\Scripts\Activate.ps1

Start the FastAPI server:

uvicorn main:app --reload


Your API is now running at http://localhost:8000

In your SECOND terminal (PowerShell):

Activate the venv: .\venv\Scripts\Activate.ps1

Start the Streamlit dashboard:

streamlit run dashboard.py


Your dashboard will automatically open in your browser at http://localhost:8501