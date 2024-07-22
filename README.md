# ALAD2
Bitcoin Price Prediction with Real-Time Dashboard
An advanced real-time dashboard application for predicting Bitcoin prices and technical indicators using state-of-the-art machine learning and deep learning models.

Features
Real-Time Data Display
Fetches real-time Bitcoin market data from Binance API.
Displays candlestick charts for Bitcoin prices.
Technical Indicator Calculation
Computes various technical indicators: SMA, EMA, RSI, MACD, ADX, ATR, CCI, OBV, MFI, and Stochastics.
Integrates these indicators for enhanced decision-making.
Price and Indicator Prediction
Utilizes multiple machine learning and deep learning models for predicting future prices and indicators:
Linear Regression
Random Forest
LSTM (Long Short-Term Memory)
DNN (Deep Neural Network)
Trains models on historical data and saves trained models for future use.
Ensemble Learning
Combines predictions from multiple models to enhance robustness and accuracy.
Feature Engineering
Incorporates multiple features to improve prediction quality.
Handles missing values with advanced filling techniques.
Market Sentiment Data
Integrates market sentiment data from Alternative.me API (Fear & Greed Index).
User Interface
Built with tkinter for an interactive user interface.
Displays predictions and indicators on matplotlib charts.
Auto-updates every 60 seconds to reflect new market data.
Installation
Prerequisites
Python 3.x
Required Python packages: ccxt, pandas, numpy, ta, requests, matplotlib, tensorflow, scikit-learn, joblib, tkinter
Setup
Clone the repository:

sh
Copier le code
git clone https://github.com/Undertheworld-hub/ALAD2.git
cd bitcoin-price-prediction
Install required packages:

sh
Copier le code
pip install -r requirements.txt
Run the application:

sh
Copier le code
python main.py
Usage
Real-Time Data Display: The application fetches and displays real-time Bitcoin market data using candlestick charts.
Technical Indicator Calculation: It calculates and integrates various technical indicators to assist in market analysis.
Price and Indicator Prediction: The application uses advanced ML and DL models to predict future prices and indicators.
Market Sentiment Integration: It fetches and displays market sentiment data to provide context to price movements.
User Interaction: The interface updates automatically every 60 seconds to provide the latest market data and predictions.
Project Structure
plaintext
Copier le code
bitcoin-price-prediction/
│
├── main.py                  # Main application file
├── requirements.txt         # Python package requirements
├── models/
│   ├── lstm_model.h5        # Pre-trained LSTM model
│   ├── dnn_model.h5         # Pre-trained DNN model
│   ├── lr_model.joblib      # Pre-trained Linear Regression model
│   ├── rf_model.joblib      # Pre-trained Random Forest model
│
└── README.md                # Project description and setup instructions
Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

Fork the Project
Create your Feature Branch (git checkout -b feature/AmazingFeature)
Commit your Changes (git commit -m 'Add some AmazingFeature')
Push to the Branch (git push origin feature/AmazingFeature)
Open a Pull Request
License
Distributed under the MIT License. See LICENSE for more information.

Contact
Your Name - your.email@example.com

Project Link: https://github.com/yourusername/bitcoin-price-prediction
