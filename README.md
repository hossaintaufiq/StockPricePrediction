# Stock Price Prediction Using Market Data and News Sentiment

This project explores a data-driven approach to forecasting stock prices by combining historical market data with sentiment analysis of Indian news headlines. The workflow is implemented in a Jupyter Notebook and includes data collection, preprocessing, exploratory analysis, feature engineering, and model training.

## Overview

The objective of this project is to predict stock market movement using:

- Historical stock price data from Yahoo Finance
- News headlines from the provided Indian news dataset
- Sentiment scores such as polarity and subjectivity extracted from text
- Traditional time-series forecasting and deep learning models

The notebook demonstrates a complete end-to-end pipeline for stock price prediction and evaluates multiple models for forecasting performance.

## Project Goals

- Collect and preprocess financial and news data
- Analyze trends, volatility, and stationarity in stock prices
- Extract sentiment from news headlines
- Merge stock and news features into a unified dataset
- Train and compare forecasting models for stock price prediction

## Project Structure

- Stock_Price_Prediction.ipynb: Main notebook containing the full workflow
- stock_data.csv: Historical stock market data downloaded from Yahoo Finance
- india-news-headlines.csv: Raw Indian news headlines dataset
- news_merged.csv: Cleaned and processed news sentiment data
- stock_news.csv: Final merged dataset combining stock data and sentiment features

## Technologies Used

- Python
- pandas
- NumPy
- Matplotlib
- Seaborn
- yfinance
- requests
- statsmodels
- scikit-learn
- TextBlob
- TensorFlow / Keras
- Jupyter Notebook

## Workflow

1. Data Collection
   - Historical market data is fetched using Yahoo Finance for the BSE market index.
   - News data is loaded from the provided CSV file.

2. Exploratory Data Analysis
   - Basic statistics and data inspection are performed.
   - Visualizations include closing price trends, rolling averages, and percentage changes.
   - Stationarity tests are used to evaluate the time-series behavior.

3. Sentiment Analysis
   - News headlines are processed using TextBlob.
   - Polarity and subjectivity scores are computed for each headline.
   - Sentiment values are aggregated by date.

4. Data Integration
   - Stock market data and news sentiment scores are merged by date.
   - The merged dataset is stored for modeling.

5. Modeling
   - ARIMA is used as a baseline time-series forecasting model.
   - An LSTM-based sequence-to-sequence model is built for multi-step forecasting.
   - Predictions are evaluated using error metrics such as MAE and RMSE.

## Installation

1. Create and activate a Python virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install the required dependencies:

   ```bash
   pip install pandas numpy matplotlib seaborn yfinance requests statsmodels scikit-learn textblob tensorflow jupyter
   ```

## Running the Project

1. Open the project folder in VS Code or Jupyter Notebook.
2. Launch Jupyter Notebook or VS Code with the Jupyter extension.
3. Open Stock_Price_Prediction.ipynb.
4. Run the notebook cells sequentially.

## Notes

- The notebook is designed for experimentation and educational purposes.
- Model performance may vary depending on hardware, data availability, and hyperparameter tuning.
- Stock market forecasting is inherently uncertain, and the results should not be treated as financial advice.

## Future Improvements

- Add more advanced feature engineering such as technical indicators
- Experiment with transformer-based models
- Compare multiple architectures and hyperparameter settings
- Build a reusable Python script for training and prediction
- Deploy the model as a simple web application or API

## License

No license file is currently included in the repository. If you plan to share or distribute this project publicly, consider adding an appropriate open-source license.
