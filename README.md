# Time Series Forecasting: Airline Passenger Traffic Analysis

This project demonstrates time series analysis and forecasting methods using airline passenger traffic data as an example. It includes data loading, stationarity checks, time series decomposition, and comparison of different forecasting approaches.

## Features

- **Data Loading**: Reads CSV files containing dates and passenger traffic values
- **Synthetic Feature Generation**: Creates additional features (e.g., "number of flights") for analysis
- **Stationarity Check**: Uses the Dickey-Fuller test to determine time series stationarity
- **Time Series Decomposition**: Breaks down the series into trend, seasonality, and residuals
- **Forecasting**:
  - **ARIMA/SARIMA**: Forecasting using autoregression, integration, and moving average, including seasonality components
  - **Exponential Smoothing**: Forecasting considering trend and seasonality
- **Method Comparison**: Compares accuracy between regression-based and time series forecasting approaches

## Dependencies

- `pandas` and `numpy` for data processing
- `matplotlib` for visualization
- `statsmodels` for time series analysis (ADF test, decomposition, ARIMA, exponential smoothing)
- `scikit-learn` for linear regression and error metrics

## Usage

1. Ensure all required libraries are installed:
   ```
   pip install pandas numpy matplotlib statsmodels scikit-learn
   ```
___

## Example Output
<img width="1300" height="673" alt="image" src="https://github.com/user-attachments/assets/b9751d2e-fbc4-4d43-8941-c090f7849a52" />
<img width="1203" height="874" alt="image" src="https://github.com/user-attachments/assets/4ae63501-df50-44c8-9c62-fbeff2296f43" />
<img width="1301" height="672" alt="image" src="https://github.com/user-attachments/assets/a0f6ae49-3324-4190-aa5e-513076e16427" />



