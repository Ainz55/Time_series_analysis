import csv
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX


def load_data(filename):
    """Загрузка данных из CSV файла"""
    dates = []
    values = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            dates.append(datetime.strptime(row[0], "%Y-%m"))
            values.append(int(row[1]))
    return dates, values


def generate_synthetic_feature(values):
    """Генерация синтетического признака (например, 'количество рейсов')"""
    return [val * 0.8 + np.random.normal(0, 10) for val in values]


def check_stationarity(data):
    """Проверка стационарности временного ряда"""
    result = adfuller(data)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')
    return result[1] < 0.05


def decompose_time_series(dates, values, period=12):
    """Декомпозиция временного ряда"""
    decomposition = seasonal_decompose(values, model='additive',
                                       period=period)
    fig = decomposition.plot()
    fig.set_size_inches(12, 8)
    plt.tight_layout()
    return decomposition


# === 3. Методы прогнозирования ===
# def simple_linear_forecast(data, months_ahead):
#     """Улучшенный линейный прогноз с учетом тренда"""
#     # Используем все данные для построения модели, а не только последние 12 точек
#     x = np.arange(len(data))
#     y = np.array(data)
#
#     # Линейная регрессия через все точки
#     A = np.vstack([x, np.ones(len(x))]).T
#     a, b = np.linalg.lstsq(A, y, rcond=None)[0]
#
#     # Прогноз на будущие периоды
#     future_x = np.arange(len(data), len(data) + months_ahead)
#     return a * future_x + b


def arima_forecast(data, months_ahead, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    """Прогноз с помощью ARIMA с учетом сезонности"""
    try:
        model = SARIMAX(data,
                       order=order,
                       seasonal_order=seasonal_order,
                       enforce_stationarity=False,
                       enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=months_ahead)
        return forecast
    except Exception as e:
        print(f"Ошибка в ARIMA прогнозе: {e}")
        return [np.mean(data)] * months_ahead


def exponential_smoothing_forecast(data, months_ahead, seasonal_periods=12):
    """Прогноз с помощью экспоненциального сглаживания"""
    model = ExponentialSmoothing(data, seasonal='additive',
                                 seasonal_periods=seasonal_periods)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=months_ahead)
    return forecast


def compare_regression_vs_time_series(dates, values):
    """Сравнение прогноза через регрессию и временной ряд"""
    flights = generate_synthetic_feature(values)
    data = pd.DataFrame({
        'Date': dates,
        'Passengers': values,
        'Flights': flights
    })

    X = data[['Flights']]
    y = data['Passengers']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        shuffle=False)

    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)
    y_pred_reg = reg_model.predict(X_test)

    train_size = len(X_train)
    arima_fc = arima_forecast(values[:train_size], len(X_test))

    print("\n=== Сравнение методов ===")
    print("MAE регрессии:", mean_absolute_error(y_test, y_pred_reg))
    print("MAE ARIMA:", mean_absolute_error(y_test, arima_fc))

    # Визуализация
    plt.figure(figsize=(13, 6))
    plt.plot(data['Date'], data['Passengers'], label='Исходные данные',
             color='#1f77b4')
    plt.scatter(data['Date'][X_test.index], y_test, color='red',
                label='Тестовые данные', s=20)
    plt.plot(data['Date'][X_test.index], y_pred_reg,
             label='Прогноз (регрессия)', linestyle='--', color='#ff7f0e')
    plt.plot(data['Date'][X_test.index], arima_fc, label='Прогноз (ARIMA)',
             linestyle='--', color='#2ca02c')
    plt.title('Сравнение регрессии и временного ряда')
    plt.xlabel('Дата')
    plt.ylabel('Пассажиропоток')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


# === 5. Основная функция ===
def main():
    dates, values = load_data("airline-passengers.csv")

    print("\nАнализ стационарности:")
    is_stationary = check_stationarity(values)
    print(f"Ряд {'стационарный' if is_stationary else 'не стационарный'}")

    print("\nДекомпозиция временного ряда:")
    decompose_time_series(dates, values)

    future_dates = [
        datetime(dates[-1].year + (dates[-1].month + i - 1) // 12,
                 (dates[-1].month + i - 1) % 12 + 1, 1)
        for i in range(1, 13)]

    # linear_fc = simple_linear_forecast(values, 12)
    arima_fc = arima_forecast(values, 12)
    es_fc = exponential_smoothing_forecast(values, 12)

    # Визуализация прогнозов
    plt.figure(figsize=(13, 6))
    plt.plot(dates, values, label="Исторические данные", color="#1f77b4")
    # plt.plot(future_dates, linear_fc, label="Линейный прогноз",
    #          linestyle='--', color="#ff7f0e")
    plt.plot(future_dates, arima_fc, label="ARIMA", linestyle='--',
             color="#2ca02c")
    plt.plot(future_dates, es_fc, label="Эксп. сглаживание", linestyle='--',
             color="#d62728")
    plt.title("Сравнение методов прогнозирования")
    plt.xlabel("Дата")
    plt.ylabel("Пассажиропоток")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    compare_regression_vs_time_series(dates, values)


if __name__ == "__main__":
    main()
