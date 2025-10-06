"""
Модуль для предсказания солнечных пятен с использованием модели NARX.

Этот скрипт загружает данные о солнечных пятнах, подготавливает их для модели,
выполняет предсказания на 18 месяцев вперед и сохраняет результаты.
"""

import argparse
import pandas as pd
import torch
import numpy as np
import os
from typing import Tuple
from scripts.data_x18 import Datax18
from scripts.model_x18 import NARXx18
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def _prepare_data(ssn_url: str, theoretical_path: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Подготавливает данные для модели: загружает и нормализует ряды солнечных пятен.
    
    Args:
        ssn_url: URL для загрузки данных о солнечных пятнах
        theoretical_path: Путь к файлу с теоретическими данными
        
    Returns:
        Tuple содержащий:
        - theoretical_series: нормализованный теоретический ряд
        - observed_series: нормализованный наблюдаемый ряд  
        - normalization_factor: коэффициент нормализации для восстановления исходных значений
    """
    df_theoretical = pd.read_csv(theoretical_path, sep=';')
    
    df_ssn = pd.read_csv(
        ssn_url,
        sep=';',
        header=None,
        names=['Year', 'Month', 'Decimal Date', 'SSN', 'Std Dev', 'Obs', 'Def'],
        usecols=[2, 3]
    )

    min_date = df_theoretical['decimal_date'].iloc[0]
    mask_valid = (df_ssn['Decimal Date'] >= min_date) & (df_ssn['SSN'].shift(1) != -1)
    df_ssn = df_ssn.loc[mask_valid].reset_index(drop=True)

    theoretical_series = df_theoretical['T1'].to_numpy(dtype=np.float32)
    observed_series = df_ssn['SSN'].to_numpy(dtype=np.float32)
    
    normalization_factor = observed_series.max()

    observed_series /= normalization_factor
    theoretical_series /= normalization_factor

    return theoretical_series, observed_series, normalization_factor

def predict(
    device: torch.device, 
    path_to_models: str, 
    theoretical_series: np.ndarray, 
    observed_series: np.ndarray, 
    normalization_factor: float, 
    depth: int
) -> np.ndarray:
    """
    Выполняет предсказания солнечных пятен на 18 месяцев вперед используя обученные модели NARX.
    
    Args:
        device: Устройство для вычислений (CPU/GPU)
        path_to_models: Путь к директории с весами моделей
        theoretical_series: Нормализованный теоретический ряд
        observed_series: Нормализованный наблюдаемый ряд
        normalization_factor: Коэффициент нормализации для восстановления исходных значений
        depth: Глубина предсказания
        
    Returns:
        Массив предсказаний формы (N, 18), где N - количество образцов,
        18 - количество горизонтов предсказания
    """
    prediction_horizons = range(1, 19)
    
    loaded_models = []
    data_objects = []
    test_indices = []
    
    for horizon in prediction_horizons:
        model = NARXx18(
            input_size=2 * 4 + horizon,
            hidden_sizes=[24],
            output_size=1,
            M=normalization_factor
        )
        
        model_path = os.path.join(path_to_models, f'model_weights_horizon{horizon}.pth')
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        loaded_models.append(model)
        
        data = Datax18(theoretical_series, observed_series, previous_values=4, horizon=horizon)
        data_objects.append(data)
        
        test_start_index = 722 + 1 - (horizon - 1) + 7 - 4 + 3 - depth
        test_indices.append(test_start_index)

    horizon_predictions = []
    for model, data_obj, start_idx in zip(loaded_models, data_objects, test_indices):
        test_input = data_obj.X[start_idx:]
        test_input = test_input.to(device)
        
        with torch.no_grad():
            prediction = model(test_input).squeeze()
            prediction = prediction.cpu().numpy() * normalization_factor
        
        horizon_predictions.append(prediction)
    
    if depth == 1:
        predictions_array = np.array([horizon_predictions])
    else:
        predictions_array = np.stack(horizon_predictions, axis=1)
    return predictions_array

def save_predictions(
    predictions: np.ndarray, 
    path_to_save: str, 
    depth: int, 
    observed_series_length: int
) -> None:
    """
    Сохраняет предсказания в CSV файлы с соответствующими датами.
    
    Args:
        predictions: Массив предсказаний формы (N, 18)
        path_to_save: Директория для сохранения файлов
        depth: Глубина предсказания
        observed_series_length: Длина наблюдаемого ряда
    """
    os.makedirs(path_to_save, exist_ok=True)
    
    df_theoretical = pd.read_csv('data/theoretical_series.csv', sep=';')
    
    for prediction_idx, prediction_values in enumerate(predictions):
        start_idx = observed_series_length - depth + prediction_idx
        end_idx = start_idx + 18
        
        prediction_decimal_dates = df_theoretical['decimal_date'][start_idx:end_idx].values
        
        years = np.floor(prediction_decimal_dates).astype(int)
        months = np.round((prediction_decimal_dates - years) * 12 + 1).astype(int)
        
        months[months == 13] = 12
        
        output_df = pd.DataFrame({
            'Year': (years % 2000).astype(int),
            'Month': months.astype(int),
            'Value': prediction_values
        })
        
        current_year = years[0] % 2000
        current_month = months[0]
        
        shifted_month = current_month + 6
        shifted_year = current_year
        
        if shifted_month > 12:
            shifted_month -= 12
            shifted_year += 1
        
        filename = f"predict_{shifted_year}_{shifted_month:02d}.csv"
        filepath = os.path.join(path_to_save, filename)
        output_df.to_csv(filepath, index=False, header=False)

def _decimal_to_datetime(decimal_date: float) -> datetime.datetime:
    """
    Преобразует десятичную дату в объект datetime.
    
    Args:
        decimal_date: Десятичная дата (например, 2024.5 для июня 2024)
        
    Returns:
        Объект datetime с первым днем соответствующего месяца
    """
    year = int(decimal_date)
    fractional_part = decimal_date - year
    month = int(fractional_part * 12) + 1
    
    if month > 12:
        year += 1
        month = 1
    
    return datetime.datetime(year, month, 1)

def plot_predictions(
    observed_series: np.ndarray,
    theoretical_series: np.ndarray, 
    normalization_factor: float,
    depth: int,
    predictions: np.ndarray,
    ssn_url: str,
    path_to_save: str
) -> None:
    """
    Создает график с наблюдаемыми данными, теоретическими значениями и предсказаниями.
    
    Args:
        observed_series: Нормализованный наблюдаемый ряд солнечных пятен
        theoretical_series: Нормализованный теоретический ряд
        normalization_factor: Коэффициент нормализации для восстановления исходных значений
        depth: Глубина предсказания
        predictions: Массив предсказаний формы (N, 18)
        ssn_url: URL для загрузки данных о солнечных пятнах
        path_to_save: Директория для сохранения графика
    """
    plt.figure(figsize=(15, 7))
    
    observed_length = len(observed_series)
    
    df_theoretical = pd.read_csv('data/theoretical_series.csv', sep=';')
    df_ssn = pd.read_csv(
        ssn_url,
        sep=';',
        header=None,
        names=['Year', 'Month', 'Decimal Date', 'SSN', 'Std Dev', 'Obs', 'Def'],
    )

    min_date = df_theoretical['decimal_date'].iloc[0]
    mask_valid = (df_ssn['Decimal Date'] >= min_date) & (df_ssn['SSN'].shift(1) != -1)
    df_ssn = df_ssn.loc[mask_valid].reset_index(drop=True)

    theoretical_x = [
        _decimal_to_datetime(dd) 
        for dd in df_theoretical['decimal_date'][observed_length - depth:observed_length + 18 - 1]
    ]
    observed_x = [
        _decimal_to_datetime(dd) 
        for dd in df_ssn['Decimal Date'][-depth:-1]
    ]

    plt.plot(
        theoretical_x,
        theoretical_series[observed_length - depth:observed_length + 18 - 1] * normalization_factor,
        label='Theoretical',
        marker='s',
        linestyle='--',
        color='black'
    )

    plt.plot(
        observed_x,
        observed_series[-depth:-1] * normalization_factor,
        label='Observed',
        marker='o',
        markersize=10,
        color='black'
    )

    for prediction_idx, prediction_values in enumerate(predictions):
        pred_x = [
            _decimal_to_datetime(dd) 
            for dd in df_theoretical['decimal_date'][
                observed_length - depth + prediction_idx:observed_length - depth + prediction_idx + 18
            ]
        ]
        
        plt.plot(
            pred_x,
            prediction_values,
            label=f'Prediction {prediction_idx + 1}',
            linestyle='--',
            marker='o',
            alpha=0.5
        )
        
        plt.scatter(
            pred_x[0],
            prediction_values[0],
            s=100,
            color=plt.gca().lines[-1].get_color(),
            edgecolor='black',
            zorder=5
        )

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
        
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_filename = f'plot_depth_{depth}_{timestamp}.png'
    plot_path = os.path.join(path_to_save, plot_filename)
    plt.savefig(plot_path)
    plt.show()


def main(
    device: torch.device, 
    path_to_save: str, 
    depth: int, 
    need_plot: bool
) -> None:
    """
    Основная функция для выполнения предсказаний солнечных пятен.
    
    Args:
        device: Устройство для вычислений (CPU/GPU)
        path_to_save: Директория для сохранения результатов
        depth: Глубина предсказания
        need_plot: Флаг для создания графика предсказаний
    """
    ssn_data_url = 'https://www.sidc.be/SILSO/DATA/SN_ms_tot_V2.0.csv'
    theoretical_data_path = 'data/theoretical_series.csv'
    models_directory = 'data/model_weights/'
    
    print("Загружаем и подготавливаем данные...")
    theoretical_series, observed_series, normalization_factor = _prepare_data(
        ssn_data_url, theoretical_data_path
    )
    
    print("Выполняем предсказания...")
    predictions = predict(
        device=device,
        path_to_models=models_directory,
        theoretical_series=theoretical_series,
        observed_series=observed_series,
        normalization_factor=normalization_factor,
        depth=depth
    )
    
    print(f"Предсказания выполнены.")
    
    save_predictions(
        predictions=predictions,
        path_to_save=path_to_save,
        depth=depth,
        observed_series_length=len(observed_series)
    )
    
    print(f"Результаты сохранены в директории: {path_to_save}")
    
    if need_plot:
        plot_predictions(
            observed_series=observed_series,
            theoretical_series=theoretical_series,
            normalization_factor=normalization_factor,
            depth=depth,
            predictions=predictions,
            ssn_url=ssn_data_url,
            path_to_save=path_to_save
        )
        print("График создан и сохранен")
    
    print("Предсказания завершены успешно!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Предсказание солнечных пятен с использованием модели NARX',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python predict.py                                    
  python predict.py --depth 6 --need_plot            
  python predict.py --device cuda --path_to_save results  
        """
    )
    
    parser.add_argument(
        '--path_to_save', 
        type=str, 
        default='predictions', 
        help='Директория для сохранения предсказаний (по умолчанию: predictions)'
    )
    parser.add_argument(
        '--depth', 
        type=int, 
        default=4, 
        help='Глубина предсказания - количество последних точек для анализа (по умолчанию: 4)'
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default=None, 
        choices=['cpu', 'cuda', 'mps'],
        help='Устройство для вычислений: "cpu", "cuda", или "mps" (по умолчанию: автоопределение)'
    )
    parser.add_argument(
        '--need_plot', 
        action='store_true', 
        help='Создать график с предсказаниями'
    )
    
    args = parser.parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    else:
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    print(f"Используемое устройство: {device}")
    print(f"Глубина предсказания: {args.depth}")
    print(f"Директория сохранения: {args.path_to_save}")
    print(f"Создание графика: {'Да' if args.need_plot else 'Нет'}")
    print("-" * 50)

    main(device, args.path_to_save, args.depth, args.need_plot)