import mlflow
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    ConfusionMatrixDisplay,
    classification_report
)

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
mlflow.sklearn.autolog(disable=True)

# Функция удаления выбросов
def remove_outliers_iqr(data, column, verbose=False):
    """
    Удаляет выбросы по методу IQR (межквартильный размах)
    
    Параметры:
        data: pd.DataFrame - исходные данные
        column: str - название колонки для обработки
        verbose: bool - вывод информации о процессе
        
    Возвращает:
        pd.DataFrame - данные без выбросов
    """
    if verbose:
        print(f"Размер датасета до удаления выбросов: {data.shape}")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)].copy()
    
    if verbose:
        print(f"Размер датасета после удаления выбросов: {filtered_data.shape}")
        print(f"Удалено {len(data) - len(filtered_data)} выбросов")
    
    return filtered_data






# Функция разделения данных на train val test
def split_and_scale(
    df,
    target_col,
    feature_cols,
    num_cols=None,
    test_size=0.3,
    val_size=0.5,
    random_state=42,
    verbose=True
):
    """
    Разделяет данные на train/val/test и масштабирует числовые признаки (если заданы).
    Возвращает pandas.DataFrame-ы, сохраняет названия колонок.

    Parameters:
    - df: pandas.DataFrame — исходные данные
    - target_col: str — целевой столбец
    - feature_cols: list — список признаков
    - num_cols: list — числовые признаки для масштабирования (можно None)
    - test_size: float — доля данных на val+test (по умолчанию 0.3)
    - val_size: float — доля от test_size, которая пойдет в test (по умолчанию 0.5)
    - random_state: int — для воспроизводимости
    - verbose: bool — логировать размеры выборок

    Returns:
    - X_train, X_val, X_test, y_train, y_val, y_test — pandas.DataFrame 
    - scaler — MinMaxScaler 
    """

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Делим на train и временную выборку
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Делим временную выборку на val и test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size, stratify=y_temp, random_state=random_state
    )

    scaler = None
    if num_cols:
        scaler = MinMaxScaler()
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_val[num_cols] = scaler.transform(X_val[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])

    if verbose:
        print(f"Размер выборок:")
        print(f"  Train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"  Val:   {X_val.shape}, y_val:   {y_val.shape}")
        print(f"  Test:  {X_test.shape}, y_test:  {y_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


# Проверяем распределение классов
def check_distribution(y, name):
    """
    Улучшенная версия с визуализацией.
    """
    print(f"\n{name} class distribution:")
    dist = y.value_counts(normalize=True)
    print(dist)
    
    # Визуализация
    plt.figure(figsize=(6, 3))
    sns.barplot(x=dist.index, y=dist.values)
    plt.title(f'{name} Class Distribution')
    plt.ylabel('Percentage')
    plt.xlabel('Class')
    plt.show()