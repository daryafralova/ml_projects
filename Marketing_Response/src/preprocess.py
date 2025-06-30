import pandas as pd

# Функция обработки nan значений
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Заполняем пропуски в Income по медиане в Education
    df['Income'] = df.groupby('Education')['Income'].transform(
        lambda x: x.fillna(x.median())
    )

    # Удаляем строки с пустым доходом
    df = df.dropna(subset=['Income'])

    return df

# Функция удаления выбросов
def remove_outliers_multiple(data: pd.DataFrame, columns: list, verbose=False) -> pd.DataFrame:
    df = data.copy()
    for col in columns:
        df = remove_outliers_iqr(df, column=col, verbose=verbose)
    return df