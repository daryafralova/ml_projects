import pandas as pd

def remove_outliers_iqr(data, column, verbose=False):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    filtered = data[(data[column] >= lower) & (data[column] <= upper)]
    
    if verbose:
        print(f"{column}: удалено {len(data) - len(filtered)} строк.")
    
    return filtered

def remove_outliers_bulk(data, columns, verbose=False):
    for col in columns:
        data = remove_outliers_iqr(data, col, verbose)
    return data