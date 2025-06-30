import numpy as np
import pandas as pd
from src.outliers import remove_outliers_bulk

def final_cleaning_step(data: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    data = data.copy()

    # Удаление ненужных колонок
    cols_to_drop_initial = [
        'ID', 'Year_Birth', 'Education', 'Marital_Status', 'Kidhome', 'Teenhome',
        'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
        'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
        'NumStorePurchases', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5',
        'AcceptedCmp1', 'AcceptedCmp2', 'Z_CostContact', 'Z_Revenue',
        'Dt_Customer', 'Customer_Since_Days', 'Customer_Since_Years', 'Complain'
    ]
    data.drop(columns=cols_to_drop_initial, inplace=True, errors='ignore')

    # Удаление дубликатов
    data.drop_duplicates(inplace=True)

    # Удаление выбросов
    data = remove_outliers_bulk(data, ['Age', 'TotalSpent', 'Income'], verbose)

    # Группировка по доходу
    data['IncomeGroup'] = pd.qcut(data['Income'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    data['IncomeInterval'] = pd.qcut(data['Income'], q=4)

    if verbose:
        response_by_income = data.groupby('IncomeInterval').agg(
            Средний_Response=('Response', 'mean'),
            Количество_клиентов=('Response', 'count')
        ).reset_index()
        response_by_income['Диапазон_дохода'] = response_by_income['IncomeInterval'].astype(str)
        print(response_by_income[['Диапазон_дохода', 'Средний_Response', 'Количество_клиентов']])

    # Признаки
    data['IsHighIncome'] = (data['IncomeGroup'] == 'Q4').astype(int)
    
    # Логарифмирование признаков
    data['IncomeLog'] = np.log1p(data['Income'])
    data['TotalSpentLog'] = np.log1p(data['TotalSpent'])

    # Удаление выбросов по другим фичам
    data = remove_outliers_bulk(data, ['TotalPurchaseActivity', 'NumWebVisitsMonth'], verbose)

    # Признак отклика на любую кампанию
    data['AcceptedAnyCmp'] = (data['TotalAcceptedCmp'] > 0).astype(int)

    # Финальное удаление признаков
    cols_to_drop_final = [
        'TotalAcceptedCmp', 'RecencyGroup', 'IncomeInterval',
        'IncomeGroup', 'TotalPurchaseActivity', 'Income', 'TotalSpent'
    ]
    data.drop(columns=cols_to_drop_final, inplace=True, errors='ignore')

    return data