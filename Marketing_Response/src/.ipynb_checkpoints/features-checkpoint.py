import pandas as pd

def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    # 1. Возраст
    data['Age'] = 2025 - data['Year_Birth']

    # 2. Наличие детей
    data['HasChildren'] = (data[['Kidhome', 'Teenhome']].sum(axis=1) > 0).astype(int)

    # 3. Семейное положение
    data['MaritalFlag'] = data['Marital_Status'].isin(['Married', 'Together']).astype(int)

    # 4. Общая сумма трат
    mnt_cols = [
        'MntWines', 'MntFruits', 'MntMeatProducts', 
        'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'
    ]
    data['TotalSpent'] = data[mnt_cols].sum(axis=1)

    # 5. Дата регистрации
    data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], dayfirst=True)
    data['Customer_Since_Days'] = (pd.Timestamp.today() - data['Dt_Customer']).dt.days
    data['Customer_Since_Years'] = (data['Customer_Since_Days'] / 365.25).astype(int)

    # 6. Кол-во принятых кампаний
    cmp_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
    data['TotalAcceptedCmp'] = data[cmp_cols].sum(axis=1)

    # 7. Высшее образование
    higher_edu = ['Graduation', 'Master', 'PhD']
    data['HigherEducation'] = data['Education'].isin(higher_edu).astype(int)

    # 8. Удаление лишних колонок
    drop_cols = [
        'ID', 'Year_Birth', 'Education', 'Marital_Status', 'Kidhome', 'Teenhome', 
        'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 
        'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 
        'NumStorePurchases', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 
        'AcceptedCmp1', 'AcceptedCmp2', 'Z_CostContact', 'Z_Revenue', 
        'Dt_Customer', 'Customer_Since_Days', 'Customer_Since_Years', 'Complain'
    ]
    data.drop(columns=drop_cols, inplace=True, errors='ignore')

    # 9. Удаление дубликатов
    data.drop_duplicates(inplace=True)

    return data