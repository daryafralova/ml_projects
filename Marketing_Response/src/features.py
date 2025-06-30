import pandas as pd

def add_basic_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    # Возраст
    data['Age'] = 2025 - data['Year_Birth']

    # HasChildren
    data['HasChildren'] = (data[['Kidhome', 'Teenhome']].sum(axis=1) > 0).astype(int)

    # MaritalFlag
    data['MaritalFlag'] = data['Marital_Status'].isin(['Married', 'Together']).astype(int)

    # TotalSpent
    mnt_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    data['TotalSpent'] = data[mnt_cols].sum(axis=1)
    

    # TotalPurchaseActivity
    purchase_cols = ['NumCatalogPurchases', 'NumWebPurchases', 'NumStorePurchases', 'NumDealsPurchases']
    data['TotalPurchaseActivity'] = data[purchase_cols].sum(axis=1)

    # TotalAcceptedCmp
    cmp_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
    data['TotalAcceptedCmp'] = data[cmp_cols].sum(axis=1)

    # HigherEducation
    higher_edu = ['Graduation', 'Master', 'PhD']
    data['HigherEducation'] = data['Education'].isin(higher_edu).astype(int)

    return data

