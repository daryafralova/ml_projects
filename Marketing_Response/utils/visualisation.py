import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def compare_all_numerical_features(numerical_features, X_train, X_val, X_test):
    """
    Сравнивает распределения числовых признаков в train, val и test наборах данных.
    
    Параметры:
    ----------
    numerical_features : list
        Список числовых признаков для сравнения
    X_train : pd.DataFrame
        Обучающая выборка
    X_val : pd.DataFrame 
        Валидационная выборка
    X_test : pd.DataFrame
        Тестовая выборка
    
    Возвращает:
    ----------
    None (отображает графики распределений)
    """
    # Создаем сетку графиков
    n_features = len(numerical_features)
    fig, axes = plt.subplots(1, n_features, figsize=(20, 4))
    
    # Обработка случая с одним признаком
    if n_features == 1:
        axes = [axes]
    
    # Строим распределения для каждого признака
    for ax, feature in zip(axes, numerical_features):
        # KDE-графики для каждой выборки
        sns.kdeplot(X_train[feature], label='Train', fill=True, alpha=0.5, ax=ax)
        sns.kdeplot(X_val[feature], label='Validation', fill=True, alpha=0.5, ax=ax)
        sns.kdeplot(X_test[feature], label='Test', fill=True, alpha=0.5, ax=ax)
        
        # Настройки оформления
        ax.set_title(f'Распределение признака "{feature}"')
        ax.set_xlabel(feature)
        ax.set_ylabel('Плотность вероятности')
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()




def compare_all_categorical_features(categorical_features, X_train, X_val, X_test):
    n_features = len(categorical_features)
    fig, axes = plt.subplots(1, n_features, figsize=(5 * n_features, 4))

    for ax, feature in zip(axes, categorical_features):
        # Получаем распределения
        train_counts = X_train[feature].value_counts(normalize=True).rename('Train')
        val_counts = X_val[feature].value_counts(normalize=True).rename('Validation')
        test_counts = X_test[feature].value_counts(normalize=True).rename('Test')

        df = pd.concat([train_counts, val_counts, test_counts], axis=1).fillna(0)

        # Строим график на соответствующей оси
        df.plot(kind='bar', ax=ax)
        ax.set_title(f'"{feature}"')
        ax.set_ylabel('Доля')
        ax.set_xlabel('Категория')
        ax.set_xticklabels(df.index, rotation=0)
        ax.legend()

    plt.tight_layout()
    plt.show()