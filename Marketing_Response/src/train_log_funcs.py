import mlflow
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
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report






# Функция автоматического подбора гиперпараметров
def perform_grid_search(model, param_grid, X_train, y_train, X_val, y_val, scoring='f1'):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    
    # Добавляем расчет полных метрик
    y_pred = best_model.predict(X_val)
    print("\nОтчет по классификации на валидации:")
    print(classification_report(y_val, y_pred))
    
    # Возвращаем полный словарь метрик
    val_metrics = get_metrics(best_model, X_val, y_val)
    return best_model, grid_search.best_params_, val_metrics







def get_metrics(model, X, y, threshold=0.4):
    """
    Вычисляет метрики качества классификации и возвращает их в виде словаря.

    Параметры:
        model: обученная модель (поддерживающая методы predict / predict_proba / decision_function)
        X: признаки
        y: истинные метки классов
        threshold: порог для перевода вероятностей в метки классов (используется, если есть вероятности или decision_function)

    Возвращает:
        Словарь с метриками:
            - recall: полнота
            - precision: точность
            - f1: F1-мера
            - roc_auc: площадь под ROC-кривой (если есть вероятности)
            - pr_auc: площадь под PR-кривой (если есть вероятности)
            - confusion_matrix: матрица ошибок
            - y_true: истинные метки
            - y_proba: предсказанные вероятности или оценки (если есть)
            - y_pred: предсказанные метки
    """
    has_proba = hasattr(model, "predict_proba")
    has_decision = hasattr(model, "decision_function")
    
    if has_proba:
        y_scores = model.predict_proba(X)[:, 1]
    elif has_decision:
        y_scores = model.decision_function(X)
    else:
        y_scores = None

    if y_scores is not None:
        y_pred = (y_scores >= threshold).astype(int)
    else:
        y_pred = model.predict(X)

    return {
        "recall": recall_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_scores) if y_scores is not None else None,
        "pr_auc": average_precision_score(y, y_scores) if y_scores is not None else None,
        "confusion_matrix": confusion_matrix(y, y_pred),
        "y_true": np.array(y),
        "y_proba": y_scores if y_scores is not None else np.array([]),
        "y_pred": y_pred
    }

def log_experiment_train_val_test(model_name, model, params, X_train, y_train, X_val, y_val, X_test, y_test, threshold=0.4):
    """
    Логирует метрики, артефакты и модель в MLflow для трёх сплитов: train, val и test.

    Параметры:
    ----------
    model_name : str
        Название модели/эксперимента, которое будет использовано как имя запуска в MLflow.
    
    model : sklearn-подобный объект
        Обученная модель, поддерживающая методы predict / predict_proba или decision_function.
    
    params : dict
        Словарь гиперпараметров, используемых для обучения модели (будет залогирован в MLflow).
    
    X_train, y_train : array-like
        Обучающая выборка и целевая переменная.
    
    X_val, y_val : array-like
        Валидационная выборка и целевая переменная.
    
    X_test, y_test : array-like
        Тестовая выборка и целевая переменная.
    
    threshold : float, по умолчанию 0.4
        Порог вероятности для перевода в бинарный предикт при расчёте метрик.

    Функциональность:
    -----------------
    - Завершает активный MLflow-запуск (если он существует) и создаёт новый.
    - Логирует гиперпараметры (включая порог) и метрики (f1, recall, precision, ROC AUC, PR AUC) для каждого сплита.
    - Сохраняет и логирует изображения:
        - Матрицы ошибок (Confusion Matrices)
        - ROC-кривые
        - Precision-Recall кривые
    - Логирует саму модель (с использованием mlflow.sklearn).
    - Выводит ID завершённого запуска в MLflow.

    Возвращает:
    -----------
    None
    """

    run = mlflow.active_run()
    if run is not None:
        mlflow.end_run()
        run = mlflow.get_run(run.info.run_id)
        print(f"run_id: {run.info.run_id}; status: {run.info.status}")

    mlflow.start_run(run_name=model_name)
    run = mlflow.active_run()
    print(f"run_id: {run.info.run_id}; status: {run.info.status}")
    mlflow.set_tag("comment", "class_weights=[1,3]")
    
    # Убрано дублирование логирования параметров
    params["threshold"] = threshold
    mlflow.log_params(params)

    roc_data = {}
    pr_data = {}
    cm_data = {}

    # Сохраняю метрики по каждому сплиту
    for split_name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        metrics = get_metrics(model, X, y, threshold=threshold)

        # Метрики для сводных графиков
        if metrics["y_proba"] is not None:
            roc_data[split_name] = (metrics["y_true"], metrics["y_proba"])
            pr_data[split_name] = (metrics["y_true"], metrics["y_proba"])
        cm_data[split_name] = metrics["confusion_matrix"]

        # Логирую метрики
        metrics_to_log = {
            f"{split_name}_{k}": float(v)
            for k, v in metrics.items()
            if k not in ["confusion_matrix", "y_true", "y_proba", "y_pred"]
        }
        mlflow.log_metrics(metrics_to_log)

    # Confusion Matrix
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, split_name in zip(axes, ["train", "val", "test"]):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_data[split_name])
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(f"{split_name.upper()}")
    fig.suptitle(f"{model_name} - Confusion Matrices", fontsize=16)
    plt.tight_layout()
    cm_all_path = f"cm_all_{model_name}.png"
    plt.savefig(cm_all_path)
    plt.close()
    mlflow.log_artifact(cm_all_path)

    # ROC
    plt.figure(figsize=(8, 6))
    for split_name, (y_true, y_proba) in roc_data.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{split_name.upper()} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"{model_name} - ROC Curve (All Splits)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    roc_all_path = f"roc_all_{model_name}.png"
    plt.savefig(roc_all_path)
    plt.close()
    mlflow.log_artifact(roc_all_path)

    # PR Curve
    plt.figure(figsize=(8, 6))
    for split_name, (y_true, y_proba) in pr_data.items():
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        avg_prec = average_precision_score(y_true, y_proba)
        plt.plot(recall, precision, label=f"{split_name.upper()} (AP = {avg_prec:.2f})")
    plt.title(f"{model_name} - Precision-Recall Curve (All Splits)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="upper right")
    pr_all_path = f"pr_all_{model_name}.png"
    plt.savefig(pr_all_path)
    plt.close()
    mlflow.log_artifact(pr_all_path)

    # Логирую модель
    mlflow.sklearn.log_model(model, "model")
    mlflow.end_run()

    run = mlflow.get_run(run.info.run_id)
    print(f"run_id: {run.info.run_id}; status: {run.info.status}")


    

def train_and_log_all_models(models_with_params, X_train, y_train, X_val, y_val, X_test, y_test): 
    trained_models = {}
    for model_name, (model, param_grid) in models_with_params.items():
        print(f"\nTraining model: {model_name}")

        if param_grid:
            model, best_params, val_metrics = perform_grid_search(
                model, 
                param_grid, 
                X_train, y_train, 
                X_val, y_val
            )
            # Полное логирование для GridSearch
            log_experiment_train_val_test(
                f"GridSearch - {model_name}", 
                model, 
                best_params, 
                X_train, y_train, 
                X_val, y_val, 
                X_test, y_test,
                threshold=0.4
            )
        else:
            model.fit(X_train, y_train)
            log_experiment_train_val_test(
                model_name, 
                model, 
                {}, 
                X_train, y_train, 
                X_val, y_val, 
                X_test, y_test,
                threshold=0.4
            )

        trained_models[model_name] = model
    
    return trained_models









