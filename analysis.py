import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
import plotly.express as px
from skopt.space import Real, Categorical, Integer 
import pandas as pd 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
import xgboost as xgb


from visualisations import rf_cv_analysis, en_cv_analysis

def perform_pca(X_train_std, X_test_std, variance_threshold=0.9):
    pca = PCA(n_components=variance_threshold)  # 90% of variance explained

    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)
    explained_variance = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance)

    return pca, X_train_pca, X_test_pca, explained_variance, cumulative_explained_variance


ElasticNet_L1_ratios = np.unique(np.concatenate(([0], np.logspace(-2, 0, 10), [1])))

ElasticNet_grid_paramgrid =  {
            'alpha': np.logspace(-1, 2, 15),
            'l1_ratio': ElasticNet_L1_ratios
}

ElasticNet_bayes_paramgrid = {
    'alpha': Real(0.1, 30, prior='log-uniform'),
    'l1_ratio': Categorical(ElasticNet_L1_ratios)
}

RandomForest_bayes_paramspace = {
    'n_estimators': Integer(50, 100), # number of trees 
    'max_depth': Integer(2, 30),
    'min_samples_split': Integer(25, 100), # min samples to split a node
    'min_samples_leaf': Integer(10, 100),  # min samples in a leaf
    'max_features': Categorical(['sqrt', 'log2', 300]) # sqrt - 95, log - 13, 
}

XGBoost_bayes_paramspace = {
    'n_estimators': Integer(70, 250), 
    'max_depth': Integer(2, 15),
    'learning_rate': Real(0.1, 0.25, prior='log-uniform'), # fewer - slower, 
    'subsample': Real(0.3, 1.0),
    'colsample_bytree': Real(0.2, 1.0),
}


def en_create_GridSearchCV(model, cv_folds):
    search = GridSearchCV(
        estimator=model,
        param_grid=ElasticNet_grid_paramgrid,
        cv=cv_folds,
        n_jobs=-1,
        scoring='neg_root_mean_squared_error',
        return_train_score=True
    )
    return search 

def en_create_BayesSearchCV(model, cv_folds, n_iter=75):
    search = BayesSearchCV(
        estimator=model,
        search_spaces=ElasticNet_bayes_paramgrid,
        n_iter=n_iter,
        cv=cv_folds,
        n_jobs=-1,
        scoring='neg_root_mean_squared_error',
        return_train_score=True,
        random_state=111
    )
    return search 

def rf_create_BayesSearchCV(model, cv_folds=3): 
    search = BayesSearchCV(
    estimator=model,
    search_spaces=RandomForest_bayes_paramspace,
    n_iter=40,  # Number of iterations for Bayesian optimization, small to speed up
    cv=cv_folds,  
    n_jobs=-1,
    scoring='neg_root_mean_squared_error',
    return_train_score=True,
    random_state=111
    )
    return search

def xg_create_BayesSearchCV(model, cv_folds=3):
    search = BayesSearchCV(
    estimator=model,
    search_spaces=XGBoost_bayes_paramspace,
    n_iter=20,
    cv=3,
    n_jobs=-1,
    verbose=3,
    random_state=111
    )
    return search

def extract_search_results(search, en):
    best_params = search.best_params_
    best_model = search.best_estimator_
    cv_results = search.cv_results_
    # Printing results:
    print(f"Best params: {best_params}") 
    mean_train_rmse = -cv_results['mean_train_score'].mean()
    print("Mean Training RMSE (cross-validated):", mean_train_rmse)
    mean_val_rmse = -cv_results['mean_test_score'].mean()
    print("Mean Validation RMSE (cross-validated):", mean_val_rmse)

    cv_results_df = pd.DataFrame(cv_results)
    cv_results_df['mean_test_score_abs'] = cv_results_df['mean_test_score'].abs()

    return {
        'best_params': best_params,
        'best_model': best_model,
        'cv_results_df': cv_results_df
    }



def cv_analysis(search, type = "en"):
    res = extract_search_results(search, type)
    cv_results_df = res['cv_results_df']
    if type == "en": 
        en_cv_analysis(cv_results_df)
    elif type == "rf": 
        rf_cv_analysis(cv_results_df)
    else: 
        print("Wrong type: rf/en")
    return res



def perform_elastic_net_cv_grid(X, y, cv_folds):
    model = ElasticNet(max_iter=10000, warm_start=True, selection='random')
    search = en_create_GridSearchCV(model, cv_folds)
    # Fit the model
    search.fit(X, y)
    return search 

def perform_elastic_net_cv_bayes(X, y, cv_folds):
    model = ElasticNet(max_iter=10000, warm_start=True, selection='random')
    search = en_create_BayesSearchCV(model, cv_folds)
    search.fit(X, y)
    return search 

def perform_random_forest_reg_bayes(X, y, cv_folds): 
    model = RandomForestRegressor(random_state=111)
    search = rf_create_BayesSearchCV(model, cv_folds)
    search.fit(X, y)
    return search 

def perform_xgboost_bayes(X, y, cv_folds=3):
    model = xgb.XGBRegressor(objective='reg:squarederror')
    search = xg_create_BayesSearchCV(model, cv_folds)
    search.fit(X, y)
    return search


# Evaluate a regression model using RMSE metric on validation data.
def evaluate_model_rmse(model, X_val, y_val):
    y_pred = model.predict(X_val)
    rmse = root_mean_squared_error(y_val, y_pred)
    print("Validation RMSE on validation split:", rmse)


def evaluate_model_metrics(y_pred, model_name, y_test):
    print(f"Metrics for: {model_name}")
    # y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = root_mean_squared_error(y_test, y_pred)  # RMSE
    mae = mean_absolute_error(y_test, y_pred)  # MAE
    r2 = r2_score(y_test, y_pred)  # R2 score
    
    # Print metrics
    print("-- RMSE:", rmse)
    print("-- MAE:", mae)
    print("-- R2 Score:", r2)
    
    # Return metrics
    return rmse, mae, r2


def evaluate_models_and_baseline(best_en_model, best_rf_model, y_train, X_test_pca, y_test):
    baseline_pred = np.full_like(y_test, y_test.mean()) 
    rf_pred = best_rf_model.predict(X_test_pca)
    en_pred = best_en_model.predict(X_test_pca)

    baseline_rmse, baseline_mae, baseline_r2 = evaluate_model_metrics(baseline_pred, "Baseline", y_test)
    rf_rmse, rf_mae, rf_r2 = evaluate_model_metrics(rf_pred, "Random Forest", y_test)
    en_rmse, en_mae, en_r2 = evaluate_model_metrics(en_pred, "ElasticNet", y_test)


    results = pd.DataFrame({
        'Model': ['ElasticNet', 'Random Forest', 'Baseline'],
        'RMSE': [en_rmse, rf_rmse, baseline_rmse],
        'MAE': [en_mae, rf_mae, baseline_mae],
        'R^2 Score': [en_r2, rf_r2, baseline_r2]
    })

    print(results)
    return baseline_pred, en_pred, rf_pred, results

def optimize_feature_selection(model, X_train_pca, X_test_pca, y_train, y_test, importances, k_min=10, k_max=200, k_to_try = 35): 
    results = []

    X_train_pca_df = pd.DataFrame(X_train_pca)
    X_test_pca_df = pd.DataFrame(X_test_pca)
    importances_df = pd.DataFrame(importances)
    importances_df = importances_df.sort_values(by=0, ascending=False)
    k_values = np.unique(np.linspace(k_min, k_max, num=k_to_try, dtype=int))

    for k in k_values:
        # Get importances 
        top_features_indices = importances_df[:k]
        idx = top_features_indices.index
        X_train_pca_top_features = X_train_pca_df.iloc[:, idx]
        X_test_pca_top_features = X_test_pca_df.iloc[:, idx]

        # Fit with model. 
        model.fit(X_train_pca_top_features, y_train)
        y_pred = model.predict(X_test_pca_top_features)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        results.append((k, rmse, r2))

    return results