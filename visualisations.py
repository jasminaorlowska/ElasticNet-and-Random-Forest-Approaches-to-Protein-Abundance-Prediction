import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt


from IPython.display import display




def basic_plots_df(df, column_name):
    """
    Generates basic statistics and plots for a specified column in the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    column_name (str): Name of the column for which the plots are to be generated.

    Returns:
    None (the function displays the plots and prints the statistics).
    """
    print("Basic statistics:")
    display(df[column_name].describe())

    # Histogram and KDE plot
    fig_hist_kde = ff.create_distplot([df[column_name]], group_labels=['Target'], bin_size=0.1)
    fig_hist_kde.update_layout(title_text='Histogram and Density Plot')

    # Boxplot
    fig_box = px.box(df, y=column_name, title='Boxplot')

    # QQ Plot
    qq_data = np.sort(df[column_name])
    qq_theoretical = np.sort(stats.norm.ppf(np.linspace(0.01, 0.99, len(qq_data)), np.mean(qq_data), np.std(qq_data)))
    fig_qq = go.Figure()
    fig_qq.add_trace(go.Scatter(x=qq_theoretical, y=qq_data, mode='markers', name='Data'))
    fig_qq.add_trace(go.Scatter(x=qq_theoretical, y=qq_theoretical, mode='lines', name='Theoretical Quantiles'))
    fig_qq.update_layout(title='QQ Plot', xaxis_title='Theoretical Quantiles', yaxis_title='Empirical Quantiles')

    # Bee Swarm Plot
    fig_bee_swarm = px.strip(df, y=column_name, title='Bee Swarm Plot')

    # Show all plots
    fig_hist_kde.show()
    fig_box.show()
    fig_qq.show()
    fig_bee_swarm.show()

def plot_top_250_feature_correlations(X, y, target_column):
    """
    Plots a heatmap of the top 250 features based on their correlation with the target variable.

    Parameters:
    X (pd.DataFrame): The full training dataset with features.
    y (pd.DataFrame or pd.Series): The target variable.
    target_column (str): The name of the target column in y (if it's a DataFrame).

    Returns:
    pd.Index: The indices of the top 250 features.
    pd.DataFrame: The correlation matrix of the top 250 features.
    """
    if X.isin([np.inf, -np.inf]).any().any():
            print("Error: Inf values found in X.")
            return
    if y[target_column].isin([np.inf, -np.inf]).any():
        print(f"Error: Inf values found in y['{target_column}'].")
        return

    correlations = X.corrwith(y[target_column]).abs()
    top_250_features = correlations.abs().nlargest(250).index

    # Utwórz zbiór danych z wybranymi cechami
    X_top250 = X[top_250_features]

    # Plot the heatmap of top 250 features correlation
    correlation_matrix = X_top250.corr()
    plt.figure(figsize=(20, 20))
    sns.heatmap(correlation_matrix, cmap='plasma')
    plt.title('Heatmap of Top 250 Features Correlation')
    plt.show()

    # Plot the distribution of correlations
    plt.figure(figsize=(10, 6))
    sns.histplot(correlations, bins=50, kde=True)
    plt.title('Distribution of Feature Correlations with Target')
    plt.xlabel('Correlation coefficient')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


    abs_values = np.abs(correlation_matrix.values)
    abs_values_flat = abs_values[np.triu_indices_from(abs_values, k=1)]
    plt.hist(abs_values_flat, bins=30, edgecolor='black')
    plt.title('Histogram of correlations between 250 most correlated with y X features')
    plt.xlabel('ABS value')
    plt.ylabel('Frequency')
    plt.show()



def plot_explained_variance(explained_variance, cumulative_explained_variance, n_components=50):
    """
    Plots the explained variance and cumulative explained variance.

    Parameters:
    explained_variance (np.ndarray): Explained variance ratio for each principal component.
    cumulative_explained_variance (np.ndarray): Cumulative explained variance ratio.
    """
    plt.figure(figsize=(12, 8))
    # Limit the number of components to display
    explained_variance = explained_variance[:n_components]
    cumulative_explained_variance = cumulative_explained_variance[:n_components]

    # Bar plot for individual explained variance
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, align='center', label='Individual explained variance', color='skyblue')
    # Line plot for cumulative explained variance
    plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', color='orange', label='Cumulative explained variance')
    
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.title(f'Explained Variance by Principal Components (Top {n_components})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def display_dict_as_dataframe(d):
    df = pd.DataFrame.from_dict(d, orient='index')
    display(df) 
    print(df)  

def rf_cv_analysis(cv_results_df):
    fig = px.scatter(
    cv_results_df,
    x='param_n_estimators',
    y='param_max_depth',
    size='param_min_samples_leaf',
    color='param_min_samples_split',
    hover_data=['mean_test_score', 'param_min_samples_split', 'param_min_samples_leaf'],
    labels={
        'param_n_estimators': 'N Estimators',
        'param_max_depth': 'Max Depth',
        'mean_test_score': 'Mean Test Score (RMSE)',
        'param_min_samples_split': 'Min Samples Split',
        'param_min_samples_leaf': 'Min Samples Leaf'
    },
    title='CV results for RandomForestRegressor'
    )
    fig.show()

    # Additional plots
    fig1 = px.scatter(
        cv_results_df,
        x='param_max_depth',
        y='param_min_samples_split',
        size='mean_test_score_abs',
        color='mean_test_score_abs',
        labels={
            'param_max_depth': 'Max Depth',
            'param_min_samples_split': 'Min Samples Split',
            'mean_test_score_abs': 'Mean Test Score (RMSE)'
        },
        title='Impact of Max Depth and Min Samples Split on RMSE'
    )
    fig1.show()

    fig2 = px.scatter(
        cv_results_df,
        x='param_n_estimators',
        y='param_min_samples_leaf',
        size='mean_test_score_abs',
        color='mean_test_score_abs',
        labels={
            'param_n_estimators': 'N Estimators',
            'param_min_samples_leaf': 'Min Samples Leaf',
            'mean_test_score_abs': 'Mean Test Score (RMSE)'
        },
        title='Impact of N Estimators and Min Samples Leaf on RMSE'
    )
    fig2.show()

def en_cv_analysis(cv_results_df):
         # Plot results
    fig = px.scatter(cv_results_df, x='param_alpha', y='param_l1_ratio', size='mean_test_score_abs', color='mean_test_score_abs',
                     labels={'param_alpha': 'Alpha', 'param_l1_ratio': 'L1 Ratio', 'mean_test_score_abs': 'Mean Test Score (RMSE)'},
                     title='CV results for ElasticNet')
    fig.show()

def plot_expected_vs_predicted(y_test, y_pred, model_name):
    # Create scatter plot
    fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Expected', 'y': 'Predicted'}, title=f'{model_name}: Expected vs Predicted')
    
    # Add lines for perfect prediction, min predicted and max predicted
    fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], 
                             mode='lines', line=dict(color='red', dash='dash'), name='Perfect Prediction'))
    fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_pred.min(), y_pred.min()], 
                             mode='lines', line=dict(color='green', dash='dash'), name='Min Predicted'))
    fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_pred.max(), y_pred.max()], 
                             mode='lines', line=dict(color='blue', dash='dash'), name='Max Predicted'))
    
    # Show the plot
    fig.show()

def plot_residuals(y_test, y_pred, model_name):
    # Calculate residuals
    residuals = y_test - y_pred
    
    # Create scatter plot
    fig = px.scatter(x=y_pred, y=residuals, labels={'x': 'Predicted', 'y': 'Residuals'}, title=f'{model_name}: Residuals')
    
    # Add line for zero error
    fig.add_trace(go.Scatter(x=[y_pred.min(), y_pred.max()], y=[0, 0], 
                             mode='lines', line=dict(color='red', dash='dash'), name='Zero Error'))
    
    # Show the plot
    fig.show()


def visiualize_features_rmse_r2_dependance(results): 
    results_df = pd.DataFrame(results, columns=['k', 'RMSE', 'R2'])

    best_result = results_df.loc[results_df['R2'].idxmax()]
    best_k = best_result['k']
    print(f"Best k: {best_k}, RMSE: {best_result['RMSE']}, R2: {best_result['R2']}")

    plt.figure(figsize=(10, 6))
    plt.plot(results_df['k'], results_df['RMSE'], marker='o', label='RMSE')
    plt.plot(results_df['k'], results_df['R2'], marker='o', label='R2')
    plt.xlabel('Number of Features (k)')
    plt.ylabel('Score')
    plt.title('Model Performance for Different Numbers of Features')
    plt.legend()
    plt.grid(True)
    plt.show()