
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def print_missing_columns(df, df_name):
  are_any = False
  for column in df.columns:
    missing_values_count = df[column].isnull().sum()
    if missing_values_count > 0:
        print(f"Column '{column}' has {missing_values_count} missing values.")
        are_any = True
  if are_any is False:
      print(f"No missing values in the table {df_name}.")


def print_non_numeric_columns(df, df_name):
    are_any = False
    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            print(f"Column '{column}' is not numeric.")
            are_any = True
    if not are_any:
        print(f"All columns in the table {df_name} are numeric.")

def get_constant_columns(df, df_name):
    identical_columns = [col for col in df.columns if df[col].nunique() == 1]
    print(f"Column in  {df_name} with all identical rows:")
    print(identical_columns)
    print(f"Number of columns in {df_name} with all identical rows: {len(identical_columns)}")
    return identical_columns

def preprocess_datasets(X_train, X_test, y_train):
    """
    Perform preprocessing on the datasets including printing shapes, 
    missing values, non-numeric columns, and removing constant columns.

    Parameters:
    X_train (pd.DataFrame): Training dataset features.
    X_test (pd.DataFrame): Testing dataset features.
    y_train (pd.DataFrame or pd.Series): Training dataset target.

    Returns:
    Tuple: Processed X_train, X_test, y_train.
    """
    print("Shapes of the datasets [rows, columns]")
    print(f"X_test shape: {X_test.shape}.")
    print(f"X_train shape: {X_train.shape}.")
    print(f"y_train shape: {y_train.shape}.")
    print("\n")

    print("Missing values:")
    print_missing_columns(X_test, "X_test")
    print_missing_columns(X_train, "X_train")
    print_missing_columns(y_train, "y_train")
    print("\n")

    print("Column types (expected - numeric): ")
    print_non_numeric_columns(X_test, "X_test")
    print_non_numeric_columns(X_train, "X_train")
    print_non_numeric_columns(y_train, "y_train")
    print("\n")

    constant_columns = get_constant_columns(X_train, "X_train")
    print("Removed constant columns from X_train, X_test")
    X_train = X_train.drop(columns=constant_columns)
    X_test = X_test.drop(columns=constant_columns)

    return X_train, X_test, y_train



def standardize_datasets(X_train, X_test):
    """
    Standardizes the training and test datasets using StandardScaler.

    Parameters:
    X_train (pd.DataFrame): The training dataset features.
    X_test (pd.DataFrame): The test dataset features.

    Returns:
    Tuple: Standardized training and test datasets, and the scaler's mean and scale.
    """
    scaler = StandardScaler()
    
    # Fit and transform the training data
    X_train_std = scaler.fit_transform(X_train)
    
    # Transform the test data (using the same scaler fitted on training data)
    X_test_std = scaler.transform(X_test)
    
    # Print the mean and scale
    print('mean_:', scaler.mean_)
    print('scale_:', scaler.scale_)
    
    return scaler, X_train_std, X_test_std

def split_dataset(X_train_std, y_train, test_size=0.2, random_state=244):
    """
    Splits the standardized training data into training and test sets, prints their shapes,
    and also creates a smaller training subset for testing purposes.

    Parameters:
    X_train_std (np.ndarray): Standardized training dataset features.
    y_train (pd.DataFrame or pd.Series): Full training dataset target.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Controls the shuffling applied to the data before applying the split.

    Returns:
    Tuple: Split datasets (X_train_std, X_test_std, y_train, y_test, X_train_std_small, y_train_small).
    """

    # Split the data into training and test sets
    X_train_std, X_test_std, y_train, y_test = train_test_split(
        X_train_std, y_train, test_size=test_size, random_state=random_state
    )
    # Print shapes of the splits
    print("X_train_std shape:", X_train_std.shape)
    print("X_test_std shape:", X_test_std.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    
    # Create a smaller training subset for testing purposes
    X_train_std_small, _, y_train_small, _ = train_test_split(
        X_train_std, y_train, test_size=0.9, random_state=random_state
    )
    # Print shapes of the smaller training subset
    print("\nSmall samples to test functions:")
    print("X_train_std_small shape:", X_train_std_small.shape)
    print("y_train_small shape:", y_train_small.shape)
    
    return X_train_std, X_test_std, y_train, y_test, X_train_std_small, y_train_small

