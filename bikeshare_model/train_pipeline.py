import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score


from bikeshare_model.config.core import config
from bikeshare_model.pipeline import bikeshare_pipe
from bikeshare_model.processing.data_manager import load_dataset, save_pipeline

def run_training() -> None:
    
    """
    Train the model.
    """

    # read training data
    data = load_dataset(file_name=config.app_config_.training_data_file)
    print(f"Initial data shape: {data.shape}")

     # Check for NaN and infinite values
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    print(f"Number of NaN values before imputation: {data[numeric_cols].isna().sum().sum()}")


    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config_.features],  # predictors
        data[config.model_config_.target],
        test_size=config.model_config_.test_size,
        random_state=config.model_config_.random_state,
    )

    print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")
    
    # Pipeline fitting
    bikeshare_pipe.fit(X_train,y_train)
   
    # persist trained model
    save_pipeline(pipeline_to_persist= bikeshare_pipe)
    # printing the score
    y_pred = bikeshare_pipe.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared (R2) Score: {r2}")

    
if __name__ == "__main__":
    run_training()
