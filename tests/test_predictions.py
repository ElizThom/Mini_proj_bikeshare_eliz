"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
import pytest
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from bikeshare_model.predict import make_prediction
from bikeshare_model.processing.features import WeekdayImputer, Mapper, WeathersitImputer, WeekdayOneHotEncoder

@pytest.fixture
def sample_input_data1():
    data = {
        'season': ['spring', 'summer', 'fall', 'winter','spring'],
        'hr': ['4pm', '5pm', '6pm', '7pm', '8pm'],
        'holiday': ['Yes', 'No', 'Yes', 'No', 'No'],
        'weekday': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'],
        'workingday': [None, 'Yes', None, 'Yes','Yes'],
        'weathersit': [None, 'Clear', None, 'Mist', 'Rain'],
        'temp': [18.01, 20.5, 15.0, 10.0, 25.0],
        'atemp': [19.5, 21.0, 16.0, 11.0, 26.0],
        'hum': [90, 80, 85, 70, 75],
        'windspeed': [15.2, 10.0, 12.0, 8.0, 5.0],
        'casual': [17, 20, 15, 10, 30],
        'registered': [230, 250, 200, 150, 300],
        'yr': [None, '2011', None, '2012', '2013'],
        'mnth': [None, 'Jan', None, 'Feb', 'Mar'],
        'dteday': ['14-01-2011', '15-01-2011', '16-01-2011', '17-01-2011', '18-01-2011']
    }
    df = pd.DataFrame(data)
    y_true = [175, 207, 175, 154, 233]  # Example true labels for testing
    return df, y_true

def test_make_prediction(sample_input_data1):
    # Given
    df, _ = sample_input_data1 
    df['mnth'] = df['mnth'].map({
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }).fillna(0)  # Ensure 'mnth' column is converted to integers
    expected_no_predictions = len(df)

    # When
    result = make_prediction(input_data=df)

    # Then
    predictions = result.get("predictions")
    assert isinstance(predictions, np.ndarray), "Predictions should be a numpy array."
    assert isinstance(predictions[0], (np.int64, np.float64)), "Prediction values should be numeric."
    assert result.get("errors") is None, f"Errors encountered: {result.get('errors')}"
    assert len(predictions) == expected_no_predictions, "Number of predictions should match input size."

    # Compare predictions with true values (if appropriate, scale y_true accordingly)
    _predictions = list(predictions)
    y_true = sample_input_data1[1]
    accuracy = accuracy_score(_predictions, y_true)
    assert accuracy > 0.05


def test_weathersit_imputer(sample_input_data1):
    # Given
    df, _ = sample_input_data1  # Use the fixture data
    df = df.head(4).copy()  # Use a subset of the data
    expected_data = {
        'weathersit': ['Clear', 'Clear', 'Clear', 'Mist']
    }
    expected_df = pd.DataFrame(expected_data)

    # Initialize the WeathersitImputer
    imputer = WeathersitImputer(variables='weathersit')

    # Fit and transform the DataFrame
    imputer.fit(df)
    result_df = imputer.transform(df)

    # Reset index before comparison
    result_df.reset_index(drop=True, inplace=True)
    expected_df.reset_index(drop=True, inplace=True)

    # Assert that the transformed DataFrame matches the expected output
    pd.testing.assert_frame_equal(result_df[['weathersit']], expected_df)


def test_weekday_onehotencoder(sample_input_data1):
    # Given
    df, _ = sample_input_data1  # Use the fixture data
    df = df.head(5).copy()  # Use a subset of the data
    expected_data = {
            'weekday_Mon': [1, 0, 0, 0, 0],
            'weekday_Tue': [0, 1, 0, 0, 0],
            'weekday_Wed': [0, 0, 1, 0, 0],
            'weekday_Thu': [0, 0, 0, 1, 0],
            'weekday_Fri': [0, 0, 0, 0, 1]
       }
    expected_df = pd.DataFrame(expected_data)

    # Initialize the WeekdayOneHotEncoder
    encoder = WeekdayOneHotEncoder(handle_unknown='ignore')

    # Fit and transform the DataFrame
    encoder.fit(df)
    result_df = encoder.transform(df)

    # Sort the columns and reset index before comparison
    #result_df = result_df[['weekday_Mon', 'weekday_Tue', 'weekday_Wed', 'weekday_Thu', 'weekday_Fri']]
    #result_df.reset_index(drop=True, inplace=True)
    #expected_df.reset_index(drop=True, inplace=True)

    # Convert data types to match expected DataFrame
    result_df = result_df.astype(expected_df.dtypes)
    # Ensure result_df only contains columns from expected_df
    result_df = result_df[expected_df.columns]

    # Sort columns before comparison
    result_df = result_df.sort_index(axis=1)
    expected_df = expected_df.sort_index(axis=1)

    # Assert that the transformed DataFrame matches the expected output
    pd.testing.assert_frame_equal(result_df, expected_df, check_like=True)


if __name__ == "__main__":
    pytest.main()