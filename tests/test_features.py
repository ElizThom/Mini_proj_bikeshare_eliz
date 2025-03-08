
"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
import pandas as pd
from bikeshare_model.config.core import config
from bikeshare_model.processing.features import WeekdayImputer, Mapper, WeathersitImputer, WeekdayOneHotEncoder


def test_weekday_imputer():
    # Create a sample DataFrame with missing 'weekday' values
    data = {
        'dteday': ['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04'],
        'weekday': [None, 'Sat', None, 'Mon']
    }
    df = pd.DataFrame(data)

    # Expected output after imputation
    expected_data = {
        'dteday': ['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04'],
        'weekday': ['Fri', 'Sat', 'Sun', 'Mon']
    }
    expected_df = pd.DataFrame(expected_data)

    # Initialize the WeekdayImputer
    imputer = WeekdayImputer(variables='weekday')

    # Transform the DataFrame
    result_df = imputer.transform(df)

    # Assert that the transformed DataFrame matches the expected output
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_mapper():
    # Create a sample DataFrame
    data = {
        'season': ['spring', 'summer', 'fall', 'winter'],
    }
    df = pd.DataFrame(data)

    # Define the mapping
    season_mapping = {
        'spring': 1,
        'summer': 2,
        'fall': 3,
        'winter': 4
    }

    # Expected output after mapping
    expected_data = {
        'season': [1, 2, 3, 4],
    }
    expected_df = pd.DataFrame(expected_data)

    # Initialize the Mapper
    mapper = Mapper(variables='season', mappings=season_mapping)

    # Transform the DataFrame
    result_df = mapper.transform(df)

    # Assert that the transformed DataFrame matches the expected output
    pd.testing.assert_frame_equal(result_df, expected_df)

def test_weathersit_imputer():
    # Create a sample DataFrame with missing 'weathersit' values
    data = {
        'weathersit': [None, 'Clear', None, 'Mist']
    }
    df = pd.DataFrame(data)

    # Expected output after imputation
    expected_data = {
        'weathersit': ['Clear', 'Clear', 'Clear', 'Mist']
    }
    expected_df = pd.DataFrame(expected_data)

    # Initialize the WeathersitImputer
    imputer = WeathersitImputer(variables='weathersit')

    # Fit and transform the DataFrame
    imputer.fit(df)
    result_df = imputer.transform(df)

    # Assert that the transformed DataFrame matches the expected output
    pd.testing.assert_frame_equal(result_df, expected_df)

def test_weekday_onehotencoder():
    # Create a sample DataFrame with 'weekday' values
    data = {
        'weekday': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    }
    df = pd.DataFrame(data)

    # Expected output after one-hot encoding
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

    # Sort columns before comparison
    result_df = result_df.sort_index(axis=1)
    expected_df = expected_df.sort_index(axis=1)

    # Cast result_df to match the dtypes of expected_df
    result_df = result_df.astype(expected_df.dtypes)

    # Assert that the transformed DataFrame matches the expected output
    pd.testing.assert_frame_equal(result_df, expected_df)