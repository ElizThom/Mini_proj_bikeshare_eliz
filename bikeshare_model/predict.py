import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from bikeshare_model import __version__ as _version
from bikeshare_model.config.core import config
from bikeshare_model.pipeline import bikeshare_pipe
from bikeshare_model.processing.data_manager import load_pipeline
from bikeshare_model.processing.data_manager import pre_pipeline_preparation
from bikeshare_model.processing.validation import validate_inputs


pipeline_file_name = "bikeshare__model_output_v0.0.1.pkl"
bikeshare_pipe= load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, input_data: Union[pd.DataFrame, dict]) -> dict:
    validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))
    
    # Convert numerical features to float
    numerical_features = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered']
    for feature in numerical_features:
        validated_data[feature] = validated_data[feature].astype(float)

    validated_data = validated_data.reindex(columns=config.model_config_.features)
    print(f"validated_data: {validated_data}")
    results = {"predictions": None, "version": _version, "errors": errors}
    
    if not errors:

        predictions = bikeshare_pipe.predict(validated_data)
        results = {"predictions": predictions,"version": _version, "errors": errors}
        print(results)

    print(results)
    return results

if __name__ == "__main__":

    data_in={
        'dteday': ['14-01-2012'],
        'season': ['winter'],
        'hr': ['4pm'],
        'holiday': ['Yes'],
        'weekday': ['Fri'],
        'weathersit': ['Mist'],
        'temp': ['18.01'],
        'atemp': ['19.5'],
        'hum': ['90'],
        'windspeed': ['15.2'],
        'casual': ['17'],
        'registered': ['230']
    }
    
    make_prediction(input_data=data_in)
