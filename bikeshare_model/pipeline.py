import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from bikeshare_model.config.core import config
from bikeshare_model.processing.features import WeekdayImputer, WeathersitImputer
from bikeshare_model.processing.features import Mapper
from bikeshare_model.processing.features import WeekdayOneHotEncoder, DropColumns

# Create an instance of OutlierHandler
#outlier_handler = OutlierHandler(variables=['temp', 'atemp', 'hum', 'windspeed'])
#print(f"OutlierHandler variables: {outlier_handler.variables}")  # Debugging print

# Define the pipeline
bikeshare_pipe = Pipeline([
    ("weekday_imputer", WeekdayImputer(variables=config.model_config_.weekday_var)),
    ("weathersit_imputer", WeathersitImputer(variables=config.model_config_.weathersit_var)),
    ("yr_mapper", Mapper(variables=config.model_config_.yr_var, mappings=config.model_config_.yr_mappings)),
    ("mnth_mapper", Mapper(variables=config.model_config_.mnth_var, mappings=config.model_config_.mnth_mappings)),
    ("season_mapper", Mapper(variables=config.model_config_.season_var, mappings=config.model_config_.season_mappings)),
    ("weathersit_mapper", Mapper(variables=config.model_config_.weathersit_var, mappings=config.model_config_.weathersit_mappings)),
    ("holiday_mapper", Mapper(variables=config.model_config_.holiday_var, mappings=config.model_config_.holiday_mappings)),
    ("workingday_mapper", Mapper(variables=config.model_config_.workingday_var, mappings=config.model_config_.workingday_mappings)),
    ("hr_mapper", Mapper(variables=config.model_config_.hr_var, mappings=config.model_config_.hr_mappings)),
#    ("outlier_handler", outlier_handler),
    ("weekday_encoder", WeekdayOneHotEncoder(handle_unknown=config.model_config_.handle_unknown_var)),
    ("drop_dteday", DropColumns(columns=config.model_config_.dteday_var)),
    ("scaler", StandardScaler()),
    ("model_rf", RandomForestClassifier(n_estimators=config.model_config_.n_estimators, 
                                        max_depth=config.model_config_.max_depth, 
                                        random_state=config.model_config_.random_state))
])