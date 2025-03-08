from typing import List
import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class WeekdayImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weekday' column by extracting dayname from 'dteday' column """

    def __init__(self, variables: str, fill_value: str = 'Monday'):

        if not isinstance(variables, str):
            raise ValueError("variables should be a list")

        self.variables = variables
        self.fill_value = fill_value

    def fit(self, X, y=None):
        return self  # Nothing to fit

    def transform(self, X):
        X_ = X.copy()  # Create a copy to avoid modifying the original DataFrame

        # Find missing values and their indices
        missing_indices = X_[X_['weekday'].isnull()].index

        # Extract day names from 'dteday' for missing indices
        day_names = pd.to_datetime(X_.loc[missing_indices, 'dteday']).dt.day_name()

        # Impute missing values with extracted day names (first 3 letters)
        X_.loc[missing_indices, 'weekday'] = day_names.str[:3]

        return X_

class WeathersitImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weathersit' column by replacing them with the most frequent category value """

    def __init__(self, variables: str, fill_value: str = 'Clear'):
        self.variables = variables
        self.fill_value = fill_value

    def fit(self, X, y=None):
        self.fill_value = X[self.variables].mode()[0]  # Get most frequent
        return self

    def transform(self, X):
        X = X.copy()
        X[self.variables] = X[self.variables].fillna(self.fill_value)
        return X
    

class Mapper(BaseEstimator, TransformerMixin):
    """
    Ordinal categorical variable mapper:
    Treat column as Ordinal categorical variable, and assign values accordingly
    """

    def __init__(self, variables, mappings):
        """
        Initialize the Mapper with the variable name.

        Args:
            variable (str): The name of the variable to be mapped.
        """
        if not isinstance(variables, str):
            raise ValueError("variable should be a string")
        self.bounds = {}
        self.variables = variables
        self.mappings = mappings

    def fit(self, X, y=None):
        """
        Fit the Mapper by learning the mapping from unique values to integers.

        Args:
            X (pd.DataFrame): The input DataFrame.
            y (Any, optional): Ignored. Defaults to None.
            mapping (dict, optional): A dictionary specifying the mapping. If None,
                                        the mapping is learned from the data. Defaults to None.

        Returns:
            self: Returns the fitted Mapper object.
        """
        
        return self

    def transform(self, X):
        """
        Transform the input DataFrame by applying the mapping.

        Args:
            X (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The transformed DataFrame with the mapped values.
        """
        X = X.copy()
        X[self.variables] = X[self.variables].map(self.mappings)
        return X
    
#class OutlierHandler(BaseEstimator, TransformerMixin):
#    def __init__(self, variables):
#        if not isinstance(variables, list):
#            raise ValueError('variables should be a list')
#        self.variables = variables
#        self.bounds = {}

#    def fit(self, X, y=None):
#        self.bounds = {}
#        for feature in self.variables:
#            X[feature] = X[feature].astype(float)  # Convert to float
#            Q1 = X[feature].quantile(0.25)
#            Q3 = X[feature].quantile(0.75)
#            IQR = Q3 - Q1
#            self.bounds[feature] = {
#                'lower': Q1 - 1.5 * IQR,
#                'upper': Q3 + 1.5 * IQR
#            }
#        return self

#    def transform(self, X):
#        X = X.copy()
#        for feature in self.variables:
#            X[feature] = X[feature].astype(float)  # Convert to float
#            lower_bound = self.bounds[feature]['lower']
#            upper_bound = self.bounds[feature]['upper']
#            X[feature] = np.clip(X[feature], lower_bound, upper_bound)
#        return X

class WeekdayOneHotEncoder(BaseEstimator, TransformerMixin):
    """ One-hot encode weekday column """

    def __init__(self, handle_unknown='ignore'):
        self.handle_unknown = handle_unknown
        self.encoder = OneHotEncoder(handle_unknown=self.handle_unknown, sparse_output=False)

    def fit(self, X, y=None):
        self.encoder.fit(X[['weekday']])
        return self

    def transform(self, X):
        X_encoded = self.encoder.transform(X[['weekday']])  # Transform the 'weekday' column
        X_encoded_df = pd.DataFrame(X_encoded, columns=self.encoder.get_feature_names_out(['weekday']))
        X.reset_index(drop=True, inplace=True)
        X_encoded_df.reset_index(drop=True, inplace=True)
        X = pd.concat([X, X_encoded_df], axis=1)
        X.drop(columns=['weekday'], inplace=True)
        return X
    

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Create a copy to avoid modifying the original DataFrame
        X_ = X.copy()
        return X_.drop(columns=self.columns)
