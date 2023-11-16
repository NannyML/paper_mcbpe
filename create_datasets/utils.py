import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


from typing import List
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import json
import boto3

from jupyter_server import serverapp
from jupyter_server.utils import url_path_join
from pathlib import Path
import re
import requests

RANDOM_STATE = 3

class LogisticRegressionWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features, continuous_features):
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.classifier = LogisticRegression(random_state=RANDOM_STATE, max_iter=200)
        pipe = Pipeline([
                ('imputer_categorical', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
            ])
        self.column_transformer = ColumnTransformer([
            ('imputer_continuous', SimpleImputer(strategy='mean'), self.continuous_features),
            ('categorical', pipe, self.categorical_features),
        ])

    def fit(self, X, y):
        X_transformed = self.column_transformer.fit_transform(X)
        self.classifier.fit(X_transformed, y)
        return self

    def predict(self, X):
        X_transformed = self.column_transformer.transform(X)
        return self.classifier.predict(X_transformed)

    def predict_proba(self, X):
        X_transformed = self.column_transformer.transform(X)
        return self.classifier.predict_proba(X_transformed)


class RandomForestClassifierWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features, continuous_features):
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.classifier = RandomForestClassifier(random_state=RANDOM_STATE)

        pipe = Pipeline([
                ('imputer_categorical', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
            ])
        self.column_transformer = ColumnTransformer([
            ('imputer_continuous', SimpleImputer(strategy='mean'), self.continuous_features),
            ('categorical', pipe, self.categorical_features),
        ])

    def fit(self, X, y):
        X = self.column_transformer.fit_transform(X)
        self.classifier.fit(X, y)
        return self

    def predict(self, X):
        X = self.column_transformer.transform(X)
        return self.classifier.predict(X)

    def predict_proba(self, X):
        X = self.column_transformer.transform(X)
        return self.classifier.predict_proba(X)

class LGBMClassifierWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features, continuous_features):
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.lgbm_classifier = LGBMClassifier(random_state=RANDOM_STATE)
        self.column_transformer = None

    def fit(self, X, y, sample_weight=None):
        pipe = Pipeline(
            [('ordinal_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))]
        )
        self.column_transformer = ColumnTransformer(
            [('categorical', pipe, self.categorical_features),],
            remainder='passthrough',
            verbose_feature_names_out=False
        )
        X_transformed = self.column_transformer.fit_transform(X)
        features_out = list(self.column_transformer.get_feature_names_out())
        df_X_transformed = pd.DataFrame(X_transformed, columns = features_out)

        if sample_weight is None:
            self.lgbm_classifier.fit(
                df_X_transformed,
                y,
                feature_name=features_out,
                categorical_feature=self.categorical_features
            )
        else:
            self.lgbm_classifier.fit(
                df_X_transformed,
                y,
                sample_weight = sample_weight,
                feature_name=features_out,
                categorical_feature=self.categorical_features
            )

        self.features_out = features_out
        return self

    def predict(self, X):
        X_transformed = self.column_transformer.transform(X)
        df_X_transformed = pd.DataFrame(X_transformed, columns = self.features_out)
        return self.lgbm_classifier.predict(df_X_transformed)

    def predict_proba(self, X):
        X_transformed = self.column_transformer.transform(X)
        df_X_transformed = pd.DataFrame(X_transformed, columns = self.features_out)
        return self.lgbm_classifier.predict_proba(df_X_transformed)


US_STATE_LIST = [
    'AL', # Alabama
 	'AK', # Alaska
 	'AZ', # Arizona
 	'AR', # Arkansas
 	'CA', # California
 	'CO', # Colorado
 	'CT', # Connecticut
 	'DE', # Delaware
 	'FL', # Florida
 	'GA', # Georgia
 	'HI', # Hawaii
 	'ID', # Idaho
 	'IL', # Illinois
 	'IN', # Indiana
 	'IA', # Iowa
 	'KS', # Kansas
    'KY', # Kentucky
 	'LA', # Louisiana
 	'ME', # Maine
 	'MD', # Maryland
    'MA', # Massachusetts
 	'MI', # Michigan
 	'MN', # Minnesota
 	'MS', # Mississippi
 	'MO', # Missouri
 	'MT', # Montana
 	'NE', # Nebraska
 	'NV', # Nevada
 	'NH', # New Hampshire
 	'NJ', # New Jersey
 	'NM', # New Mexico
 	'NY', # New York
 	'NC', # North Carolina
 	'ND', # North Dakota
 	'OH', # Ohio
 	'OK', # Oklahoma
 	'OR', # Oregon
    'PA', # Pennsylvania
 	'RI', # Rhode Island
 	'SC', # South Carolina
 	'SD', # South Dakota
 	'TN', # Tennessee
 	'TX', # Texas
 	'UT', # Utah
 	'VT', # Vermont
    'VA', # Virginia
 	'WA', # Washington
 	'WV', # West Virginia
 	'WI', # Wisconsin
 	'WY', # Wyoming
]


MODELS_LIST = [
    [LogisticRegressionWrapper, "LogisticRegression"],
    [LGBMClassifierWrapper, "LGBMClassifier"],
    [RandomForestClassifierWrapper, "RandomForestClassifier"]
]
