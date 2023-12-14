import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

RANDOM_STATE=42



class LGBMClassifierWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features, continuous_features):
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.lgbm_classifier = None
        self.column_transformer = None

    def fit(self, X, y, sample_weight=None):
    
    
        pipe = Pipeline([
                ('ordinal_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ])
    
        self.column_transformer = ColumnTransformer([
            ('categorical', pipe, self.categorical_features),
        ], remainder='passthrough', verbose_feature_names_out=False)

        X_transformed = self.column_transformer.fit_transform(X)
        
        features_out = list(self.column_transformer.get_feature_names_out())
                
        df_X_transformed = pd.DataFrame(X_transformed, columns = features_out)
        
        self.lgbm_classifier = LGBMClassifier(random_state=RANDOM_STATE)
        
        if sample_weight is None:
            self.lgbm_classifier.fit(X_transformed, y, 
                                     feature_name=features_out,  categorical_feature=self.categorical_features)
        else:
            self.lgbm_classifier.fit(X_transformed, y, sample_weight = sample_weight,
                                    feature_name=features_out,  categorical_feature=self.categorical_features)
        
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
