from datetime import datetime

import yaml
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler

from utils.mlflow_utils import MlflowManager
from data_validation.validators import DataValidator
from feature_engineering.feature_pipeline import FeatureEngineer
from typing import Optional,List

import logging
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, config_path: str = 'usr/local/airflow/include/config/ml_config.yaml' ):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model_config = self.config["models"]
        self.model_training = self.model_config["training"]
        self.mlflow_manager = MlflowManager(config_path)
        self.feature_engineer = FeatureEngineer(config_path)
        self.data_validator = DataValidator(config_path)

        self.models = {}
        self.scalers = {}
        self.encoders = {}

    def prepare_data(self,df: pd.DataFrame, target_col:str= "sales",
                     date_col: str = "date", group_cols: Optional[List[str]] = None,
                     categorial_cols: Optional[List[str]] = None):
        logger.info(f"Preparing data for training")

        required_cols = ["date" , target_col]
        if group_cols:
            required_cols.extend(group_cols)

        missing_cols = set(required_cols)- set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        df_features = self.feature_engineer.create_all_features(
            df, target_col=target_col,
            date_col=date_col, group_cols=group_cols,
            categorical_cols=categorial_cols )

        # split data chronologically for the time series
        df_sorted = df_features.sort_values(by=date_col)

        train_size = int(len(df_sorted) * (1 - self.training_config["test_size"] - self.training_config["validation_size"]))
        val_size = int(len(df_sorted) * self.training_config["validation_size"])

        train_df = df_sorted[:train_size]
        val_df = df_sorted[train_size: train_size+val_size]
        test_df = df_sorted[train_size+val_size:]

        train_df = train_df.dropna(subset=[target_col])
        val_df = val_df.dropna(subset=[target_col])
        test_df = test_df.dropna(subset=[target_col])

        logger.info(f"Train data split: {len(train_df)}, Validation data split: {len(val_df)}, Test data split: {len(test_df)} ")
        return train_df, val_df, test_df

    def preprocess_features(self,train_df: pd.DataFrame, val_df: pd.DataFrame,
                        test_df: pd.DataFrame, target_col:str,
                        exclude_cols: List[str] = ["date"] ):
        logger.info(f"Preprocessing features")
        feature_cols= [col for col in train_df.columns if col not in exclude_cols + [target_col]]

        X_train = train_df[feature_cols].copy()
        X_val = val_df[feature_cols].copy()
        X_test = test_df[feature_cols].copy()

        y_train=train_df[target_col].values()
        y_val=val_df[target_col].values()
        y_test=test_df[target_col].values()

        #encode categorial variables
        categorial_cols = X_train.select_dtypes(include=["object"]).columns
        for col in categorial_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                X_train.loc[:,col] = self.encoders[col].fit_transform(X_train[col].astype(str))
            else:
                X_train.loc[:,col] = self.encoders[col].transform(X_train[col].astype(str))



        X_val.loc[:,col]=self.encoders[col].transform(X_val[col].astype(str))
        X_test.loc[:,col] = self.encoders[col].transform(X_test[col].astype(str))


        #scale numerical features
        scaler = StandardScaler()
        X_train_scaled=scaler.fit_transform(X_train)
        X_val_scaled=scaler.transform(X_val)
        X_test_scaled=scaler.transform(X_test)

        X_train_scaled=pd.DataFrame(X_train_scaled,columns=feature_cols,index=X_train.index)
        X_val_scaled=pd.DataFrame(X_val_scaled,columns=feature_cols,index=X_train.index)
        X_test_scaled=pd.DataFrame(X_test_scaled,columns=feature_cols,index=X_test.index)

        self.scalers["standard"] = scaler
        self.feature_cols=feature_cols

        return X_train_scaled, X_val_scaled, X_test_scaled , y_train,y_val,y_test











