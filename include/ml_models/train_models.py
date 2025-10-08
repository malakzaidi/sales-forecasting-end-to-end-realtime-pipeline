import yaml
import pandas as pd

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








