import os
import yaml
import logging
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

from service_discovery import get_minio_endpoint,get_mlflow_endpoint

logger = logging.getLogger(__name__)


class MlflowManager:
    def __init__(self, config_path: str="/usr/local/airflow/include/config/ml_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        mlflow_config = self.config["mlflow"]
        self.tracking_uri = get_mlflow_endpoint()

        self.experiment_name = mlflow_config["experiment_name"]
        self.registry_name = mlflow_config["registry_name"]

        mlflow.set_tracking_uri(self.tracking_uri)

        try:
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            logger.warning(f"Failed to set experiment {self.experiment_name} to {e}")
            if "mlflow" in self.tracking_uri:
                self.tracking_uri = "http://localhost:5001"
                mlflow.set_tracking_uri(self.tracking_uri)
                os.environ["MLFLOW_TRACKING_URI"] = self.tracking_uri
                logger.info(f"Retrying with localhost : {self.tracking_uri}")
                try:
                    mlflow.set_experiment(self.experiment_name)
                except Exception as e2:
                    logger.warning(f"Failed to connect to Mlflow : {e2}")

        os.environ["MLFLOW_S3_ENDPOINT_URL"] = get_minio_endpoint()
        os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID","minioadmin")
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY","minioadmin")

        self.client = MlflowClient(tracking_uri=self.tracking_uri)


