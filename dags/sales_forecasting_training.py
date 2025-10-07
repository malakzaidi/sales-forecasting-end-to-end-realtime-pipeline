from datetime import datetime, timedelta

import logger
import pandas as pd
from airflow.decorators import dag, task
import os
import sys

from include.utils.data_generator import RealisticSalesDataGenerator

# Add include path
sys.path.append("/usr/local/airflow/include")



default_args = {
    "owner": "data-team",
    "depends_on_past": False,
    "start_date": datetime(2025, 7, 22),
    "email_on_failure": True,
    "email_on_retry": False,
    "email": ["admin@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    schedule="@weekly",
    start_date=datetime(2025, 7, 22),
    catchup=False,
    default_args=default_args,
    description="Train sales forecasting models",
    tags=["ml", "training", "sales"],
)
def sales_forecast_training():
    @task()
    def extract_data_task():

        data_output_dir = "/tmp/sales_data"
        generator = RealisticSalesDataGenerator(
            start_date="2021-01-01", end_date="2021-12-31"
        )
        print("Generating realistic sales data...")
        file_paths = generator.generate_sales_data(output_dir=data_output_dir)
        total_files = sum(len(paths) for paths in file_paths.values())
        print(f"Generated {total_files} files:")
        for data_type, paths in file_paths.items():
            print(f"  - {data_type}: {len(paths)} files")
        return {
            "data_output_dir": data_output_dir,
            "file_paths": file_paths,
            "total_files": total_files,
        }

    @task()
    def validate_data_task(extract_result):
        import glob

        file_paths = extract_result["file_paths"]
        total_rows = 0
        issues_found = []
        print(f"Validating {len(file_paths['sales'])} sales files...")
        for i, sales_file in enumerate(file_paths["sales"][:10]):
            df = pd.read_parquet(sales_file)
            if i == 0:
                print(f"Sales data columns: {df.columns.tolist()}")
            if df.empty:
                issues_found.append(f"Empty file: {sales_file}")
                continue
            required_cols = [
                "date",
                "store_id",
                "product_id",
                "quantity_sold",
                "revenue",
            ]
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                issues_found.append(f"Missing columns in {sales_file}: {missing_cols}")
            total_rows += len(df)
            if df["quantity_sold"].min() < 0:
                issues_found.append(f"Negative quantities in {sales_file}")
            if df["revenue"].min() < 0:
                issues_found.append(f"Negative revenue in {sales_file}")
        for data_type in ["promotions", "store_events", "customer_traffic"]:
            if data_type in file_paths and file_paths[data_type]:
                sample_file = file_paths[data_type][0]
                df = pd.read_parquet(sample_file)
                print(f"{data_type} data shape: {df.shape}")
                print(f"{data_type} columns: {df.columns.tolist()}")
        validation_summary = {
            "total_files_validated": len(file_paths["sales"][:10]),
            "total_rows": total_rows,
            "issues_found": len(issues_found),
            "issues": issues_found[:5],
        }
        if issues_found:
            print(f"Validation completed with {len(issues_found)} issues:")
            for issue in issues_found[:5]:
                print(f"  - {issue}")
        else:
            print(f"Validation passed! Total rows: {total_rows}")
        return validation_summary
    @task
    def train_models_task(validation_summary):
        file_paths = validation_summary["file_paths"]
        logger.info(f"Training models ...")
        sales_df=[]
        max_files=50
        for i,sales_file in enumerate(file_paths["sales"][:max_files]):
            df = pd.read_parquet(sales_file)
            sales_df.append(df)
            if (i+1) % 10 == 0:
                logger.info(f"loaded {i+1} sales files")

        sales_df = pd.concat(sales_df,ignore_index=True)
        print(f"Combined sales data shape: {sales_df.shape}")

        daily_sales = (
            sales_df.groupby(["store_id", "product_id", "category"])
            .agg({"quantity_sold": "sum",
                  "revenue": "sum",
                  "cost": "sum",
                  "profit" : "sum",
                  "discount_percent":"mean",
                  "unit_cost": "mean",
                })
            .reset_index()
        )

        daily_sales = daily_sales.rename(columns={"revenue":"sales"})
        if file_paths.get("promotions"):
            promo_df = pd.read_parquet(file_paths["promotions"][0])
            promo_summary = (
                promo_df.groupby(["date", "product_id"])["discount_percent"]
                .max()
                .reset_index())

            promo_summary["has_promotion"] = 1

            daily_sales = daily_sales.merge (
              promo_summary[("date", "product_id","has_promotion")],
              on=["date", "product_id"],
              how="left",
            )
            daily_sales["has_promotion"] = daily_sales["has_promotion"].fillna(0).astype(int)

        if file_paths.get("customer_traffic"):
            traffic_dfs = []
            for traffic_file in file_paths["customer_traffic"][:10]:
                traffic_dfs.append(pd.read_parquet(traffic_file))
            traffic_df = pd.concat(traffic_dfs,ignore_index=True)
            traffic_summary = (
                traffic_df.groupby(["date", "store_id"])
                    .agg({"customer_traffic": "sum","is_holiday":"max"})
                    .reset_index()
            )
            daily_sales = daily_sales.merge (
                traffic_summary,
                on=["date", "store_id"],
                how="left",
            )
        logger.info(f"Final training data shape:{daily_sales.shape}")
        logger.info(f"Columns:{daily_sales.columns.tolist}]")



    extract_result = extract_data_task()
    validation_summary = validate_data_task(extract_result)
    sales_forecast_training_dag = sales_forecast_training()