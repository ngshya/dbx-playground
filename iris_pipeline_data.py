from iris.log import CustomLogger
from iris.data import load_data, process_data, train_val_test_split, features_and_target

from typing import Optional
from pyspark.sql import SparkSession
import os
import argparse


def iris_pipeline_data(
    catalog: str = "workspace",
    schema: str = "schema_iris",
    volume: str = "volume_iris",
    raw_csv_file_name: str = "Iris.csv",
    raw_table_name: str = "table_iris_raw",
    processed_table_name: str = "table_iris_processed",
    id_column_name: str = "id",
    train_table_name: str = "table_iris_train",
    val_table_name: str = "table_iris_val",                # Added val table name
    test_table_name: str = "table_iris_test",
    train_ratio: float = 0.7,                             # Default adjusted for train_val_test_split
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    csv_separator: str = ",",
    logging_level: str = "INFO"
) -> None:
    """
    Main script to run the data workflow:
    1. Load CSV data into a Unity Catalog Delta table.
    2. Process the data: drop ID column, rename columns, randomize rows, save result.
    3. Split processed data into train/val/test and save them as tables.
    4. Extract features and target from train, val, and test tables and save separately.

    Parameters can be overridden via function arguments.
    """
    # Initialize logger with logging_level param
    logger: CustomLogger = CustomLogger(name="IrisDataPipeline", level=logging_level)

    # Initialize SparkSession
    spark = SparkSession.builder.appName("IrisPipelineData").getOrCreate()

    # Step 1: Load raw CSV data into Unity Catalog Delta table
    logger.info("Starting data load from CSV to Delta table.")
    load_data(
        spark=spark,
        catalog=catalog,
        schema=schema,
        volume=volume,
        file_name=raw_csv_file_name,
        table_name=raw_table_name,
        header=True,
        infer_schema=True,
        sep=csv_separator,
        mode="overwrite",
        logger=logger,
    )

    # Step 2: Process raw table (drop ID, rename columns, randomize rows)
    logger.info("Starting data processing (drop ID, rename columns, randomize).")
    process_data(
        spark=spark,
        catalog=catalog,
        schema=schema,
        source_table=raw_table_name,
        id_column=id_column_name,
        target_table=processed_table_name,
        seed=random_seed,
        logger=logger,
    )

    # Step 3: Split processed data into train, val, and test sets and save them
    logger.info("Starting train/val/test split of processed data.")
    train_val_test_split(
        spark=spark,
        catalog=catalog,
        schema=schema,
        source_table=processed_table_name,
        train_table=train_table_name,
        val_table=val_table_name,
        test_table=test_table_name,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=random_seed,
        logger=logger,
    )

    # Step 4: Extract features and target from train, val, and test tables, save separately
    logger.info("Extracting features and target for train table.")
    features_and_target(
        spark=spark,
        catalog=catalog,
        schema=schema,
        table_name=train_table_name,
    )

    logger.info("Extracting features and target for validation table.")
    features_and_target(
        spark=spark,
        catalog=catalog,
        schema=schema,
        table_name=val_table_name,
    )

    logger.info("Extracting features and target for test table.")
    features_and_target(
        spark=spark,
        catalog=catalog,
        schema=schema,
        table_name=test_table_name,
    )

    logger.info("Data pipeline completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Iris data pipeline with optional overrides.")
    parser.add_argument("--catalog", type=str, default="workspace")
    parser.add_argument("--schema", type=str, default="schema_iris")
    parser.add_argument("--volume", type=str, default="volume_iris")
    parser.add_argument("--raw_csv_file_name", type=str, default="Iris.csv")
    parser.add_argument("--raw_table_name", type=str, default="table_iris_raw")
    parser.add_argument("--processed_table_name", type=str, default="table_iris_processed")
    parser.add_argument("--id_column_name", type=str, default="id")
    parser.add_argument("--train_table_name", type=str, default="table_iris_train")
    parser.add_argument("--val_table_name", type=str, default="table_iris_val")
    parser.add_argument("--test_table_name", type=str, default="table_iris_test")
    parser.add_argument("--train_ratio", type=float, default=0.5)
    parser.add_argument("--val_ratio", type=float, default=0.25)
    parser.add_argument("--test_ratio", type=float, default=0.25)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--csv_separator", type=str, default=",")
    parser.add_argument("--logging_level", type=str, default="INFO")

    args = parser.parse_args()

    iris_pipeline_data(
        catalog=args.catalog,
        schema=args.schema,
        volume=args.volume,
        raw_csv_file_name=args.raw_csv_file_name,
        raw_table_name=args.raw_table_name,
        processed_table_name=args.processed_table_name,
        id_column_name=args.id_column_name,
        train_table_name=args.train_table_name,
        val_table_name=args.val_table_name,
        test_table_name=args.test_table_name,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed,
        csv_separator=args.csv_separator,
        logging_level=args.logging_level,
    )