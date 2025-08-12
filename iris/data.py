from .log import CustomLogger

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import rand
from typing import Tuple, Optional
from pandas import DataFrame, Series
import re


def load_data(
    spark: SparkSession,
    catalog: str,
    schema: str,
    volume: str,
    file_name: str,
    table_name: str,
    header: bool = True,
    infer_schema: bool = True,
    sep: str = ",",
    mode: str = "overwrite",
    logger: CustomLogger | None = None,
) -> None:
    """
    Reads a CSV file from a specified Unity Catalog volume path and saves it as a Delta table.

    Args:
        spark (SparkSession): The active SparkSession.
        catalog (str): Unity Catalog catalog name.
        schema (str): Unity Catalog schema name.
        volume (str): Volume name where the CSV file is stored.
        file_name (str): Name of the CSV file to read.
        table_name (str): Name of the Unity Catalog table to save the DataFrame as.
        header (bool, optional): Whether the CSV file has a header row. Defaults to True.
        infer_schema (bool, optional): Whether to infer the schema automatically. Defaults to True.
        sep (str, optional): CSV separator character. Defaults to ",".
        mode (str, optional): Mode to save the table, e.g. 'overwrite' or 'append'. Defaults to "overwrite".
        logger (CustomLogger | None, optional): CustomLogger instance for logging. Defaults to None.

    Returns:
        None
    """
    # Initialize default logger if none provided
    if logger is None:
        logger = CustomLogger(name="load_data")

    logger.debug("Building paths for volume and table.")
    path_volume: str = f"/Volumes/{catalog}/{schema}/{volume}"
    path_table: str = f"{catalog}.{schema}.{table_name}"
    
    logger.debug(f"Reading CSV file from path: {path_volume}/{file_name}")
    df: DataFrame = spark.read.csv(
        f"{path_volume}/{file_name}",
        header=header,
        inferSchema=infer_schema,
        sep=sep,
    )
    logger.info(f"Successfully read CSV file into DataFrame with {df.count()} rows and {len(df.columns)} columns.")

    logger.debug(f"Saving DataFrame as Delta table at: {path_table} with mode={mode}")
    df.write.format("delta").mode(mode).saveAsTable(path_table)
    logger.info(f"DataFrame saved successfully to Delta table: {path_table}")


def process_data(
    spark: SparkSession,
    catalog: str,
    schema: str,
    source_table: str,
    id_column: str,
    target_table: str,
    seed: int = 42,
    logger: Optional[CustomLogger] = None,
) -> None:
    """
    Loads a Unity Catalog table, drops the specified ID column, randomizes the row order,
    and saves the processed DataFrame back as a Delta table to the catalog.

    Args:
        spark (SparkSession): The active SparkSession.
        catalog (str): Unity Catalog catalog name.
        schema (str): Unity Catalog schema name.
        source_table (str): Source Unity Catalog table name to read.
        id_column (str): Column name to drop before processing.
        target_table (str): Target Unity Catalog table name to save the processed data.
        seed (int, optional): Seed for random ordering to ensure reproducibility. Defaults to 42.
        logger (CustomLogger | None, optional): Logger instance for logging. Defaults to None.

    Returns:
        None
    """
    if logger is None:
        logger = CustomLogger(name="process_data")

    source_table_full = f"{catalog}.{schema}.{source_table}"
    target_table_full = f"{catalog}.{schema}.{target_table}"

    logger.info(f"Loading table '{source_table_full}' from Unity Catalog.")
    df = spark.table(source_table_full)

    logger.info(f"Dropping ID column '{id_column}'.")
    df_no_id = df.drop(id_column)

    # Transform camelCase column names to lower case with underscores
    new_columns = [
        re.sub(r'(?<!^)(?=[A-Z])', '_', col).lower()
        for col in df_no_id.columns
    ]
    df_no_id = df_no_id.toDF(*new_columns)

    logger.info(f"Randomizing row order with seed={seed}.")
    df_randomized = df_no_id.orderBy(rand(seed))

    logger.info(f"Processed DataFrame has {df_randomized.count()} rows and {len(df_randomized.columns)} columns.")

    logger.info(f"Saving processed DataFrame to '{target_table_full}' with overwrite mode.")
    df_randomized.write.format("delta").mode("overwrite").saveAsTable(target_table_full)

    logger.info(f"Processed data saved successfully to '{target_table_full}'.")


def train_val_test_split(
    spark: SparkSession,
    catalog: str,
    schema: str,
    source_table: str,
    train_table: str,
    val_table: str,
    test_table: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    logger: Optional["CustomLogger"] = None,
) -> None:
    """
    Loads a Unity Catalog table, splits it randomly into train, validation and test sets
    according to the given ratios, and saves all sets as Delta tables back to Unity Catalog.

    Args:
        spark (SparkSession): The active SparkSession.
        catalog (str): Unity Catalog catalog name.
        schema (str): Unity Catalog schema name.
        source_table (str): Source Unity Catalog table name to read.
        train_table (str): Target Unity Catalog table name for the training set.
        val_table (str): Target Unity Catalog table name for the validation set.
        test_table (str): Target Unity Catalog table name for the test set.
        train_ratio (float, optional): Fraction of rows for training set. Defaults to 0.7.
        val_ratio (float, optional): Fraction of rows for validation set. Defaults to 0.15.
        test_ratio (float, optional): Fraction of rows for test set. Defaults to 0.15.
        seed (int, optional): Seed for random split reproducibility. Defaults to 42.
        logger (CustomLogger|None, optional): Logger instance for logging. Defaults to None.

    Returns:
        None
    """
    if logger is None:
        logger = CustomLogger(name="train_val_test_split")

    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not abs(total_ratio - 1.0) < 1e-6:
        raise ValueError(f"Sum of train_ratio, val_ratio and test_ratio must be 1.0 but got {total_ratio}")

    source_table_full = f"{catalog}.{schema}.{source_table}"
    train_table_full = f"{catalog}.{schema}.{train_table}"
    val_table_full = f"{catalog}.{schema}.{val_table}"
    test_table_full = f"{catalog}.{schema}.{test_table}"

    logger.info(f"Loading table '{source_table_full}' from Unity Catalog.")
    df = spark.table(source_table_full)

    logger.info(
        f"Splitting data into train ({train_ratio*100:.1f}%), "
        f"validation ({val_ratio*100:.1f}%), and test ({test_ratio*100:.1f}%) sets with seed={seed}."
    )
    train_df, val_df, test_df = df.randomSplit([train_ratio, val_ratio, test_ratio], seed=seed)

    logger.info(f"Training set: {train_df.count()} rows, {len(train_df.columns)} columns.")
    logger.info(f"Validation set: {val_df.count()} rows, {len(val_df.columns)} columns.")
    logger.info(f"Test set: {test_df.count()} rows, {len(test_df.columns)} columns.")

    logger.info(f"Saving training set to '{train_table_full}'.")
    train_df.write.format("delta").mode("overwrite").saveAsTable(train_table_full)
    logger.info(f"Training set saved successfully to '{train_table_full}'.")

    logger.info(f"Saving validation set to '{val_table_full}'.")
    val_df.write.format("delta").mode("overwrite").saveAsTable(val_table_full)
    logger.info(f"Validation set saved successfully to '{val_table_full}'.")

    logger.info(f"Saving test set to '{test_table_full}'.")
    test_df.write.format("delta").mode("overwrite").saveAsTable(test_table_full)
    logger.info(f"Test set saved successfully to '{test_table_full}'.")


def features_and_target(
    spark: SparkSession,
    catalog: str,
    schema: str,
    table_name: str,
    logger: Optional[CustomLogger] = None,
) -> None:
    """
    Loads a Unity Catalog table as Spark DataFrame, extracts features and target directly using Spark,
    and saves X and y as separate Delta tables with suffixes '_x' and '_y'.

    Args:
        spark: Active Spark session.
        catalog: Unity Catalog catalog name.
        schema: Unity Catalog schema name.
        table_name: Unity Catalog table name to process.
        logger: Optional CustomLogger instance for logging.

    Returns:
        None
    """
    if logger is None:
        logger = CustomLogger(name="features_and_target")

    full_table_name = f"{catalog}.{schema}.{table_name}"
    x_table_name = f"{catalog}.{schema}.{table_name}_x"
    y_table_name = f"{catalog}.{schema}.{table_name}_y"

    logger.info(f"Loading table {full_table_name} from Unity Catalog...")
    df = spark.table(full_table_name)

    # Define feature and target columns in snake_case as per your data
    feature_columns = [
        "sepal_length_cm",
        "sepal_width_cm",
        "petal_length_cm",
        "petal_width_cm"
    ]
    target_column = "species"

    # Select features and target as separate DataFrames
    df_features: DataFrame = df.select(*feature_columns)
    df_target: DataFrame = df.select(target_column)

    logger.info(f"Saving features DataFrame as {x_table_name} (Delta table)...")
    df_features.write.format("delta").mode("overwrite").saveAsTable(x_table_name)
    logger.info(f"Features table saved successfully.")

    logger.info(f"Saving target DataFrame as {y_table_name} (Delta table)...")
    df_target.write.format("delta").mode("overwrite").saveAsTable(y_table_name)
    logger.info(f"Target table saved successfully.")
