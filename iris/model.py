from .log import CustomLogger
from pyspark.sql import SparkSession
from typing import Dict, Any, Optional, Type
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from datetime import datetime
import mlflow
import mlflow.sklearn


def hp_tune(
    spark: SparkSession,
    catalog: str,
    schema: str,
    base_train_table_name: str,  # base name for train (_x and _y suffixes)
    base_val_table_name: str,    # base name for val (_x and _y suffixes)
    model_class: Type[BaseEstimator],           # scikit-learn estimator class (e.g., RandomForestClassifier)
    hyperopt_space: Dict[str, Any],             # hyperopt search space dictionary for this model
    max_evals: int = 3,
    scoring_function = accuracy_score,          # function(y_true, y_pred) -> float; higher is better
    maximize: bool = True,                       # maximize or minimize scoring_function
    logger: Optional[CustomLogger] = None,
) -> Dict[str, Any]:
    """
    Generic hyperparameter tuning for any sklearn model using training and validation data,
    records trials with MLflow, and returns the best hyperparameters and MLflow model run references.

    Args:
        spark: SparkSession.
        catalog: Unity Catalog catalog name.
        schema: Unity Catalog schema name.
        base_train_table_name: Base train table name prefix.
        base_val_table_name: Base validation table name prefix.
        model_class: Scikit-learn classifier/regressor class (uninitialized).
        hyperopt_space: Hyperopt search space dictionary for the model's params.
        max_evals: Number of hyperopt iterations.
        scoring_function: Callable metric function; higher is better (e.g., accuracy_score).
        maximize: Whether to maximize (`True`) or minimize (`False`) the scoring function.
        logger: Optional CustomLogger instance for logging.

    Returns:
        Dict with keys:
          - "best_params": Dict with best params found via tuning.
          - "best_loss": Float validation loss of best run.
          - "best_run_info": Dict with MLflow run details for best trial.
          - "mlflow_runs": List of dicts with MLflow run details for all trials.
    """

    if logger is None:
        logger = CustomLogger(name="hyperparameter_tuning")

    train_x_table = f"{catalog}.{schema}.{base_train_table_name}_x"
    train_y_table = f"{catalog}.{schema}.{base_train_table_name}_y"
    val_x_table = f"{catalog}.{schema}.{base_val_table_name}_x"
    val_y_table = f"{catalog}.{schema}.{base_val_table_name}_y"

    logger.info(f"Loading training features from {train_x_table}")
    logger.info(f"Loading training target from {train_y_table}")
    logger.info(f"Loading validation features from {val_x_table}")
    logger.info(f"Loading validation target from {val_y_table}")

    # Load Spark tables
    df_train_x = spark.table(train_x_table)
    df_train_y = spark.table(train_y_table)
    df_val_x = spark.table(val_x_table)
    df_val_y = spark.table(val_y_table)

    # Convert to Pandas
    X_train = df_train_x.toPandas()
    y_train = df_train_y.toPandas().iloc[:, 0]  # Assume single target column
    X_val = df_val_x.toPandas()
    y_val = df_val_y.toPandas().iloc[:, 0]

    # Set MLflow registry URI if needed (adjust as per your environment)
    mlflow.set_registry_uri("databricks-uc")

    # Generate a timestamp string, e.g. "20250812_1950"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create experiment name incorporating the datetime
    experiment_path = f"/Users/y@shuyi.it/mlflow_experiments/hp_experiments_{timestamp}"

    # Set MLflow experiment with the timestamped name
    mlflow.set_experiment(experiment_path)

    mlflow_runs = []

    def objective(params):

        with mlflow.start_run(nested=True) as run:
            mlflow.log_params(params)
            model = model_class(**params)
            model.fit(X_train, y_train)

            preds_val = model.predict(X_val)
            score = scoring_function(y_val, preds_val)

            # Convert scoring to loss (to minimize)
            loss = -score if maximize else score

            mlflow.log_metric("val_score", score)
            mlflow.log_metric("val_loss", loss)
            mlflow.sklearn.log_model(model, artifact_path="model")

            run_info = {
                "run_id": run.info.run_id,
                "artifact_uri": run.info.artifact_uri,
                "params": params,
                "val_score": score,
                "val_loss": loss
            }
            mlflow_runs.append(run_info)

            logger.info(f"Trial params: {params} -> validation score: {score:.4f}, loss: {loss:.4f}")

            return {'loss': loss, 'status': STATUS_OK}

    trials = Trials()
    logger.info(f"Starting hyperparameter tuning (max_evals={max_evals}) for model {model_class.__name__}...")
    best = fmin(
        fn=objective,
        space=hyperopt_space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=None,
    )
    logger.info(f"Hyperparameter tuning completed. Best hyperparameters (raw): {best}")

    # Retrieve best run info by lowest loss
    best_loss = None
    best_run_info = None
    for run_info in mlflow_runs:
        if best_loss is None or run_info["val_loss"] < best_loss:
            best_loss = run_info["val_loss"]
            best_run_info = run_info

    return best_run_info

