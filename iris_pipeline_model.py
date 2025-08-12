from iris.model import hp_tune
from iris.log import CustomLogger

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from pyspark.sql import SparkSession
from hyperopt import hp
from pandas import concat
from numpy import argmax
import argparse


def iris_pipeline_model(
    catalog: str,
    schema: str,
    base_train_table_name: str = "table_iris_train",
    base_val_table_name: str = "table_iris_val",
    base_test_table_name: str = "table_iris_test",
    max_evals: int = 3,
    logging_level: str = "INFO"
) -> None:
    """
    Runs hyperparameter tuning for Random Forest and Logistic Regression on the iris dataset
    stored in Unity Catalog tables, trains final ensemble model on train+val data,
    and evaluates on test data. Logs progress and results with CustomLogger.

    Args:
        catalog: Unity Catalog catalog name.
        schema: Unity Catalog schema name.
        base_train_table_name: Base train table name prefix (without _x/_y).
        base_val_table_name: Base validation table name prefix.
        base_test_table_name: Base test table name prefix.
        max_evals: Number of hyperopt iterations for tuning.
        logging_level: Logger level (e.g., DEBUG, INFO).
    """

    logger = CustomLogger(name="IrisPipelineModel", level=logging_level)
    logger.info("Starting Iris pipeline hyperparameter tuning and final evaluation.")

    spark = SparkSession.builder.appName("IrisPipelineModel").getOrCreate()

    # Define search spaces
    rf_space = {
        'n_estimators': hp.choice('n_estimators', range(50, 301, 10)),
        'max_depth': hp.choice('max_depth', range(5, 31)),
        'min_samples_split': hp.uniform('min_samples_split', 0.01, 0.3),
        'min_samples_leaf': hp.uniform('min_samples_leaf', 0.01, 0.3),
        'bootstrap': hp.choice('bootstrap', [True, False])
    }

    lr_space = {
        'penalty': hp.choice('penalty', ['l2', 'none']),
        'C': hp.loguniform('C', -4, 4),
        'solver': hp.choice('solver', ['saga', 'liblinear', 'lbfgs', 'newton-cg', 'sag']),
        'l1_ratio': hp.uniform('l1_ratio', 0, 1)  # only used if penalty is elasticnet (not used here)
    }

    # Hyperparameter tuning RandomForest
    logger.info("Starting hyperparameter tuning for RandomForestClassifier...")
    best_rf_results = hp_tune(
        spark=spark,
        catalog=catalog,
        schema=schema,
        base_train_table_name=base_train_table_name,
        base_val_table_name=base_val_table_name,
        model_class=RandomForestClassifier,
        hyperopt_space=rf_space,
        max_evals=max_evals,
        scoring_function=accuracy_score,
        maximize=True,
        logger=logger,
    )
    logger.info(f"Best RandomForest hyperparameters: {best_rf_results}")

    # Hyperparameter tuning LogisticRegression
    logger.info("Starting hyperparameter tuning for LogisticRegression...")
    best_lr_results = hp_tune(
        spark=spark,
        catalog=catalog,
        schema=schema,
        base_train_table_name=base_train_table_name,
        base_val_table_name=base_val_table_name,
        model_class=LogisticRegression,
        hyperopt_space=lr_space,
        max_evals=max_evals,
        scoring_function=accuracy_score,
        maximize=True,
        logger=logger,
    )
    logger.info(f"Best LogisticRegression hyperparameters: {best_lr_results}")

    # Load train and val features and targets, convert to pandas
    train_x = spark.table(f"{catalog}.{schema}.{base_train_table_name}_x").toPandas()
    train_y = spark.table(f"{catalog}.{schema}.{base_train_table_name}_y").toPandas().iloc[:, 0]
    val_x = spark.table(f"{catalog}.{schema}.{base_val_table_name}_x").toPandas()
    val_y = spark.table(f"{catalog}.{schema}.{base_val_table_name}_y").toPandas().iloc[:, 0]

    X_train_val = concat([train_x, val_x], ignore_index=True)
    y_train_val = concat([train_y, val_y], ignore_index=True)

    logger.info("Loaded and combined train + val datasets.")

    # Load test set
    test_x = spark.table(f"{catalog}.{schema}.{base_test_table_name}_x").toPandas()
    test_y = spark.table(f"{catalog}.{schema}.{base_test_table_name}_y").toPandas().iloc[:, 0]

    logger.info("Loaded test dataset.")

    # Train final models on combined train+val set
    rf_model = RandomForestClassifier(random_state=42, **best_rf_results["params"])
    rf_model.fit(X_train_val, y_train_val)
    logger.info("RandomForest trained on combined train+val dataset.")

    lr_model = LogisticRegression(random_state=42, max_iter=1000, **best_lr_results["params"])
    lr_model.fit(X_train_val, y_train_val)
    logger.info("LogisticRegression trained on combined train+val dataset.")

    # Predict probabilities for test
    prob_rf = rf_model.predict_proba(test_x)
    prob_lr = lr_model.predict_proba(test_x)

    # Average predicted probabilities for ensemble prediction
    avg_proba = (prob_rf + prob_lr) / 2.0

    # Predict class for each sample
    y_pred_ensemble_idx = argmax(avg_proba, axis=1)

    # Map indices back to actual labels if test_y uses string labels
    if test_y.dtype.name == 'category' or test_y.dtype == object:
        classes = rf_model.classes_
        y_pred_ensemble = classes[y_pred_ensemble_idx]
    else:
        y_pred_ensemble = y_pred_ensemble_idx

    acc = accuracy_score(test_y, y_pred_ensemble)
    logger.info(f"Ensemble model accuracy on test set: {acc:.4f}")

    logger.info("Iris pipeline completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Iris pipeline hyperparameter tuning and evaluation")

    parser.add_argument("--catalog", type=str, default="workspace", help="Unity Catalog catalog name")
    parser.add_argument("--schema", type=str, default="schema_iris", help="Unity Catalog schema name")
    parser.add_argument("--base_train_table_name", type=str, default="table_iris_train", help="Base training table name prefix")
    parser.add_argument("--base_val_table_name", type=str, default="table_iris_val", help="Base validation table name prefix")
    parser.add_argument("--base_test_table_name", type=str, default="table_iris_test", help="Base test table name prefix")
    parser.add_argument("--max_evals", type=int, default=3, help="Max evaluations for hyperopt tuning")
    parser.add_argument("--logging_level", type=str, default="INFO", help="Logging level (DEBUG, INFO, etc.)")

    args = parser.parse_args()

    iris_pipeline_model(
        catalog=args.catalog,
        schema=args.schema,
        base_train_table_name=args.base_train_table_name,
        base_val_table_name=args.base_val_table_name,
        base_test_table_name=args.base_test_table_name,
        max_evals=args.max_evals,
        logging_level=args.logging_level,
    )
