"""Tune hyperparameters of model algorithms.

Best parameters are saved to a yaml-file which is loaded in final train_eval
"""

from pathlib import Path
from pprint import pformat

import optuna
import pandas as pd
import yaml
from rich.console import Console
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import ShuffleSplit, cross_validate

from src.utility import train_features_manual_domain

# CONSTANTS
#########################################################

# Choose model, feature and label to optimize
MODEL_TO_OPTIMIZE = "extra_trees"  # random_forest knn
TRAINING_FEATURES = train_features_manual_domain
LABEL = "Grouting time"  # Total grout take

MODEL_SEED = 0
CV_SPLITS = 4
SPLITTING_SEED = 0

# TODO: get all default params from some algorithms using a prebuilt library, hydra-zen?


def suggest_hyperparameters(trial: optuna.trial.Trial, algorithm: str) -> dict:
    """Hyperparameter suggestions for optuna optimization for a chosen algorithm."""
    match algorithm:
        case "extra_trees":
            params = {
                "n_estimators": trial.suggest_int(
                    name="n_estimators", low=10, high=800, step=10
                ),
                "max_depth": trial.suggest_categorical(
                    "max_depth", [None, 1, 2, 4, 8, 16, 32, 64]
                ),
                "min_samples_split": trial.suggest_int(
                    "min_samples_split", 2, 14, step=2
                ),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", None, 1.0]
                ),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf", 1, 10, step=1
                ),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                "criterion": trial.suggest_categorical(
                    "criterion",
                    ["poisson", "squared_error", "friedman_mse", "absolute_error"],
                ),
            }

        case "knn":
            params = dict(
                n_neighbors=trial.suggest_int("n_neighbors", low=1, high=20, step=1),
                weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
                algorithm=trial.suggest_categorical(
                    "algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
                ),
                p=trial.suggest_int("p", low=1, high=2, step=1),
                leaf_size=trial.suggest_int("leaf_size", low=5, high=50, step=5),
                n_jobs=-1,
            )
        case _:
            raise ValueError(f"{algorithm} is not implemented. Only: knn, extra_trees")
    return params


def objective(trial: optuna.trial.Trial):
    console = Console()
    # get data
    path_model_ready_data_longholes = Path(
        "./data/model_ready/model_ready_dataset_longholes.xlsx"
    )
    df = pd.read_excel(path_model_ready_data_longholes, header=0, index_col=0)
    df = df.sample(frac=1, random_state=42)
    features = df[TRAINING_FEATURES]
    labels = df[LABEL]
    features_encoded = pd.get_dummies(features)

    # define params
    suggested_hyperparams = suggest_hyperparameters(trial, MODEL_TO_OPTIMIZE)

    console.print(f"\nSuggested hyperparameters: \n{pformat(trial.params)}")

    # model define
    clf = ExtraTreesRegressor(
        verbose=False, random_state=MODEL_SEED, n_jobs=-1, **suggested_hyperparams
    )

    # model fit

    metrics = {
        "neg_mean_squared_error": "neg_mean_squared_error",
        "neg_root_mean_squared_error": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    splitter = ShuffleSplit(
        n_splits=CV_SPLITS, test_size=0.25, random_state=SPLITTING_SEED
    )
    scores = cross_validate(
        clf,
        features_encoded,
        labels,
        cv=splitter,
        n_jobs=-1,
        scoring=metrics,
        return_train_score=True,
        return_estimator=True,
    )

    # return score
    r2 = scores["test_r2"]
    mse = scores["test_neg_mean_squared_error"]
    rmse = scores["test_neg_root_mean_squared_error"]

    metrics = dict(r2=r2.mean(), mse=mse.mean(), rmse=rmse.mean())

    console.print(
        f"R2 mean +/- std. dev.: \t" f"{r2.mean():.3f} +/- " f"{r2.std():.3f}"
    )
    console.print(
        f"MSE mean +/- std. dev.: \t" f"{mse.mean():.3f} +/- " f"{mse.std():.3f}"
    )
    console.print(
        f"RMSE. mean +/- std. dev.: \t" f"{rmse.mean():.3f} +/- " f"{rmse.std():.3f}\n"
    )
    # return r2.mean()
    return rmse.mean()


def optimize(direction="maximize", metric="r2"):
    console = Console()

    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(
        study_name="total_grout_take",
        direction=direction,
        sampler=sampler,
    )

    study.optimize(objective, n_trials=50)

    console.rule("Study statistics")
    console.print("Number of finished trials: ", len(study.trials))

    console.print("Best trial:")
    trial = study.best_trial

    console.print("Trial number: \t", trial.number)
    console.print("R2: \t", trial.value)

    console.print("Params: ")
    for key, value in trial.params.items():
        console.print(f"{key}: {value}")

    label_short = ("_").join(LABEL.split(" "))

    path_params = Path(
        "./src/config", f"best_params_{MODEL_TO_OPTIMIZE}_{metric}_{label_short}.yaml"
    )
    with open(path_params, "w") as file:
        yaml.dump(trial.params, file)


if __name__ == "__main__":
    optimize(direction="maximize", metric="rmse")
