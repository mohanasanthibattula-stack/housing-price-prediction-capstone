"""Model pipelines and hyperparameter search configurations."""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from scipy.stats import loguniform, uniform, randint

def get_pipelines():
    """Construct sklearn pipelines for all 4 candidate model families."""
    pipelines = {
        "Dummy": Pipeline([
            ("regressor", DummyRegressor(strategy="mean"))
        ]),
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge(random_state=42))
        ]),
        "RandomForest": Pipeline([
            ("regressor", RandomForestRegressor(random_state=42, n_jobs=-1))
        ]),
        "HistGradientBoosting": Pipeline([
            ("regressor", HistGradientBoostingRegressor(random_state=42))
        ])
    }
    return pipelines

def get_param_distributions():
    """Return lean hyperparameter distributions for RandomizedSearchCV."""
    param_dists = {
        "Dummy": {},
        "Ridge": {
            "regressor__alpha": loguniform(1e-3, 1e3)
        },
        "RandomForest": {
            "regressor__n_estimators": randint(50, 150),
            "regressor__max_depth": randint(10, 25),
            "regressor__min_samples_split": randint(2, 10),
            "regressor__max_features": uniform(0.5, 0.5)
        },
        "HistGradientBoosting": {
            "regressor__learning_rate": loguniform(0.02, 0.2),
            "regressor__max_iter": randint(100, 250),
            "regressor__max_leaf_nodes": randint(20, 60),
            "regressor__min_samples_leaf": randint(10, 40),
            "regressor__l2_regularization": loguniform(1e-3, 10.0)
        }
    }
    return param_dists
