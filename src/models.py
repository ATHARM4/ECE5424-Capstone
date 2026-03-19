
from typing import Iterable
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# baseline model - logistic regression using "pipelining"
def train_logistic_regression( X_train, y_train, numeric_cols: Iterable[str], categorical_cols: Iterable[str],) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, list(numeric_cols)),
            ("cat", categorical_transformer, list(categorical_cols)),
        ]
    )

    clf = LogisticRegression(max_iter=1000, n_jobs=1)
    model = Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])
    model.fit(X_train, y_train)
    return model

# need to add random forest as well (at least)
# def train_random_forest()