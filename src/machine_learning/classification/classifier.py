from xgboost import XGBClassifier
import streamlit as st
import pandas as pd
import typing as t
from sklearn.metrics import classification_report


def model_hash_func(model: XGBClassifier) -> t.Dict[str, t.Any]:
    return {key: val for key, val in model.__dict__.items() if key != "_Booster"}


class ClassificationManager:
    def __init__(self) -> None:
        pass

    @classmethod
    def get_params(cls):
        columns = st.columns(3)
        return {
            "max_depth": columns[0].slider(
                label="`max_depth`",
                min_value=1,
                max_value=10,
                value=3,
                step=1,
                help="Maximum depth of a tree. "
                "Increasing this value will make the model more complex and more likely to overfit. "
                "0 indicates no limit on depth.",
            ),
            "learning_rate": columns[0].slider(
                label="`learning_rate`",
                min_value=0.01,
                max_value=1.0,
                value=0.1,
                step=0.01,
                help="Step size shrinkage used in update to prevents overfitting. "
                "After each boosting step, we can directly get the weights of new features, and `learning_rate` shrinks the feature weights to make the boosting process more conservative.",
            ),
            "n_estimators": columns[0].slider(
                label="`n_estimators`",
                min_value=10,
                max_value=200,
                value=100,
                step=10,
                help="Number of gradient boosted trees. "
                "Equivalent to number of boosting rounds.",
            ),
            "subsample": columns[1].slider(
                label="`subsample`",
                min_value=0.1,
                max_value=1.0,
                value=0.8,
                step=0.05,
                help="Subsample ratio of the training instances. "
                "Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees and this will prevent overfitting. "
                "Subsampling will occur once in every boosting iteration.",
            ),
            "colsample_bytree": columns[1].slider(
                label="`colsample_bytree`",
                min_value=0.1,
                max_value=1.0,
                value=0.8,
                step=0.05,
                help="Subsample ratio of columns when constructing each tree. "
                "Subsampling occurs once for every tree constructed.",
            ),
            "min_split_loss": columns[1].slider(
                label="`min_split_loss`",
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=0.5,
                help="Minimum loss reduction required to make a further partition on a leaf node of the tree. "
                "The larger `min_split_loss` is, the more conservative the algorithm will be.",
            ),
            "min_child_weight": columns[2].slider(
                label="`min_child_weight`",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.5,
                help="Minimum sum of instance weight (hessian) needed in a child. "
                "If the tree partition step results in a leaf node with the sum of instance weight less than `min_child_weight`, then the building process will give up further partitioning. "
                "In linear regression task, this simply corresponds to minimum number of instances needed to be in each node. "
                "The larger `min_child_weight` is, the more conservative the algorithm will be.",
            ),
            "reg_alpha": columns[2].slider(
                label="`reg_alpha`",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.5,
                help="L1 regularization term on weights. "
                "Increasing this value will make model more conservative.",
            ),
            "reg_lambda": columns[2].slider(
                label="`reg_lambda`",
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=0.5,
                help="L2 regularization term on weights. "
                "Increasing this value will make model more conservative.",
            ),
        }

    @classmethod
    @st.cache_data(show_spinner=False)
    def get_model(_cls, **params):
        return XGBClassifier(**params)

    @classmethod
    @st.cache_data(show_spinner=False, hash_funcs={XGBClassifier: model_hash_func})
    def fit_model(
        _cls, model: XGBClassifier, X_train: pd.DataFrame, y_train: pd.Series
    ) -> XGBClassifier:
        return model.fit(X_train, y_train)

    @classmethod
    @st.cache_data(show_spinner=False, hash_funcs={XGBClassifier: model_hash_func})
    def evaluate_model(
        _cls,
        model: XGBClassifier,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        mapping: t.Dict[int, str],
    ) -> pd.DataFrame:
        y_pred = model.predict(X_test)
        report = classification_report(
            y_true=y_test,
            y_pred=y_pred,
            output_dict=True,
            target_names=mapping.values(),
            zero_division=0.0,
        )
        return pd.DataFrame(report).transpose()
