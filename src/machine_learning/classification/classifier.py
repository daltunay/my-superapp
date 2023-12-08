import typing as t

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
import streamlit as st
from xgboost import XGBClassifier


class ClassificationManager:
    def __init__(self) -> None:
        self.model: XGBClassifier | None = None
        self.current_params: t.Dict[str, float | int] | None = None
        self.fitted: bool = False
        self.classification_report: pd.DataFrame | None = None
        self.confusion_matrix: pd.DataFrame | None = None

    @property
    def params(self) -> t.Dict[str, float | int]:
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
                max_value=100,
                value=50,
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

    @staticmethod
    @st.cache_resource(show_spinner=True)
    def _get_model(label_mapping: t.Dict[str, int], **params) -> XGBClassifier:
        return XGBClassifier(**params)

    def set_model(self, label_mapping: t.Dict[str, int]) -> None:
        self.model = self._get_model(label_mapping, **self.params)

    @staticmethod
    @st.cache_resource(
        show_spinner=True,
        hash_funcs={
            XGBClassifier: lambda model: {
                key: val for key, val in vars(model).items() if key != "_Booster"
            }
        },
    )
    def _fit_model(
        model: XGBClassifier, X_train: pd.DataFrame, y_train: pd.Series
    ) -> XGBClassifier:
        return model.fit(X_train, y_train)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.model = self._fit_model(self.model, X_train, y_train)
        self.fitted = True

    @staticmethod
    @st.cache_data(show_spinner=True)
    def _classification_report(
        y_true: pd.Series, y_pred: pd.Series, target_names: t.List[str]
    ):
        return (
            pd.DataFrame(
                sklearn.metrics.classification_report(
                    y_true=y_true,
                    y_pred=y_pred,
                    target_names=target_names,
                    output_dict=True,
                    zero_division=np.nan,
                )
            )
            .astype(float)
            .round(4)
            .transpose()
        )

    @staticmethod
    @st.cache_data(show_spinner=True)
    def _confusion_matrix(y_true: pd.Series, y_pred: pd.Series):
        return pd.DataFrame(
            sklearn.metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
        )

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        target_names: t.List[str],
    ):
        y_pred = self.model.predict(X_test)
        self.classification_report = self._classification_report(
            y_true=y_test,
            y_pred=y_pred,
            target_names=target_names,
        )
        self.confusion_matrix = self._confusion_matrix(
            y_true=y_test,
            y_pred=y_pred,
        )

    @staticmethod
    @st.cache_data(show_spinner=True)
    def _confusion_matrix_display(
        confusion_matrix: pd.DataFrame, display_labels: t.List[str]
    ) -> sklearn.metrics.ConfusionMatrixDisplay:
        return sklearn.metrics.ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix,
            display_labels=display_labels,
        )

    def confusion_matrix_display(self, display_labels: t.List[str]):
        confusion_matrix_display = self._confusion_matrix_display(
            confusion_matrix=self.confusion_matrix.to_numpy(),
            display_labels=display_labels,
        )
        fig, ax = plt.subplots()
        confusion_matrix_display.plot(ax=ax)
        return fig
