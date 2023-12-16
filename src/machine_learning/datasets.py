import typing as t

import pandas as pd
import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split


class DatasetParams(t.TypedDict):
    source: t.Literal["iris", "digits", "breast_cancer"]
    test_size: float | None
    shuffle: bool
    stratify: bool


class Dataset:
    def __init__(
        self,
        type: t.Literal["classification", "regression"] | None = None,
    ):
        self.type = type
        self.X: t.Tuple[pd.DataFrame, pd.DataFrame] | None = None
        self.y: t.Tuple[pd.Series, pd.Series] | None = None
        self.label_mapping: t.Dict[int, str] | None = None
        self.description: str | None = None

    @property
    def params(self) -> t.Dict[str, t.Any]:
        columns = st.columns(3)
        return {
            "source": columns[0].selectbox(
                label="source",
                options=["iris", "digits", "breast_cancer"]
                if self.type == "classification"
                else ["diabetes"]
                if self.type == "regression"
                else ["iris", "digits", "breast_cancer", "diabetes"],
                help="The scikit-learn toy dataset to use.",
            ),
            "test_size": columns[1].slider(
                "test_size",
                min_value=0.05,
                max_value=0.3,
                value=0.2,
                step=0.05,
                help="The proportion of the dataset to include in the test split",
            )
            if self.type is not None
            else None,
            "shuffle": columns[2].checkbox(
                label="shuffle",
                value=True,
                help="Whether to shuffle the dataset or not.",
            )
            if self.type is not None
            else None,
            "stratify": columns[2].checkbox(
                label="stratify",
                value=False,
                help="Whether to stratify the dataset or not. "
                "Stratifying means keeping the same label distribution in the initial, train and test datasets. "
                "Available for classification only.",
                disabled=self.type == "regression",
            )
            if self.type is not None
            else None,
        }

    @staticmethod
    @st.cache_data(show_spinner=False)
    def get_dataset(
        split: bool = False, **params: t.Unpack[DatasetParams]
    ) -> t.Dict[str, t.Any]:
        raw_dataset = getattr(datasets, f"load_{params['source']}")(as_frame=True)
        X, y = raw_dataset.data, raw_dataset.target
        if split:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=params["test_size"],
                shuffle=params["shuffle"],
                stratify=y if params["stratify"] else None,
                random_state=0,
            )
            X = X_train, X_test
            y = y_train, y_test
        return {
            "X": X,
            "y": y,
            "label_mapping": dict(enumerate(raw_dataset.target_names))
            if "target_names" in raw_dataset
            else None,
            "description": raw_dataset.DESCR,
        }

    def set(self, raw_dataset_dict: t.Dict[str, t.Any]):
        self.X = raw_dataset_dict["X"]
        self.y = raw_dataset_dict["y"]
        self.label_mapping = raw_dataset_dict["label_mapping"]
        self.description = raw_dataset_dict["description"]
