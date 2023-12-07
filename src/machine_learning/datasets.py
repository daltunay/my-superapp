from sklearn import datasets
from sklearn.model_selection import train_test_split
import typing as t
import streamlit as st
from pandas import DataFrame


@st.cache_data(show_spinner=False)
def get_dataset(
    source: t.Literal["iris", "digits", "breast_cancer"],
    test_size: float | None = 0.2,
    shuffle: bool = True,
    stratify: bool = False,
) -> t.List[DataFrame]:
    dataset = getattr(datasets, f"load_{source}")(as_frame=True)
    X, y = dataset.data, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=shuffle,
        stratify=y if stratify else None,
        random_state=0,
    )
    return {
        "train": (X_train, y_train),
        "test": (X_test, y_test),
        "mapping": dict(enumerate(dataset.target_names)),
        "descr": dataset.DESCR,
    }
