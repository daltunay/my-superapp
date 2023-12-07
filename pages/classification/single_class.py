import streamlit as st

import utils
from src.machine_learning.classification import ClassificationManager
from src.machine_learning.datasets import get_dataset

loader = utils.PageConfigLoader(__file__)
loader.set_page_config(globals())

logger = utils.CustomLogger(__file__)

st_ss = st.session_state


def main():
    utils.show_source_code("src/machine_learning/classification/classifier.py")
    columns = st.columns(3)
    dataset_params = {
        "source": columns[0].selectbox(
            label="source", options=["iris", "digits", "breast_cancer"]
        ),
        "test_size": columns[1].slider(
            "test_size", min_value=0.05, max_value=0.5, value=0.2, step=0.05
        ),
        "shuffle": columns[2].checkbox(label="shuffle", value=True),
        "stratify": columns[2].checkbox(label="stratify", value=False),
    }
    dataset = get_dataset(**dataset_params)
    with st.expander(label="Dataset description"):
        st.markdown(dataset["descr"])
    X_train, y_train = dataset["train"]
    X_test, y_test = dataset["test"]
    label_mapping = dataset["mapping"]

    train_tab, test_tab = st.tabs(tabs=["Train", "Test"])
    with train_tab:
        col1, col2 = st.columns([0.8, 0.2])
        col1.dataframe(data=X_train, use_container_width=True)
        col2.dataframe(data=y_train.map(label_mapping), use_container_width=True)

    with test_tab:
        col1, col2 = st.columns([0.8, 0.2])
        col1.dataframe(data=X_test, use_container_width=True)
        col2.dataframe(data=y_test.map(label_mapping), use_container_width=True)

    params = ClassificationManager.get_params()
    model = ClassificationManager.get_model(**params)
    model_fit = ClassificationManager.fit_model(model, X_train, y_train)
    report = ClassificationManager.evaluate_model(model_fit, X_test, y_test, label_mapping)
    st.dataframe(data=report, use_container_width=True)
