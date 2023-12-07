import streamlit as st

import utils
from src.machine_learning.classification import Classifier
from src.machine_learning.datasets import Dataset

loader = utils.PageConfigLoader(__file__)
loader.set_page_config(globals())

logger = utils.CustomLogger(__file__)

st_ss = st.session_state


def main():
    utils.show_source_code("src/machine_learning/classification/classifier.py")
    dataset = Dataset(type="classification")
    raw_dataset_dict = dataset.get_dataset(**dataset.params)
    dataset.set(raw_dataset_dict)

    with st.expander(label="Dataset description"):
        st.markdown(dataset.description)

    X_train, X_test = dataset.X
    y_train, y_test = dataset.y
    label_mapping = dataset.label_mapping

    train_tab, test_tab = st.tabs(tabs=["Train", "Test"])
    with train_tab:
        col1, col2 = st.columns([0.8, 0.2])
        col1.dataframe(data=X_train, use_container_width=True)
        col2.dataframe(data=y_train.map(label_mapping), use_container_width=True)

    with test_tab:
        col1, col2 = st.columns([0.8, 0.2])
        col1.dataframe(data=X_test, use_container_width=True)
        col2.dataframe(data=y_test.map(label_mapping), use_container_width=True)

    classifier = Classifier()
    classifier.set_model()
    classifier.fit(X_train, y_train)
    classifier.evaluate(X_test, y_test, label_mapping)

    st.dataframe(data=classifier.report, use_container_width=True)
