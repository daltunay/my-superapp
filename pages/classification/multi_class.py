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

    st.header("Dataset", divider="gray")
    dataset = Dataset(type="classification")
    raw_dataset_dict = dataset.get_dataset(**dataset.params)
    dataset.set(raw_dataset_dict)

    with st.expander(label="Dataset description"):
        st.markdown(dataset.description)

    X_train, X_test = dataset.X
    y_train, y_test = dataset.y
    label_mapping = dataset.label_mapping

    st.subheader("Visualize data")
    train_tab, test_tab = st.tabs(tabs=["Train", "Test"])
    with train_tab:
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            st.markdown(
                "<h3 style='text-align: center;'>X_train</h3>", unsafe_allow_html=True
            )
            st.dataframe(data=X_train, use_container_width=True)
        with col2:
            st.markdown(
                "<h3 style='text-align: center;'>y_train</h3>", unsafe_allow_html=True
            )
            st.dataframe(data=y_train.map(label_mapping), use_container_width=True)

    with test_tab:
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            st.markdown(
                "<h3 style='text-align: center;'>X_test</h3>", unsafe_allow_html=True
            )
            st.dataframe(data=X_test, use_container_width=True)
        with col2:
            st.markdown(
                "<h3 style='text-align: center;'>y_test</h3>", unsafe_allow_html=True
            )
            st.dataframe(data=y_test.map(label_mapping), use_container_width=True)

    st.header("Classification", divider="gray")
    st.markdown("Chosen model: XGBClassifier")
    classifier = Classifier()

    st.subheader("Hyperparameters")
    classifier.set_model()

    st.subheader("Evaluation")
    classifier.fit(X_train, y_train)
    classifier.evaluate(X_test, y_test, label_mapping)
    st.dataframe(data=classifier.report, use_container_width=True)
