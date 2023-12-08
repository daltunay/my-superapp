import streamlit as st

import utils
from src.machine_learning.classification import ClassificationManager
from src.machine_learning.datasets import Dataset

loader = utils.PageConfigLoader(__file__)
loader.set_page_config(globals())

logger = utils.CustomLogger(__file__)

st_ss = st.session_state


def main():
    utils.tabs_config()
    utils.show_source_code("src/machine_learning/classification/classifier.py")

    st.header("Dataset", divider="gray")
    dataset = Dataset(type="classification")
    raw_dataset_dict = Dataset.get_dataset(**dataset.params)
    dataset.set(raw_dataset_dict)

    with st.expander(label="Dataset description"):
        st.markdown(dataset.description)

    X_train, X_test = dataset.X
    y_train, y_test = dataset.y
    label_mapping = dataset.label_mapping

    st.subheader("Visualize data")
    train_tab, test_tab = st.tabs(
        tabs=["Train".center(1, "\u2001"), "Test".center(1, "\u2001")]
    )
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
    st.markdown("Classification model: `XGBClassifier` from `xgboost`")
    classification_manager = ClassificationManager()

    st.subheader("Hyperparameters")
    classification_manager.set_model(label_mapping=label_mapping)

    st.subheader("Evaluation")
    classification_manager.fit(X_train, y_train)
    classification_manager.evaluate(
        X_test, y_test, target_names=list(label_mapping.values())
    )
    st.markdown("Classification Report")
    st.dataframe(
        data=classification_manager.classification_report, use_container_width=True
    )
    st.markdown("Confusion Matrix")
    st.pyplot(
        fig=classification_manager.confusion_matrix_display(
            display_labels=list(label_mapping.values())
        )
    )
