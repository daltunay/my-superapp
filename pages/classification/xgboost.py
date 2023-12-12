import streamlit as st

import utils
from src.machine_learning import XGBoostManager
from src.machine_learning.datasets import Dataset

loader = utils.PageConfigLoader(__file__)
loader.set_page_config(globals())

logger = utils.CustomLogger(__file__)

st_ss = st.session_state


def main():
    utils.tabs_config()
    utils.show_source_code("src/machine_learning/xgboost_manager.py")

    st.header("Dataset", divider="gray")
    dataset = Dataset(type="classification")
    raw_dataset_dict = Dataset.get_dataset(**dataset.params, split=True)
    dataset.set(raw_dataset_dict)

    with st.expander(label="Dataset description"):
        st.markdown(dataset.description)

    X_train, X_test = dataset.X
    y_train, y_test = dataset.y
    label_mapping = dataset.label_mapping

    st.subheader("Visualize data")
    train_tab, test_tab = st.tabs(tabs=["Train", "Test"])
    with train_tab:
        with st.container(border=True):
            utils.display_tab_content("train", X_train, y_train, label_mapping)
    with test_tab:
        with st.container(border=True):
            utils.display_tab_content("test", X_test, y_test, label_mapping)

    st.header("Classification", divider="gray")
    st.markdown(
        "Classification model: `XGBClassifier` from `xgboost` "
        "([official documentation](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier))"
    )
    classification_manager = XGBoostManager(task="classification")

    st.subheader("Hyperparameters")
    with st.container(border=True):
        classification_manager.set_model(label_mapping=label_mapping)

    st.subheader("Evaluation")
    classification_manager.fit(X_train, y_train)
    classification_manager.evaluate(
        X_test, y_test, target_names=list(label_mapping.values())
    )
    st.markdown("Classification Report")
    st.columns(3)[1].dataframe(
        data=classification_manager.classification_report, use_container_width=True
    )
    st.markdown("Confusion Matrix")
    st.columns([0.1, 1, 0.1])[1].pyplot(
        fig=classification_manager.confusion_matrix_display(
            display_labels=list(label_mapping.values())
        )
    )
    st.subheader("Explainability")
    st.markdown("SHAP force plot")
    utils.st_shap(plot=classification_manager.shap_force_plot(X_test), height=400)
