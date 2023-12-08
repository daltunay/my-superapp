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
    dataset = Dataset(type="regression")
    raw_dataset_dict = Dataset.get_dataset(**dataset.params)
    dataset.set(raw_dataset_dict)

    with st.expander(label="Dataset description"):
        st.markdown(dataset.description)

    X_train, X_test = dataset.X
    y_train, y_test = dataset.y

    st.subheader("Visualize data")
    train_tab, test_tab = st.tabs(tabs=["Train", "Test"])
    with train_tab:
        utils.display_tab_content("train", X_train, y_train)
    with test_tab:
        utils.display_tab_content("test", X_test, y_test)

    st.header("Regression", divider="gray")
    st.markdown(
        "Regression model: `XGBRegressor` from `xgboost` "
        "([official documentation](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor))"
    )
    regression_manager = XGBoostManager(task="regression")

    st.subheader("Hyperparameters")
    regression_manager.set_model()

    st.subheader("Evaluation")
    regression_manager.fit(X_train, y_train)
    regression_manager.evaluate(X_test, y_test)
    st.markdown("Metrics Report")
    st.dataframe(data=regression_manager.metrics_report, use_container_width=True)
