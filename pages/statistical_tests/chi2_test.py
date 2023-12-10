import pandas as pd
import streamlit as st

import utils
from src.statistics.statistical_tests import Chi2Testing

loader = utils.PageConfigLoader(__file__)
loader.set_page_config(globals())


def main():
    st.header("Data", divider="gray")
    observed_template = pd.DataFrame(
        data=[[0, 0], [0, 0]],
        index=["Group A", "Group B"],
        columns=["Category 1", "Category 2"],
    )
    observed = st.data_editor(data=observed_template, use_container_width=True)

    st.header("Settings", divider="gray")
    settings_container = st.container(border=True)
    confidence_col, alpha_col = settings_container.columns(2)
    confidence = confidence_col.columns([0.15, 1, 0.15])[1].select_slider(
        "Confidence level",
        options=[0.9, 0.95, 0.99],
        value=0.95,
        key="chi2_test.confidence",
        format_func=lambda x: f"{100*x}%",
        on_change=utils.update_slider_callback,
        kwargs={"updated": "chi2_test.confidence", "to_update": "chi2_test.alpha"},
    )
    alpha = alpha_col.columns([0.15, 1, 0.15])[1].select_slider(
        "Alpha value",
        options=[0.01, 0.05, 0.1],
        value=0.05,
        key="chi2_test.alpha",
        format_func=lambda x: f"{100*x}%",
        on_change=utils.update_slider_callback,
        kwargs={"updated": "chi2_test.alpha", "to_update": "chi2_test.confidence"},
    )

    chi2_testing = Chi2Testing(observed, alpha)

    st.header("Results", divider="gray")
    result = chi2_testing.perform_chi2_test()
    st.json(result)
