import streamlit as st

import utils
from src.statistics.statistical_tests import ABTesting

loader = utils.PageConfigLoader(__file__)
loader.set_page_config(globals())


def main():
    container = st.container(border=True)
    container.header("Test data")
    a_col, b_col = container.columns(2, gap="small")
    with a_col.container(border=True):
        st.subheader("Group A")
        a_visitors = st.number_input(
            "Total visitors for Group A",
            min_value=1,
            value=1000,
            step=1,
        )
        a_conversions = st.number_input(
            "Conversions for Group A",
            min_value=0,
            max_value=a_visitors,
            value=50,
            step=1,
        )
    with b_col.container(border=True):
        st.subheader("Group B")
        b_visitors = st.number_input(
            "Total visitors for Group B",
            min_value=1,
            value=200,
            step=1,
        )
        b_conversions = st.number_input(
            "Conversions for Group B",
            min_value=0,
            max_value=b_visitors,
            value=35,
            step=1,
        )

    container.header("Settings")
    confidence_col, alpha_col = container.container(border=True).columns(2)
    _ = confidence_col.columns([0.1, 1, 0.1])[1].select_slider(
        "Confidence level",
        options=[0.9, 0.95, 0.99],
        value=0.95,
        key="ab_test.confidence",
        format_func=lambda x: f"{100*x}%",
        on_change=utils.update_slider_callback,
        kwargs={"updated": "ab_test.confidence", "to_update": "ab_test.alpha"},
    )
    alpha = alpha_col.columns([0.1, 1, 0.1])[1].select_slider(
        "Alpha value",
        options=[0.01, 0.05, 0.1],
        value=0.05,
        key="ab_test.alpha",
        format_func=lambda x: f"{100*x}%",
        on_change=utils.update_slider_callback,
        kwargs={"updated": "ab_test.alpha", "to_update": "ab_test.confidence"},
    )

    ab_testing = ABTesting(
        a_conversions,
        a_visitors,
        b_conversions,
        b_visitors,
        alpha,
    )

    result = ab_testing.perform_ab_test()

    st.json(result)
