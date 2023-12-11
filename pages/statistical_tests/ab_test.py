import streamlit as st

import utils
from src.statistics.statistical_tests import ABTesting, input_group_data

loader = utils.PageConfigLoader(__file__)
loader.set_page_config(globals())


def main():
    st.header("Data", divider="gray")
    a_col, b_col = st.columns(2, gap="small")
    with a_col.container(border=True):
        st.subheader("Group A")
        a_visitors, a_conversions, a_rate = input_group_data(
            group_name="A", default_visitors=1000, default_conversions=50
        )
    with b_col.container(border=True):
        st.subheader("Group B")
        b_visitors, b_conversions, b_rate = input_group_data(
            group_name="B", default_visitors=200, default_conversions=35
        )

    st.header("Settings", divider="gray")
    settings_container = st.container(border=True)
    test_type = settings_container.selectbox(
        label="Test type",
        key="ab_test.test_type",
        options=["one-sided", "two-sided"],
        index=1,
        format_func=lambda x: x.replace("-", " ").capitalize(),
    )
    confidence_col, alpha_col = settings_container.columns(2)
    confidence = confidence_col.columns([0.15, 1, 0.15])[1].select_slider(
        "Confidence level",
        options=[0.9, 0.95, 0.99],
        value=0.95,
        key="ab_test.confidence",
        format_func=lambda x: f"{100*x}%",
        on_change=utils.update_slider_callback,
        kwargs={"updated": "ab_test.confidence", "to_update": "ab_test.alpha"},
    )
    alpha = alpha_col.columns([0.15, 1, 0.15])[1].select_slider(
        "Alpha value",
        options=[0.01, 0.05, 0.1],
        value=0.05,
        key="ab_test.alpha",
        format_func=lambda x: f"{100*x}%",
        on_change=utils.update_slider_callback,
        kwargs={"updated": "ab_test.alpha", "to_update": "ab_test.confidence"},
    )

    ab_testing = ABTesting(a_visitors, a_rate, b_visitors, b_rate, alpha, test_type)

    st.header("Results", divider="gray")
    result = ab_testing.perform_ab_test()

    if result["is_significant"]:
        st.success("The difference is significant", icon="✅")
    else:
        st.error("The difference is not significant", icon="❌")

    st.expander(label="Test details").json(result)
