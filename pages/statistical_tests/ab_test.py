import streamlit as st

import utils
from src.statistics.statistical_tests import ABTesting

loader = utils.PageConfigLoader(__file__)
loader.set_page_config(globals())


def main():
    container = st.container(border=True)
    a_col, b_col = container.columns(2, gap="large")
    with a_col:
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
    with b_col:
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
    confidence = container.slider(
        "Confidence level (%)",
        min_value=1,
        max_value=99,
        value=95,
        step=1,
    )

    ab_testing = ABTesting(
        a_conversions,
        a_visitors,
        b_conversions,
        b_visitors,
        confidence,
    )

    result = ab_testing.perform_ab_test()

    st.subheader("Results:")
    st.write(f"P-value: {result['p_value']:.4f}")
    st.write(f"Confidence Interval: {result['confidence_interval']}")
    st.write(f"Significant: {result['significant']}")
