from src.statistics.statistical_tests import ABTesting
import utils
import streamlit as st

loader = utils.PageConfigLoader(__file__)
loader.set_page_config(globals())


def main():
    st.title("A/B Testing Tool")

    a_conversions = st.number_input("Conversions for Group A", min_value=0, step=1)
    a_visitors = st.number_input("Total visitors for Group A", min_value=0, step=1)
    b_conversions = st.number_input("Conversions for Group B", min_value=0, step=1)
    b_visitors = st.number_input("Total visitors for Group B", min_value=0, step=1)
    confidence = st.slider(
        "Confidence level (%)", min_value=1, max_value=99, value=95, step=1
    )

    ab_testing = ABTesting(
        a_conversions, a_visitors, b_conversions, b_visitors, confidence
    )

    result = ab_testing.perform_ab_test()

    st.subheader("Results:")
    st.write(f"P-value: {result['p_value']:.4f}")
    st.write(f"Confidence Interval: {result['confidence_interval']}")
    st.write(f"Conclusion: {result['result']}")
