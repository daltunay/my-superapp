import pandas as pd
import streamlit as st

import utils
from src.statistics.statistical_tests import Chi2Testing

loader = utils.PageConfigLoader(__file__)
loader.set_page_config(globals())


def main():
    st.header("Data", divider="gray")
    observed_template = pd.DataFrame(
        data=[["Group A", 30, 20], ["Group B", 70, 80]],
        index=None,
        columns=["group", "category_1", "category_2"],
    )
    col_df, col_sum = st.columns([0.8, 0.2])
    with col_df:
        observed = st.data_editor(
            data=observed_template,
            hide_index=True,
            column_config={
                "group": st.column_config.TextColumn(
                    "Group",
                    help="The name of the considered group.",
                ),
                "category_1": st.column_config.NumberColumn(
                    "Category 1",
                    min_value=1,
                    required=True,
                    help="The observed values for the category 1.",
                ),
                "category_2": st.column_config.NumberColumn(
                    "Category 2",
                    min_value=1,
                    required=True,
                    help="The observed values for the category 2.",
                ),
            },
            disabled=False,
            use_container_width=True,
        )
        st.info("Click on any cell to change its content.", icon="💡")
    with col_sum:
        total_col = observed.drop("group", axis=1).sum(axis=1).to_frame(name="Total")
        st.dataframe(total_col, hide_index=True, use_container_width=True)

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

    chi2_testing = Chi2Testing(observed.drop("group", axis=1), alpha)

    st.header("Results", divider="gray")
    result = chi2_testing.perform_chi2_test()

    if result["is_significant"]:
        st.success("The difference is significant", icon="✅")
    else:
        st.error("The difference is not significant", icon="❌")

    st.expander(label="Test details").json(result)
