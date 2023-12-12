import typing as t

import pandas as pd
import streamlit as st


def tabs_config():
    st.markdown(
        """
            <style>
                button[data-baseweb="tab"] {
                font-size: 24px;
                margin: 0;
                width: 100%;
                }
            </style>
            """,
        unsafe_allow_html=True,
    )


def display_tab_content(
    label: t.Literal["train", "test"],
    X_data: pd.DataFrame,
    y_data: pd.DataFrame,
    label_mapping: t.Dict[int, str] | None = None,
):
    data_container = st.container()
    col1, col2 = data_container.columns([0.65, 0.35], gap="medium")
    with col1:
        st.markdown(
            f"<h3 style='text-align: center;'>X_{label}</h3>", unsafe_allow_html=True
        )
        st.dataframe(data=X_data, use_container_width=True)

    with col2:
        st.markdown(
            f"<h3 style='text-align: center;'>y_{label}</h3>", unsafe_allow_html=True
        )
        st.dataframe(
            data=y_data.map(label_mapping or (lambda x: x)), use_container_width=True
        )

    describe_container = st.expander("Data statistics").container()
    col1, col2 = describe_container.columns([0.65, 0.35], gap="medium")
    with col1:
        st.dataframe(X_data.describe(), use_container_width=True)
    with col2:
        if label_mapping:
            st.dataframe(
                pd.concat(
                    [
                        y_data.map(label_mapping).value_counts().sort_index(),
                        y_data.map(label_mapping)
                        .value_counts(normalize=True)
                        .sort_index(),
                    ],
                    axis=1,
                ).round(3)
            )
        else:
            st.dataframe(y_data.describe(), use_container_width=True)
