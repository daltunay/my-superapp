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
