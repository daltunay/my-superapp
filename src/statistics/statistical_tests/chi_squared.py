import typing as t

import pandas as pd
import streamlit as st
from scipy.stats import chi2_contingency


class Chi2Testing:
    def __init__(
        self,
        observed: pd.DataFrame,
        alpha: float,
    ):
        self.observed = observed
        self.alpha = alpha

    @staticmethod
    @st.cache_data(show_spinner=False)
    def chi2_test(
        observed: pd.DataFrame,
    ) -> t.Tuple[float, float, int, t.List[t.List[float]]]:
        chi2, p_value, dof, expected = chi2_contingency(observed)
        return chi2, p_value, dof, expected

    @staticmethod
    @st.cache_data(show_spinner=False)
    def is_statistically_significant(p_value: float, alpha: float) -> bool:
        return p_value < alpha

    def perform_chi2_test(self) -> t.Dict[str, t.Any]:
        chi2, p_value, dof, expected = self.chi2_test(self.observed)
        is_significant = self.is_statistically_significant(p_value, self.alpha)

        return {
            "chi2_statistic": chi2,
            "p_value": p_value,
            "degrees_of_freedom": dof,
            "expected_frequencies": expected,
            "is_significant": is_significant,
        }
