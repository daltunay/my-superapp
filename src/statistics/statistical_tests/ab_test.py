from scipy import stats
import streamlit as st
import typing as t


class ABTesting:
    def __init__(
        self,
        a_visitors: int,
        a_rate: float,
        b_visitors: int,
        b_rate: float,
        alpha: float,
    ):
        self.a_visitors, self.a_rate = a_visitors, a_rate
        self.b_visitors, self.b_rate = b_visitors, b_rate
        self.alpha = alpha

    @staticmethod
    @st.cache_data(show_spinner=False)
    def compute_standard_deviation(rate: float, visitors: int) -> float:
        return (rate * (1 - rate) / visitors) ** 0.5

    @classmethod
    @st.cache_data(show_spinner=False)
    def compute_confidence_interval(
        _cls,
        a_rate: float,
        b_rate: float,
        a_visitors: int,
        b_visitors: int,
        alpha: float,
    ) -> t.Tuple[float, float]:
        a_std = _cls.compute_standard_deviation(a_rate, a_visitors)
        b_std = _cls.compute_standard_deviation(b_rate, b_visitors)
        interval = (
            stats.norm.ppf(1 - alpha / 2)
            * ((a_std**2 / a_visitors) + (b_std**2 / b_visitors)) ** 0.5
        )
        return b_rate - a_rate - interval, b_rate - a_rate + interval

    @staticmethod
    @st.cache_data(show_spinner=False)
    def is_statistically_significant(p_value: float, alpha: float) -> bool:
        return p_value < alpha

    @staticmethod
    @st.cache_data(show_spinner=False)
    def t_test(a_rate, a_std, a_visitors, b_rate, b_std, b_visitors):
        return stats.ttest_ind_from_stats(
            mean1=a_rate,
            std1=a_std,
            nobs1=a_visitors,
            mean2=b_rate,
            std2=b_std,
            nobs2=b_visitors,
        )

    def perform_ab_test(self) -> t.Dict[str, any]:
        a_std = self.compute_standard_deviation(self.a_rate, self.a_visitors)
        b_std = self.compute_standard_deviation(self.b_rate, self.b_visitors)

        t_statistic, p_value = self.t_test(
            self.a_rate,
            a_std,
            self.a_visitors,
            self.b_rate,
            b_std,
            self.b_visitors,
        )

        confidence_interval = self.compute_confidence_interval(
            self.a_rate,
            self.b_rate,
            self.a_visitors,
            self.b_visitors,
            self.alpha,
        )

        is_significant = self.is_statistically_significant(p_value, self.alpha)

        return {
            "t_statistic": t_statistic,
            "p_value": p_value,
            "confidence_interval": confidence_interval,
            "is_significant": is_significant,
        }
