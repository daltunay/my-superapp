from scipy import stats


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
    def compute_standard_deviation(rate: float, visitors: int) -> float:
        return (rate * (1 - rate) / visitors) ** 0.5

    def confidence_interval(
        self, a_rate: float, b_rate: float, std_a: float, std_b: float
    ) -> tuple[float, float]:
        interval = (
            stats.norm.ppf(1 - self.alpha / 2)
            * ((std_a**2 / self.a_visitors) + (std_b**2 / self.b_visitors)) ** 0.5
        )
        return b_rate - a_rate - interval, b_rate - a_rate + interval

    @staticmethod
    def is_statistically_significant(p_value: float, alpha: float) -> bool:
        return p_value < alpha

    def perform_ab_test(self) -> dict[str, any]:
        std_a = self.compute_standard_deviation(self.a_rate, self.a_visitors)
        std_b = self.compute_standard_deviation(self.b_rate, self.b_visitors)

        t_statistic, p_value = stats.ttest_ind_from_stats(
            mean1=self.a_rate,
            std1=std_a,
            nobs1=self.a_visitors,
            mean2=self.b_rate,
            std2=std_b,
            nobs2=self.b_visitors,
        )

        confidence_interval = self.confidence_interval(
            self.a_rate, self.b_rate, std_a, std_b
        )

        is_significant = self.is_statistically_significant(p_value, self.alpha)

        return {
            "t_statistic": t_statistic,
            "p_value": p_value,
            "confidence_interval": confidence_interval,
            "is_significant": is_significant,
        }
