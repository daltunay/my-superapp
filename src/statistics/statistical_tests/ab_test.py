from scipy import stats


class ABTesting:
    def __init__(
        self,
        a_conversions: int,
        a_visitors: int,
        b_conversions: int,
        b_visitors: int,
        confidence: float,
    ):
        self.a_conversions, self.a_visitors = a_conversions, a_visitors
        self.b_conversions, self.b_visitors = b_conversions, b_visitors
        self.confidence = confidence

    def conversion_rate(self, conversions: int, visitors: int) -> float:
        return conversions / visitors

    def standard_deviation(self, rate: float, visitors: int) -> float:
        return (rate * (1 - rate) / visitors) ** 0.5

    def confidence_interval(
        self, rate_a: float, rate_b: float, std_a: float, std_b: float
    ) -> tuple[float, float]:
        alpha = 1 - self.confidence / 100
        interval = (
            stats.norm.ppf(1 - alpha / 2)
            * ((std_a**2 / self.a_visitors) + (std_b**2 / self.b_visitors)) ** 0.5
        )
        return rate_b - rate_a - interval, rate_b - rate_a + interval

    def is_statistically_significant(self, p_value: float, alpha: float) -> bool:
        return p_value < alpha

    def perform_ab_test(self) -> dict[str, any]:
        rate_a = self.conversion_rate(self.a_conversions, self.a_visitors)
        rate_b = self.conversion_rate(self.b_conversions, self.b_visitors)

        std_a = self.standard_deviation(rate_a, self.a_visitors)
        std_b = self.standard_deviation(rate_b, self.b_visitors)

        _, p_value = stats.ttest_ind_from_stats(
            mean1=rate_a,
            std1=std_a,
            nobs1=self.a_visitors,
            mean2=rate_b,
            std2=std_b,
            nobs2=self.b_visitors,
        )

        confidence_interval = self.confidence_interval(rate_a, rate_b, std_a, std_b)

        alpha = 1 - self.confidence / 100
        significant = self.is_statistically_significant(p_value, alpha)

        return {
            "p_value": p_value,
            "confidence_interval": confidence_interval,
            "significant": significant,
        }
