from scipy import stats


class ABTesting:
    def __init__(
        self, a_conversions, a_visitors, b_conversions, b_visitors, confidence
    ):
        self.a_conversions = a_conversions
        self.a_visitors = a_visitors
        self.b_conversions = b_conversions
        self.b_visitors = b_visitors
        self.confidence = confidence

    def calculate_conversion_rate(self, conversions, visitors):
        return 0 if visitors == 0 else conversions / visitors

    def perform_ab_test(self):
        # Conversion rates
        rate_a = self.calculate_conversion_rate(self.a_conversions, self.a_visitors)
        rate_b = self.calculate_conversion_rate(self.b_conversions, self.b_visitors)

        # Conduct A/B test using independent two-sample t-test
        std_a = (rate_a * (1 - rate_a) / self.a_visitors) ** 0.5
        std_b = (rate_b * (1 - rate_b) / self.b_visitors) ** 0.5

        _, p_value = stats.ttest_ind_from_stats(
            mean1=rate_a,
            std1=std_a,
            nobs1=self.a_visitors,
            mean2=rate_b,
            std2=std_b,
            nobs2=self.b_visitors,
        )

        # Calculate confidence interval
        alpha = 1 - self.confidence / 100
        interval = (
            stats.norm.ppf(1 - alpha / 2)
            * (
                (rate_a * (1 - rate_a) / self.a_visitors)
                + (rate_b * (1 - rate_b) / self.b_visitors)
            )
            ** 0.5
        )

        # Determine if the difference is statistically significant
        if p_value < alpha:
            result = "Statistically significant difference"
        else:
            result = "No statistically significant difference"

        return {
            "p_value": p_value,
            "confidence_interval": (
                rate_b - rate_a - interval,
                rate_b - rate_a + interval,
            ),
            "result": result,
        }
