import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import DBSCAN


class DBScanManager:
    def __init__(self):
        self.model: DBSCAN | None = None

    @property
    def params(self) -> dict:
        columns = st.columns(2)
        return {
            "eps": columns[0].slider(
                label="Maximum Distance (eps)",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1,
                help="Maximum distance between two samples for one to be considered as in the neighborhood of the other.",
            ),
            "min_samples": columns[1].slider(
                label="Minimum Samples",
                min_value=1,
                max_value=10,
                value=5,
                step=1,
                help="The number of samples in a neighborhood for a point to be considered as a core point.",
            ),
        }

    @staticmethod
    @st.cache_resource(show_spinner=True)
    def _get_model(eps: float, min_samples: int) -> DBSCAN:
        return DBSCAN(eps=eps, min_samples=min_samples)

    def set_model(self) -> None:
        self.model = self._get_model(**self.params)

    @staticmethod
    @st.cache_resource(
        show_spinner=True,
        hash_funcs={DBSCAN: lambda model: (model.eps, model.min_samples)},
    )
    def _perform_clustering(model: DBSCAN, data: pd.DataFrame) -> pd.DataFrame:
        clusters = model.fit_predict(data)
        data = data.assign(Cluster=clusters)
        data["Cluster"] = data["Cluster"].astype(str)
        return model, data

    def fit(self, data: pd.DataFrame):
        self.model, self.data_clustered = self._perform_clustering(
            model=self.model, data=data
        )

    def scatter_plot(self, x_col: str, y_col: str) -> None:
        return px.scatter(
            self.data_clustered,
            x=x_col,
            y=y_col,
            color="Cluster",
            labels={"color": "Cluster"},
        )
