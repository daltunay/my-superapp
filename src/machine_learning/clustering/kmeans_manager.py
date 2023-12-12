import typing as t

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans


class KMeansManager:
    def __init__(self, max_n_clusters: int):
        self.max_n_clusters = max_n_clusters
        self.model: KMeans | None = None

    @property
    def params(self) -> t.Dict[str, int]:
        columns = st.columns(2)
        return {
            "n_clusters": columns[0].slider(
                label="Number of Clusters",
                min_value=1,
                max_value=self.max_n_clusters,
                value=2,
                step=1,
                help="Number of clusters to form.",
            ),
        }

    @staticmethod
    @st.cache_resource(show_spinner=True)
    def _get_model(n_clusters: int) -> KMeans:
        return KMeans(n_clusters=n_clusters, n_init="auto")

    def set_model(self) -> None:
        params = self.params
        self.model = self._get_model(params["n_clusters"])

    @staticmethod
    @st.cache_resource(
        show_spinner=True,
        hash_funcs={KMeans: lambda model: model.n_clusters},
    )
    def _perform_clustering(model: KMeans, data: pd.DataFrame) -> pd.DataFrame:
        model = model.fit(data)
        clusters = model.predict(data)
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

    def centroids_plot(self, x_col: str, y_col: str) -> None:
        centroids = pd.DataFrame(
            self.model.cluster_centers_,
            columns=[f"{col}_centroid" for col in self.data_clustered.columns[:-1]],
        )
        centroids[x_col] = centroids[f"{x_col}_centroid"]
        centroids[y_col] = centroids[f"{y_col}_centroid"]

        fig = px.scatter(
            self.data_clustered,
            x=x_col,
            y=y_col,
            color="Cluster",
            labels={"color": "Cluster"},
        )

        fig.add_trace(
            go.Scatter(
                x=centroids[x_col],
                y=centroids[y_col],
                mode="markers",
                marker=dict(size=20, symbol="x", color="white"),
                name="Centroids",
            )
        )
        return fig
