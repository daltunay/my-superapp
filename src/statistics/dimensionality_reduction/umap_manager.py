import typing as t

import pandas as pd
import plotly.express as px
import streamlit as st
from umap import UMAP


class UMAPManager:
    def __init__(self, max_n_components: int):
        self.max_n_components = max_n_components
        self.model: UMAP | None = None
        self.target_col: pd.Series | None = None
        self.embedded_data_df: pd.DataFrame | None = None

    @property
    def params(self) -> t.Dict[str, int | float]:
        columns = st.columns(3)
        return {
            "n_components": columns[0].slider(
                label="Number of Components",
                min_value=1,
                max_value=self.max_n_components,
                value=3,
                step=1,
                help="Number of components to compute.",
            ),
            "n_neighbors": columns[1].slider(
                label="Number of Neighbors",
                min_value=2,
                max_value=100,
                value=15,
                step=1,
                help="Size of local neighborhood used for manifold approximation.",
            ),
            "min_dist": columns[2].slider(
                label="Minimum Distance",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Minimum distance between embedded points.",
            ),
        }

    @st.cache_resource(show_spinner=True)
    def _get_model(_self, params: t.Dict[str, int | float]) -> UMAP:
        return UMAP(
            n_components=params["n_components"],
            n_neighbors=params["n_neighbors"],
            min_dist=params["min_dist"],
        )

    def set_model(self) -> None:
        params = self.params
        self.model = self._get_model(params)

    @st.cache_resource(
        show_spinner=True,
        hash_funcs={
            UMAP: lambda model: (model.n_components, model.n_neighbors, model.min_dist)
        },
    )
    def _compute_umap(_self, model: UMAP, data: pd.DataFrame) -> pd.DataFrame:
        embedded_data = model.fit_transform(data)
        column_names = [f"D{i}" for i in range(1, model.n_components + 1)]
        return pd.DataFrame(embedded_data, columns=column_names)

    def fit(self, data: pd.DataFrame, target_col: pd.Series):
        self.embedded_data_df = self._compute_umap(model=self.model, data=data)
        self.target_col = target_col

    def scatter_matrix_plot(self) -> None:
        return px.scatter_matrix(
            self.embedded_data_df, color=self.target_col, labels={"color": "target"}
        )

    def scatter_2d_plot(self) -> None:
        return px.scatter(
            self.embedded_data_df,
            x="D1",
            y="D2",
            color=self.target_col,
            labels={"color": "target"},
        )

    def scatter_3d_plot(self) -> None:
        return px.scatter_3d(
            self.embedded_data_df,
            x="D1",
            y="D2",
            z="D3",
            color=self.target_col,
            labels={"color": "target"},
        )
