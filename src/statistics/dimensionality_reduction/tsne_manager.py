import typing as t

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.manifold import TSNE


class TSNEManager:
    def __init__(self, max_n_components: int):
        self.max_n_components = max_n_components
        self.model: TSNE | None = None
        self.target_col: pd.Series | None = None
        self.embedded_data_df: pd.DataFrame | None = None

    @property
    def params(self) -> t.Dict[str, int]:
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
            "perplexity": columns[1].slider(
                label="Perplexity",
                min_value=1,
                max_value=100,
                value=30,
                step=1,
                help="A measure of how to balance attention between local and global aspects of the data.",
            ),
            "learning_rate": columns[2].slider(
                label="Learning Rate",
                min_value=10.0,
                max_value=500.0,
                value=200.0,
                step=50.0,
                help="Step size for each iteration in optimizing the cost function.",
            ),
        }

    @st.cache_resource(show_spinner=True)
    def _get_model(_self, params: t.Dict[str, int]) -> TSNE:
        return TSNE(
            n_components=params["n_components"],
            perplexity=params["perplexity"],
            learning_rate=params["learning_rate"],
        )

    def set_model(self) -> None:
        params = self.params
        self.model = self._get_model(params)

    @st.cache_resource(
        show_spinner=True,
        hash_funcs={
            TSNE: lambda model: (
                model.n_components,
                model.perplexity,
                model.learning_rate,
            )
        },
    )
    def _compute_tsne(_self, model: TSNE, data: pd.DataFrame) -> pd.DataFrame:
        embedded_data = model.fit_transform(data)
        column_names = [f"D{i}" for i in range(1, model.n_components + 1)]
        return pd.DataFrame(embedded_data, columns=column_names)

    def fit(self, data: pd.DataFrame, target_col: pd.Series):
        self.embedded_data_df = self._compute_tsne(model=self.model, data=data)
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
