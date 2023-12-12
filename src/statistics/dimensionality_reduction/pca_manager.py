import typing as t

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA


class PCAManager:
    def __init__(self, max_n_components: int):
        self.max_n_components = max_n_components
        self.normalize: bool | None = None
        self.model: PCA | None = None
        self.target_col: pd.Series | None = None

    @property
    def params(self) -> t.Dict[str, int]:
        columns = st.columns(2)
        return {
            "n_components": columns[0].slider(
                label="Number of Components",
                min_value=1,
                max_value=self.max_n_components,
                value=3,
                step=1,
                help="Number of principal components to compute.",
            ),
            "normalize": (
                columns[1]
                .columns([0.5, 1, 0.5])[1]
                .toggle("Normalize data", value=False)
            ),
        }

    @st.cache_resource(show_spinner=True)
    def _get_model(_self, n_components: int) -> PCA:
        return PCA(n_components)

    def set_model(self) -> None:
        params = self.params
        self.model = self._get_model(params["n_components"])
        self.model.normalize = params["normalize"]

    @st.cache_resource(
        show_spinner=True,
        hash_funcs={PCA: lambda model: (model.n_components, model.normalize)},
    )
    def _compute_pca(
        _self, model: PCA, data: pd.DataFrame
    ) -> t.Tuple[pd.DataFrame, PCA]:
        data_normalized = (
            (data - data.mean()) / (data.std() + 1e-5) if model.normalize else data
        )
        components = model.fit_transform(data_normalized)

        return pd.DataFrame(
            components, columns=[f"PC{i+1}" for i in range(components.shape[1])]
        )

    def fit(self, data: pd.DataFrame, target_col: pd.Series):
        self.components_df = self._compute_pca(model=self.model, data=data)
        self.target_col = target_col

    def scatter_matrix_plot(self) -> None:
        return px.scatter_matrix(
            self.components_df, color=self.target_col, labels={"color": "target"}
        )

    def explained_variance_plot(self) -> None:
        exp_var_cumul = np.cumsum(self.model.explained_variance_ratio_)
        x_ticks = list(range(1, exp_var_cumul.shape[0] + 1))
        fig = px.bar(
            x=x_ticks,
            y=exp_var_cumul,
            labels={"x": "# Components", "y": "Explained Variance"},
        )
        fig.update_xaxes(tickvals=x_ticks, ticktext=list(map(str, x_ticks)))
        fig.add_trace(
            go.Scatter(
                x=x_ticks,
                y=exp_var_cumul,
                mode="lines+markers",
                line=dict(color="red", width=3),
                marker=dict(size=10),
                showlegend=False,
            )
        )
        return fig

    def scatter_2d_plot(self) -> None:
        return px.scatter(
            self.components_df,
            x="PC1",
            y="PC2",
            color=self.target_col,
            labels={"color": "target"},
        )

    def scatter_3d_plot(self) -> None:
        return px.scatter_3d(
            self.components_df,
            x="PC1",
            y="PC2",
            z="PC3",
            color=self.target_col,
            labels={"color": "target"},
        )

    def loadings_plot(self) -> None:
        loadings = self.model.components_.T * np.sqrt(self.model.explained_variance_)

        fig = px.scatter(
            self.components_df,
            x="PC1",
            y="PC2",
            color=self.target_col,
            labels={"color": "target"},
        )

        for i, feature in enumerate(self.components_df.columns):
            fig.add_annotation(
                ax=0,
                ay=0,
                axref="x",
                ayref="y",
                x=loadings[i, 0],
                y=loadings[i, 1],
                showarrow=True,
                arrowsize=2,
                arrowhead=2,
                xanchor="right",
                yanchor="top",
            )
            fig.add_annotation(
                x=loadings[i, 0],
                y=loadings[i, 1],
                ax=0,
                ay=0,
                xanchor="center",
                yanchor="bottom",
                text=feature,
                yshift=5,
            )
        return fig
