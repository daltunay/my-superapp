import streamlit as st

import utils
from src.machine_learning.clustering import KMeansManager
from src.machine_learning.datasets import Dataset

loader = utils.PageConfigLoader(__file__)
loader.set_page_config(globals())

logger = utils.CustomLogger(__file__)

st_ss = st.session_state


def main():
    utils.tabs_config()
    utils.show_source_code("src/statistics/dimensionality_rediction/kmeans_manager.py")

    st.header("Dataset", divider="gray")
    dataset = Dataset(type=None)
    raw_dataset_dict = Dataset.get_dataset(**dataset.params, split=False)
    dataset.set(raw_dataset_dict)

    with st.expander(label="Dataset description"):
        st.markdown(dataset.description)

    X, y = dataset.X, dataset.y
    if label_mapping := dataset.label_mapping:
        y = y.map(label_mapping)

    st.subheader("Visualize data")
    with st.container(border=True):
        utils.display_tab_content("data", X, y)

    st.subheader("K-Means")
    with st.container(border=True):
        kmeans_manager = KMeansManager(max_n_clusters=10)
        kmeans_manager.set_model()

    kmeans_manager.fit(data=X)

    st.subheader("Scatter plot", divider="gray")
    col_x, col_y = st.columns(2)
    x_col_scatter = col_x.selectbox(
        label="X column", key="scatter_x", options=X.columns, index=0
    )
    y_col_scatter = col_y.selectbox(
        label="Y column", key="scatter_y", options=X.columns, index=1
    )
    st.plotly_chart(
        kmeans_manager.scatter_plot(x_col_scatter, y_col_scatter),
        use_container_width=True,
    )

    st.subheader("Centroids plot", divider="gray")
    col_x, col_y = st.columns(2)
    x_col_centroids = col_x.selectbox(
        label="X column", key="centroids_x", options=X.columns, index=0
    )
    y_col_centroids = col_y.selectbox(
        label="Y column", key="centroids_y", options=X.columns, index=1
    )
    st.plotly_chart(
        kmeans_manager.centroids_plot(x_col_centroids, y_col_centroids),
        use_container_width=True,
    )
