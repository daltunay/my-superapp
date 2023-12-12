import streamlit as st

import utils
from src.machine_learning.datasets import Dataset
from src.statistics.dimensionality_reduction import PCAManager

loader = utils.PageConfigLoader(__file__)
loader.set_page_config(globals())

logger = utils.CustomLogger(__file__)

st_ss = st.session_state


def main():
    utils.tabs_config()
    utils.show_source_code("src/statistics/dimensionality_rediction/pca_manager.py")

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

    st.subheader("PCA")
    with st.container(border=True):
        pca_manager = PCAManager(max_n_components=3)
        pca_manager.set_model()

    pca_manager.fit(data=X, target_col=y)

    st.subheader("Scatter matrix plot", divider="gray")
    st.plotly_chart(pca_manager.scatter_matrix_plot(), use_container_width=True)

    st.subheader("Explained variance plot", divider="gray")
    st.plotly_chart(pca_manager.explained_variance_plot(), use_container_width=True)

    st.subheader("Scatter 2D + Loadings plot", divider="gray")
    try:
        st.plotly_chart(pca_manager.loadings_plot(), use_container_width=True)
    except ValueError:
        st.error("Number of principal components not sufficient for the plot")

    st.subheader("Scatter 3D plot", divider="gray")
    try:
        st.plotly_chart(pca_manager.scatter_3d_plot(), use_container_width=True)
    except ValueError:
        st.error("Number of principal components not sufficient for the plot")
