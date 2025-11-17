import streamlit as st

st.set_page_config(
    page_title="Machine Learning Final Project",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Final Project by Kevin Kruzel")

st.markdown(
    """
    ### 🔴 About this Project
    
    This project explores and analyzes...
    
    **Pages:**
    - 🔴 **app** - Overview of the Project
    - 🔴 **EDA Gallery** - Exploratory data analysis of the dataset, complete with charts and initial observations
    - 🔴 **Prediction** - Uses machine learning model to predict
    
    Navigate using the sidebar to begin exploring the other pages of the project.
    """
)

st.caption("Built with Streamlit")
