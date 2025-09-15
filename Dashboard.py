import streamlit as st
from PIL import Image
import base64

# --- Page Config (must be first Streamlit command) ---
st.set_page_config(
    page_title="Breast Cancer Prediction System",
    page_icon="🧬",
    layout="wide"
)

# --- Hide Sidebar Completely ---
st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none; }
        [data-testid="collapsedControl"] { display: none; }
    </style>
""", unsafe_allow_html=True)

# --- Load & Encode Banner Image ---
def get_image_as_base64(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

banner_img = get_image_as_base64("banner.jpg")  

# --- Display Full-Width Banner Image at Top ---
st.markdown(
    f"""
    <style>
        .banner {{
            width: 100%;
            height: 250px;
            border-radius: 10px;
            margin-top: -50px;
            margin-bottom: 20px;
        }}
    </style>
    <img src="data:image/png;base64,{banner_img}" class="banner">
    """,
    unsafe_allow_html=True
)

# --- Title Section ---
st.markdown(
    """
    <div style="text-align:center;">
        <h1 style="color:#5B9AA0; font-size: 48px;">🧬 Breast Cancer Prediction System</h1>
        <h4 style="color:#7D7D7D;">A Web Application for Early Detection Using Machine Learning</h4>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Overview Section with Softer Pastel Background ---
st.markdown("---")
st.markdown(
    """
    <div style="padding: 25px;">
        <h3 style="color: #D46A6A;">📌 Project Overview</h3>
        <p style="color:white; font-size: 17px;">
            This application is designed to assist in the early detection of breast cancer using machine learning techniques.
            It guides users through data analysis, model training, SVM optimization, and classifier comparison —
            all through an intuitive and visual interface.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Navigation Buttons ---
st.markdown("### 🔽 Explore Each Phase")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🔍 Identify Problem"):
        st.switch_page("pages/1_Identify_problem_and_data_clean.py")
with col2:
    if st.button("📊 Exploratory_Data_Analysis"):
        st.switch_page("pages/2_Exploratory_Data_Analysis.py")
with col3:
    if st.button("🧹 Preprocessing"):
        st.switch_page("pages/3_Data_PreProcessing.py")

col4, col5, col6 = st.columns(3)
with col4:
    if st.button("🤖 SVM Model"):
        st.switch_page("pages/4_Predictive_Model_Using_SVM.py")
with col5:
    if st.button("⚙️ Optimize SVM"):
        st.switch_page("pages/5_Optimizing_SVM_Classifier.py")
with col6:
    if st.button("📈 Compare Models"):
        st.switch_page("pages/6_Comparision_between_different_Classifiers.py")

col7, _, col8 = st.columns([1, 1, 1])
with col7:
    if st.button("🧪 Test Model"):
        st.switch_page("pages/7_Test_Model.py")

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; color:#A9A9A9; font-size: 25px;padding-top:30px'>
        <em>“Teaching computers to save lives: ML for proactive breast cancer diagnosis.”</em>
    </div>
    """,
    unsafe_allow_html=True
)