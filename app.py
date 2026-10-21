
from pathlib import Path
import json

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


# ---------- Project paths ----------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "house_price_model.joblib"
METRICS_PATH = BASE_DIR / "outputs" / "metrics.json"
PREDICTIONS_PATH = BASE_DIR / "outputs" / "test_predictions.csv"
DATA_PATH = BASE_DIR / "data" / "train.csv"


# ---------- Page configuration ----------
st.set_page_config(
    page_title="EstateIQ | House Price Analytics",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- Styling ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    .stApp {
        background: #f4f7fb;
        color: #182338;
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stSidebar"] {
        background: #101b32;
    }

    [data-testid="stSidebar"] * {
        color: #eef3ff;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    .hero {
        padding: 27px 30px;
        border-radius: 18px;
        background: linear-gradient(120deg, #142544, #234d79);
        color: white;
        margin-bottom: 24px;
    }

    .hero h1 {
        font-family: 'Manrope', sans-serif;
        color: white;
        font-size: 31px;
        margin: 0 0 8px 0;
        font-weight: 800;
    }

    .hero p {
        color: #d8e5f7;
        margin: 0;
        font-size: 15px;
    }

    .section-title {
        font-family: 'Manrope', sans-serif;
        color: #182338;
        font-size: 22px;
        font-weight: 800;
        margin: 12px 0 15px 0;
    }

    .small-note {
        color: #66758c;
        font-size: 13px;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e2e9f2;
        padding: 17px;
        border-radius: 14px;
        box-shadow: 0 3px 12px rgba(24, 45, 78, 0.04);
    }

    div[data-testid="stMetricLabel"] {
        color: #63728a;
    }

    div[data-testid="stMetricValue"] {
        color: #173b67;
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
    }

    div[data-testid="stForm"] {
        background: white;
        padding: 22px;
        border: 1px solid #e2e9f2;
        border-radius: 16px;
    }

    .result-card {
        background: #e8f3ff;
        border: 1px solid #c7e0ff;
        border-radius: 14px;
        padding: 23px;
        margin-top: 16px;
    }

    .result-label {
        color: #42638a;
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .result-price {
        color: #123c70;
        font-family: 'Manrope', sans-serif;
        font-size: 32px;
        font-weight: 800;
        margin-top: 6px;
    }

    .stButton button, .stFormSubmitButton button {
        background: #245a91;
        color: white;
        border: 0;
        border-radius: 9px;
        font-weight: 700;
        min-height: 44px;
    }

    .stButton button:hover, .stFormSubmitButton button:hover {
        background: #173f6c;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Load saved artifacts ----------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    with open(METRICS_PATH, encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_predictions():
    return pd.read_csv(PREDICTIONS_PATH)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH, usecols=["SalePrice", "GrLivArea"])


required_paths = [
    MODEL_PATH,
    METRICS_PATH,
    PREDICTIONS_PATH,
    DATA_PATH,
]

if not all(path.exists() for path in required_paths):
    st.error(
        "Project files are missing. Run `python train_model.py` "
        "from the project terminal first."
    )
    st.stop()

try:
    model = load_model()
    metrics = load_metrics()
    predictions = load_predictions()
    dataset = load_dataset()
except Exception as error:
    st.error(f"Unable to load project files: {error}")
    st.stop()


model_name = metrics.get("model", "Selected Model")
features = metrics.get(
    "features",
    ["GrLivArea", "BedroomAbvGr", "FullBath"],
)


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🏡 EstateIQ")
    st.caption("HOUSE PRICE INTELLIGENCE")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Price Prediction",
            "Model Insights",
            "Dataset Explorer",
        ],
    )

    st.divider()
    st.markdown("**MODEL STATUS**")
    st.success("Trained model loaded")
    st.caption(f"Algorithm: {model_name}")
    st.caption(f"Features: {len(features)}")
    st.caption("Evaluation: held-out test set")
    st.markdown("---")
    st.caption(
        "Academic ML project · Not a formal property valuation"
    )


# ---------- Main header ----------
st.markdown(
    """
    <div class="hero">
        <h1>House Price Intelligence</h1>
        <p>
            Explore property data, estimate a sale price, and inspect
            the performance of a trained machine learning model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------- Reusable metric cards ----------
def metric_cards():
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Dataset Records",
        f"{metrics['dataset_rows']:,}",
    )
    col2.metric(
        "Test Records",
        f"{metrics['testing_rows']:,}",
    )
    col3.metric(
        "Mean Absolute Error",
        f"${metrics['mae']:,.0f}",
    )
    col4.metric(
        "R² Score",
        f"{metrics['r2']:.3f}",
    )


# ---------- Reusable prediction form ----------
def prediction_form():
    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.markdown(
            '<div class="section-title">Property Details</div>',
            unsafe_allow_html=True,
        )
        st.write(
            "Enter the property's basic features to estimate its price."
        )

        with st.form("prediction_form"):
            area = st.number_input(
                "Living area (square feet)",
                min_value=300,
                max_value=10000,
                value=1800,
                step=50,
                help="Above-ground living area.",
            )

            bedrooms = st.number_input(
                "Bedrooms",
                min_value=0,
                max_value=10,
                value=3,
                step=1,
            )

            bathrooms = st.number_input(
                "Full bathrooms",
                min_value=0,
                max_value=6,
                value=2,
                step=1,
            )

            submitted = st.form_submit_button(
                "Predict House Price",
                use_container_width=True,
            )

        if submitted:
            input_features = pd.DataFrame(
                [[area, bedrooms, bathrooms]],
                columns=features,
            )

            price = float(model.predict(input_features)[0])

            st.session_state["predicted_price"] = price
            st.session_state["submitted_features"] = {
                "Living area": int(area),
                "Bedrooms": int(bedrooms),
                "Full bathrooms": int(bathrooms),
            }

    with right:
        st.markdown(
            '<div class="section-title">Estimated Valuation</div>',
            unsafe_allow_html=True,
        )

        if "predicted_price" in st.session_state:
            price = st.session_state["predicted_price"]

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Predicted sale price</div>
                    <div class="result-price">${price:,.0f}</div>
                    <p style="color:#42638a; margin-bottom:0;">
                        Model estimate based on three property features.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("**Submitted property details**")

            for label, value in st.session_state[
                "submitted_features"
            ].items():
                st.write(f"- {label}: {value:,}")

        else:
            st.info(
                "Your estimated price will appear here after you "
                "submit the property details."
            )

        st.caption(
            "This is an educational estimate, not a guaranteed market "
            "price. Location, property condition, and other important "
            "factors are not included in this model."
        )


# ---------- Overview page ----------
if page == "Overview":
    metric_cards()

    st.markdown(
        '<div class="section-title">Explore the Dataset</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")

    with left:
        fig = px.histogram(
            dataset,
            x="SalePrice",
            nbins=35,
            title="Distribution of Sale Prices",
            labels={
                "SalePrice": "Sale Price ($)",
                "count": "Properties",
            },
            color_discrete_sequence=["#326da8"],
            template="plotly_white",
        )
        fig.update_layout(title_x=0.05)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.scatter(
            dataset,
            x="GrLivArea",
            y="SalePrice",
            title="Living Area vs. Sale Price",
            labels={
                "GrLivArea": "Living Area (sq ft)",
                "SalePrice": "Sale Price ($)",
            },
            color_discrete_sequence=["#326da8"],
            opacity=0.7,
            template="plotly_white",
        )
        fig.update_layout(title_x=0.05)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="section-title">Try a Prediction</div>',
        unsafe_allow_html=True,
    )
    prediction_form()


# ---------- Price prediction page ----------
elif page == "Price Prediction":
    prediction_form()


# ---------- Model insights page ----------
elif page == "Model Insights":
    metric_cards()

    st.markdown("### Selected Model")
    st.info(f"The current best model is **{model_name}**.")

    st.markdown(
        '<div class="section-title">Actual vs. Predicted</div>',
        unsafe_allow_html=True,
    )

    fig = px.scatter(
        predictions,
        x="Actual Price",
        y="Predicted Price",
        title=f"Test Set: Actual vs. Predicted — {model_name}",
        labels={
            "Actual Price": "Actual Sale Price ($)",
            "Predicted Price": "Predicted Sale Price ($)",
        },
        opacity=0.75,
        color_discrete_sequence=["#326da8"],
        template="plotly_white",
    )

    lower = min(
        predictions["Actual Price"].min(),
        predictions["Predicted Price"].min(),
    )
    upper = max(
        predictions["Actual Price"].max(),
        predictions["Predicted Price"].max(),
    )

    fig.add_shape(
        type="line",
        x0=lower,
        y0=lower,
        x1=upper,
        y1=upper,
        line={"color": "#d64a4a", "dash": "dash"},
    )

    fig.update_layout(title_x=0.05)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Model Comparison")

    comparison = metrics.get("model_comparison", [])

    if comparison:
        comparison_df = pd.DataFrame(comparison)
        comparison_df = comparison_df.rename(
            columns={
                "model": "Model",
                "mae": "MAE ($)",
                "rmse": "RMSE ($)",
                "r2": "R² Score",
            }
        )

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Lower MAE and RMSE indicate smaller prediction errors. "
            "Higher R² indicates more variation explained on this test set."
        )
    else:
        st.warning(
            "No model comparison is saved yet. Run train_model.py "
            "to generate the comparison."
        )

    st.markdown("### How to Interpret the Results")

    st.write(
        f"- **MAE:** ${metrics['mae']:,.0f}. "
        "The average absolute difference between predicted and actual "
        "sale prices on the test set."
    )

    st.write(
        f"- **RMSE:** ${metrics['rmse']:,.0f}. "
        "This metric penalizes larger prediction errors more strongly."
    )

    st.write(
        f"- **R²:** {metrics['r2']:.3f}. "
        "This measures the proportion of target variation explained "
        "by the model on the test set."
    )

    st.caption(
        "These metrics come from a held-out split of the supplied dataset. "
        "They do not guarantee accuracy for current market prices."
    )


# ---------- Dataset explorer page ----------
elif page == "Dataset Explorer":
    st.markdown(
        '<div class="section-title">Dataset Summary</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Properties", f"{len(dataset):,}")
    col2.metric(
        "Median Sale Price",
        f"${dataset['SalePrice'].median():,.0f}",
    )
    col3.metric(
        "Median Living Area",
        f"{dataset['GrLivArea'].median():,.0f} sq ft",
    )

    st.markdown("### Preview")
    st.dataframe(
        dataset.head(20),
        use_container_width=True,
    )

    st.markdown("### Descriptive Statistics")
    st.dataframe(
        dataset.describe().round(2),
        use_container_width=True,
    )

    st.download_button(
        "Download Dataset Preview",
        data=dataset.head(100).to_csv(index=False),
        file_name="house_price_dataset_preview.csv",
        mime="text/csv",
    )


# ---------- Footer ----------
st.divider()
st.caption(
    f"EstateIQ · House Price Prediction · {model_name} · "
    "Built for educational and portfolio purposes."
)
