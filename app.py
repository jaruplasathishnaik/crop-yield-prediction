import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🌾 Crop Yield Prediction")
st.write(
    """
    This application predicts crop yield using machine learning.
    
    Five different regression algorithms are trained and compared:
    
    - Linear Regression
    - Decision Tree
    - Random Forest
    - Support Vector Regression
    - Neural Network
    """
)


# ============================================================
# DATA GENERATION
# ============================================================

@st.cache_data
def generate_dataset(number_of_rows=2000):

    np.random.seed(42)

    regions = [
        "North",
        "South",
        "East",
        "West",
        "Central"
    ]

    crops = [
        "Wheat",
        "Rice",
        "Maize",
        "Cotton",
        "Sugarcane"
    ]

    data = []

    for i in range(number_of_rows):

        region = np.random.choice(regions)
        crop = np.random.choice(crops)

        # Weather features
        rainfall = np.random.uniform(400, 1800)
        temperature = np.random.uniform(18, 35)
        humidity = np.random.uniform(40, 90)
        sunshine = np.random.uniform(4, 11)

        # Soil features
        soil_ph = np.random.uniform(5.0, 8.0)

        nitrogen = np.random.uniform(30, 150)
        phosphorus = np.random.uniform(20, 100)
        potassium = np.random.uniform(20, 120)

        organic_matter = np.random.uniform(1.0, 6.0)

        # Farming features
        fertilizer = np.random.uniform(50, 300)
        irrigation = np.random.uniform(10, 100)
        pesticide = np.random.uniform(0.2, 5.0)

        # Previous year yield
        previous_yield = np.random.uniform(1.0, 7.0)

        # Crop-specific effects
        crop_effect = {
            "Wheat": 0.5,
            "Rice": 1.0,
            "Maize": 0.7,
            "Cotton": 0.3,
            "Sugarcane": 1.5
        }[crop]

        # Region-specific effects
        region_effect = {
            "North": 0.2,
            "South": 0.5,
            "East": 0.7,
            "West": 0.3,
            "Central": 0.4
        }[region]

        # Generate yield
        yield_value = (
            0.0025 * rainfall
            + 0.08 * nitrogen
            + 0.04 * phosphorus
            + 0.03 * potassium
            + 0.015 * fertilizer
            + 0.015 * irrigation
            + 0.25 * organic_matter
            + 0.30 * previous_yield
            + 0.10 * sunshine
            - 0.05 * abs(temperature - 25)
            - 0.15 * abs(soil_ph - 6.5)
            + crop_effect
            + region_effect
            + np.random.normal(0, 0.5)
        )

        # Prevent negative yield
        yield_value = max(yield_value, 0.1)

        data.append([
            region,
            crop,
            rainfall,
            temperature,
            humidity,
            sunshine,
            soil_ph,
            nitrogen,
            phosphorus,
            potassium,
            organic_matter,
            fertilizer,
            irrigation,
            pesticide,
            previous_yield,
            yield_value
        ])

    columns = [
        "Region",
        "Crop",
        "Rainfall_mm",
        "Avg_Temperature_C",
        "Humidity_pct",
        "Sunshine_Hours",
        "Soil_pH",
        "Nitrogen_kg_ha",
        "Phosphorus_kg_ha",
        "Potassium_kg_ha",
        "Organic_Matter_pct",
        "Fertilizer_kg_ha",
        "Irrigation_pct",
        "Pesticide_kg_ha",
        "Prev_Year_Yield_ton_ha",
        "Crop_Yield_ton_ha"
    ]

    df = pd.DataFrame(
        data,
        columns=columns
    )

    return df


# ============================================================
# LOAD DATASET
# ============================================================

df = generate_dataset()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Settings")

test_size = st.sidebar.slider(
    "Test data percentage",
    min_value=10,
    max_value=40,
    value=20,
    step=5
)

random_state = st.sidebar.number_input(
    "Random state",
    min_value=1,
    max_value=100,
    value=42
)


# ============================================================
# DATASET INFORMATION
# ============================================================

st.header("📊 Dataset")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Number of Rows",
        len(df)
    )

with col2:
    st.metric(
        "Number of Features",
        len(df.columns) - 1
    )

with col3:
    st.metric(
        "Target",
        "Crop Yield"
    )

st.dataframe(
    df.head(10),
    use_container_width=True
)


# ============================================================
# DATA PREPROCESSING
# ============================================================

categorical_features = [
    "Region",
    "Crop"
]

numeric_features = [
    "Rainfall_mm",
    "Avg_Temperature_C",
    "Humidity_pct",
    "Sunshine_Hours",
    "Soil_pH",
    "Nitrogen_kg_ha",
    "Phosphorus_kg_ha",
    "Potassium_kg_ha",
    "Organic_Matter_pct",
    "Fertilizer_kg_ha",
    "Irrigation_pct",
    "Pesticide_kg_ha",
    "Prev_Year_Yield_ton_ha"
]

target = "Crop_Yield_ton_ha"


X = df[
    categorical_features + numeric_features
]

y = df[target]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size / 100,
    random_state=int(random_state)
)


# ============================================================
# PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[

        (
            "numeric",
            StandardScaler(),
            numeric_features
        ),

        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


# ============================================================
# FIVE MACHINE LEARNING MODELS
# ============================================================

models = {

    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=10,
        min_samples_split=5,
        random_state=int(random_state)
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        max_depth=15,
        min_samples_split=4,
        random_state=int(random_state),
        n_jobs=-1
    ),

    "Support Vector Regression": SVR(
        kernel="rbf",
        C=10,
        epsilon=0.1
    ),

    "Neural Network": MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=1000,
        early_stopping=True,
        random_state=int(random_state)
    )
}


# ============================================================
# TRAIN MODELS
# ============================================================

st.header("🤖 Model Training")

if st.button("🚀 Train All 5 Models"):

    results = []

    progress = st.progress(0)

    for index, (model_name, model) in enumerate(models.items()):

        pipeline = Pipeline(
            steps=[
                ("preprocessing", preprocessor),
                ("model", model)
            ]
        )

        pipeline.fit(
            X_train,
            y_train
        )

        predictions = pipeline.predict(
            X_test
        )

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        results.append({
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2 Score": r2
        })

        progress.progress(
            (index + 1) / len(models)
        )

    results_df = pd.DataFrame(
        results
    )

    # Sort by R2 score
    results_df = results_df.sort_values(
        by="R2 Score",
        ascending=False
    )

    # Save results in session state
    st.session_state["results"] = results_df

    # Find best model
    best_model_name = results_df.iloc[0]["Model"]

    st.session_state[
        "best_model_name"
    ] = best_model_name

    # Train best model again
    best_model = models[
        best_model_name
    ]

    best_pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", best_model)
        ]
    )

    best_pipeline.fit(
        X_train,
        y_train
    )

    st.session_state[
        "best_pipeline"
    ] = best_pipeline

    st.success(
        f"Training completed! Best model: {best_model_name}"
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

if "results" in st.session_state:

    st.header("📈 Model Comparison")

    results_df = st.session_state[
        "results"
    ]

    st.dataframe(
        results_df.style.format({
            "MAE": "{:.3f}",
            "RMSE": "{:.3f}",
            "R2 Score": "{:.4f}"
        }),
        use_container_width=True
    )

    best_model_name = st.session_state[
        "best_model_name"
    ]

    st.success(
        f"🏆 Best Model: **{best_model_name}**"
    )

    # --------------------------------------------------------
    # R2 SCORE GRAPH
    # --------------------------------------------------------

    st.subheader("R² Score Comparison")

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.bar(
        results_df["Model"],
        results_df["R2 Score"]
    )

    ax.set_xlabel(
        "Machine Learning Model"
    )

    ax.set_ylabel(
        "R² Score"
    )

    ax.set_title(
        "R² Score Comparison"
    )

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.tight_layout()

    st.pyplot(fig)

    # --------------------------------------------------------
    # RMSE GRAPH
    # --------------------------------------------------------

    st.subheader("RMSE Comparison")

    fig2, ax2 = plt.subplots(
        figsize=(10, 5)
    )

    ax2.bar(
        results_df["Model"],
        results_df["RMSE"]
    )

    ax2.set_xlabel(
        "Machine Learning Model"
    )

    ax2.set_ylabel(
        "RMSE"
    )

    ax2.set_title(
        "RMSE Comparison"
    )

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.tight_layout()

    st.pyplot(fig2)

    # --------------------------------------------------------
    # MAE GRAPH
    # --------------------------------------------------------

    st.subheader("MAE Comparison")

    fig3, ax3 = plt.subplots(
        figsize=(10, 5)
    )

    ax3.bar(
        results_df["Model"],
        results_df["MAE"]
    )

    ax3.set_xlabel(
        "Machine Learning Model"
    )

    ax3.set_ylabel(
        "MAE"
    )

    ax3.set_title(
        "Mean Absolute Error Comparison"
    )

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.tight_layout()

    st.pyplot(fig3)


# ============================================================
# DATA VISUALIZATION
# ============================================================

st.header("📊 Data Visualization")


# Yield distribution
st.subheader("Crop Yield Distribution")

fig4, ax4 = plt.subplots(
    figsize=(10, 5)
)

ax4.hist(
    df["Crop_Yield_ton_ha"],
    bins=30
)

ax4.set_xlabel(
    "Crop Yield (ton/ha)"
)

ax4.set_ylabel(
    "Frequency"
)

ax4.set_title(
    "Distribution of Crop Yield"
)

st.pyplot(fig4)


# Rainfall vs Yield
st.subheader("Rainfall vs Crop Yield")

fig5, ax5 = plt.subplots(
    figsize=(10, 5)
)

ax5.scatter(
    df["Rainfall_mm"],
    df["Crop_Yield_ton_ha"],
    alpha=0.5
)

ax5.set_xlabel(
    "Rainfall (mm)"
)

ax5.set_ylabel(
    "Crop Yield (ton/ha)"
)

ax5.set_title(
    "Rainfall vs Crop Yield"
)

st.pyplot(fig5)


# Crop average yield
st.subheader("Average Yield by Crop")

crop_average = (
    df.groupby("Crop")[
        "Crop_Yield_ton_ha"
    ]
    .mean()
    .sort_values(
        ascending=False
    )
)

fig6, ax6 = plt.subplots(
    figsize=(10, 5)
)

ax6.bar(
    crop_average.index,
    crop_average.values
)

ax6.set_xlabel(
    "Crop"
)

ax6.set_ylabel(
    "Average Yield (ton/ha)"
)

ax6.set_title(
    "Average Crop Yield"
)

st.pyplot(fig6)


# ============================================================
# PREDICTION SECTION
# ============================================================

st.header("🌱 Predict Crop Yield")

if "best_pipeline" not in st.session_state:

    st.warning(
        "Please click 'Train All 5 Models' before making a prediction."
    )

else:

    best_pipeline = st.session_state[
        "best_pipeline"
    ]

    st.write(
        f"Using the best model: "
        f"**{st.session_state['best_model_name']}**"
    )

    # --------------------------------------------------------
    # INPUT FORM
    # --------------------------------------------------------

    with st.form("prediction_form"):

        st.subheader(
            "Enter Farm Information"
        )

        col1, col2 = st.columns(2)

        # ----------------------------------------------------
        # Categorical inputs
        # ----------------------------------------------------

        with col1:

            region = st.selectbox(
                "Region",
                [
                    "North",
                    "South",
                    "East",
                    "West",
                    "Central"
                ]
            )

            crop = st.selectbox(
                "Crop",
                [
                    "Wheat",
                    "Rice",
                    "Maize",
                    "Cotton",
                    "Sugarcane"
                ]
            )

            rainfall = st.number_input(
                "Rainfall (mm)",
                min_value=0.0,
                max_value=3000.0,
                value=1000.0
            )

            temperature = st.number_input(
                "Average Temperature (°C)",
                min_value=0.0,
                max_value=50.0,
                value=25.0
            )

            humidity = st.number_input(
                "Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=65.0
            )

            sunshine = st.number_input(
                "Sunshine Hours",
                min_value=0.0,
                max_value=24.0,
                value=8.0
            )

            soil_ph = st.number_input(
                "Soil pH",
                min_value=0.0,
                max_value=14.0,
                value=6.5
            )

        # ----------------------------------------------------
        # Numerical inputs
        # ----------------------------------------------------

        with col2:

            nitrogen = st.number_input(
                "Nitrogen (kg/ha)",
                min_value=0.0,
                max_value=300.0,
                value=100.0
            )

            phosphorus = st.number_input(
                "Phosphorus (kg/ha)",
                min_value=0.0,
                max_value=200.0,
                value=50.0
            )

            potassium = st.number_input(
                "Potassium (kg/ha)",
                min_value=0.0,
                max_value=250.0,
                value=60.0
            )

            organic_matter = st.number_input(
                "Organic Matter (%)",
                min_value=0.0,
                max_value=20.0,
                value=3.0
            )

            fertilizer = st.number_input(
                "Fertilizer (kg/ha)",
                min_value=0.0,
                max_value=500.0,
                value=170.0
            )

            irrigation = st.number_input(
                "Irrigation (%)",
                min_value=0.0,
                max_value=100.0,
                value=70.0
            )

            pesticide = st.number_input(
                "Pesticide (kg/ha)",
                min_value=0.0,
                max_value=20.0,
                value=2.0
            )

            previous_yield = st.number_input(
                "Previous Year Yield (ton/ha)",
                min_value=0.0,
                max_value=20.0,
                value=3.5
            )

        submitted = st.form_submit_button(
            "🔮 Predict Crop Yield"
        )


    # ========================================================
    # MAKE PREDICTION
    # ========================================================

    if submitted:

        input_data = pd.DataFrame([{

            "Region": region,

            "Crop": crop,

            "Rainfall_mm": rainfall,

            "Avg_Temperature_C": temperature,

            "Humidity_pct": humidity,

            "Sunshine_Hours": sunshine,

            "Soil_pH": soil_ph,

            "Nitrogen_kg_ha": nitrogen,

            "Phosphorus_kg_ha": phosphorus,

            "Potassium_kg_ha": potassium,

            "Organic_Matter_pct": organic_matter,

            "Fertilizer_kg_ha": fertilizer,

            "Irrigation_pct": irrigation,

            "Pesticide_kg_ha": pesticide,

            "Prev_Year_Yield_ton_ha": previous_yield
        }])

        prediction = best_pipeline.predict(
            input_data
        )[0]

        st.success(
            "Prediction completed successfully!"
        )

        st.metric(
            "Predicted Crop Yield",
            f"{prediction:.2f} ton/ha"
        )

        # Yield interpretation
        if prediction < 3:

            st.info(
                "🌱 The predicted yield is relatively low."
            )

        elif prediction < 5:

            st.info(
                "🌾 The predicted yield is moderate."
            )

        else:

            st.success(
                "🌾 The predicted yield is high."
            )


# ============================================================
# FEATURE INFORMATION
# ============================================================

st.header("📚 Feature Information")

feature_information = pd.DataFrame({

    "Feature": [
        "Rainfall",
        "Temperature",
        "Humidity",
        "Sunshine",
        "Soil pH",
        "Nitrogen",
        "Phosphorus",
        "Potassium",
        "Organic Matter",
        "Fertilizer",
        "Irrigation",
        "Pesticide",
        "Previous Year Yield"
    ],

    "Description": [
        "Amount of rainfall received by the farm.",
        "Average temperature during crop growth.",
        "Average atmospheric humidity.",
        "Daily sunshine duration.",
        "Acidity or alkalinity of soil.",
        "Nitrogen applied to soil.",
        "Phosphorus applied to soil.",
        "Potassium applied to soil.",
        "Percentage of organic matter in soil.",
        "Amount of fertilizer used.",
        "Percentage of irrigation coverage.",
        "Amount of pesticide used.",
        "Yield produced during the previous year."
    ]
})

st.dataframe(
    feature_information,
    use_container_width=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.write(
    """
    🌾 **Crop Yield Prediction System**
    
    Built with Python, Pandas, NumPy, Scikit-learn,
    Matplotlib and Streamlit.
    """
)