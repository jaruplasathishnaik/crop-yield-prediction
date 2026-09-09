# 🌾 Crop Yield Prediction System

A machine learning web application built with **Python, Scikit-learn, Pandas, NumPy, Matplotlib, and Streamlit** to predict crop yield from agricultural, weather, and soil-related features.

> **Important:** This project currently generates a **synthetic dataset inside the application** for demonstration and learning purposes. The predictions should not be treated as real-world agricultural recommendations.

## 🚀 Features

- Generates a synthetic crop-yield dataset automatically.
- Preprocesses numerical and categorical features using Scikit-learn pipelines.
- Compares five regression algorithms:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Support Vector Regression (SVR)
  - Multi-Layer Perceptron Regressor (MLP)
- Evaluates models using:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R² Score
- Displays model comparison results.
- Provides visualizations for model performance and prediction analysis.
- Includes an interactive prediction form in the Streamlit interface.
- Handles categorical variables with `OneHotEncoder(handle_unknown="ignore")`.
- Scales numerical features with `StandardScaler`.

## 🧠 Machine Learning Workflow

The application follows this general workflow:

1. Generate the synthetic agricultural dataset.
2. Separate input features and target yield.
3. Split the data into training and testing sets.
4. Identify numerical and categorical columns.
5. Apply preprocessing:
   - Standard scaling for numerical features.
   - One-hot encoding for categorical features.
6. Train multiple regression models.
7. Evaluate each model using MAE, RMSE, and R².
8. Select the best-performing model.
9. Use the selected model for interactive crop-yield prediction.

## 📊 Input Features

The application uses features such as:

- Region
- Crop
- Rainfall
- Temperature
- Humidity
- Sunshine
- Soil pH
- Nitrogen
- Phosphorus
- Potassium
- Organic Matter
- Fertilizer
- Irrigation
- Pesticide
- Previous Year Yield

## 🛠️ Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Scikit-learn**
- **Streamlit**

## 📁 Project Structure

```text
crop-yield-prediction/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 💻 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/crop-yield-prediction.git
cd crop-yield-prediction
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📈 Model Evaluation

The application compares models using:

| Metric | Purpose |
|---|---|
| MAE | Average absolute prediction error |
| RMSE | Penalizes larger prediction errors more strongly |
| R² Score | Measures how well the model explains target variation |

The best model is selected based on the evaluation performed by the application.

## 🌱 Dataset

The current version does **not** require a separate dataset file.

The application creates a synthetic dataset programmatically using generated values for agricultural and environmental variables.

For a real-world version of this project, the synthetic data should be replaced or supplemented with a reliable agricultural dataset.

## 🔮 Future Improvements

Possible improvements include:

- Use a real crop-yield dataset.
- Add more crop and region categories.
- Add feature-importance analysis.
- Add cross-validation.
- Perform hyperparameter tuning.
- Save and load trained models.
- Add model explainability using SHAP or similar tools.
- Add real-time weather and soil data.
- Add user authentication.
- Deploy the application publicly.
- Add automated tests.
- Add screenshots and demonstration GIFs to the repository.

## ⚠️ Disclaimer

This project is intended for **educational and demonstration purposes**.

Because the current dataset is synthetic, predictions do not represent validated agricultural forecasts and should not be used to make farming, financial, or operational decisions.

## 👨‍💻 Author

**Your Name**

- GitHub: jaruplasathishnaik
-Linkedin:[Sathish Naik Jarupla]
 [(https://www.linkedin.com/in/sathish-naik-jarupla/)] 

Replace the author information with your own details before publishing.

## ⭐ Contributing

Contributions, suggestions, and improvements are welcome.

If you find this project useful, consider giving the repository a ⭐ on GitHub.

## 📄 License

This project is released under the MIT License. If you add a separate license file, update this section accordingly.
