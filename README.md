# 🫀 Heart Disease Prediction System

An end-to-end machine learning system for predicting heart disease risk using clinical patient data. Built with Python, scikit-learn, XGBoost, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-red)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![License](https://img.shields.io/badge/License-MIT-green)
=
## 🎯 Overview

This project implements a complete machine learning pipeline for heart disease prediction, from data preprocessing and model training to deployment via an interactive web application. The system analyzes 13 clinical features to predict the likelihood of heart disease in patients.

**Key Highlights:**
- ✅ Multiple ML algorithms compared (Logistic Regression, Random Forest, XGBoost)
- ✅ Comprehensive model evaluation with cross-validation
- ✅ Interactive web application with real-time predictions
- ✅ Feature importance analysis and visualizations
- ✅ Docker support for easy reproducibility
- ✅ Production-ready code with proper documentation

## ✨ Features

### Machine Learning Pipeline
- **Data Preprocessing:** StandardScaler normalization, train-test split with stratification
- **Model Training:** Multiple algorithms with hyperparameter tuning
- **Evaluation:** Accuracy, AUC-ROC, cross-validation, confusion matrix
- **Feature Analysis:** Importance ranking and correlation analysis

### Web Application
- **Interactive UI:** Clean, modern interface built with Streamlit
- **Real-time Predictions:** Instant risk assessment with probability scores
- **Risk Factor Analysis:** Personalized identification of cardiovascular risk factors
- **Clinical Recommendations:** Tailored advice based on prediction results
- **Visualizations:** Model performance metrics and data insights
- **Report Generation:** Downloadable prediction reports in JSON format

### Reproducibility
- **Docker Support:** Complete containerization for consistent deployment
- **Version Control:** All dependencies pinned for reproducibility
- **Setup Scripts:** Automated environment setup

## 📊 Dataset

**Source:** [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease) / [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

**Samples:** 303 patients  
**Features:** 13 clinical measurements  
**Target:** Binary classification (0 = No disease, 1 = Disease)

### Features Description

| Feature | Description | Range/Type |
|---------|-------------|------------|
| `age` | Age in years | 29-77 |
| `sex` | Sex (1=male, 0=female) | Binary |
| `cp` | Chest pain type | 0-3 |
| `trestbps` | Resting blood pressure (mm Hg) | 94-200 |
| `chol` | Serum cholesterol (mg/dl) | 126-564 |
| `fbs` | Fasting blood sugar > 120 mg/dl | Binary |
| `restecg` | Resting ECG results | 0-2 |
| `thalach` | Maximum heart rate achieved | 71-202 |
| `exang` | Exercise induced angina | Binary |
| `oldpeak` | ST depression induced by exercise | 0-6.2 |
| `slope` | Slope of peak exercise ST segment | 0-2 |
| `ca` | Number of major vessels (0-3) | 0-3 |
| `thal` | Thalassemia | 0-3 |


## 🚀 Quickstart

### Prerequisites

- Python 3.8+ OR Docker
- 2GB free disk space
- Internet connection (for initial setup)

## Docker — primary

### 1) Clone the repo
```bash
git clone https://github.com/VaniPant/Heart-Disease-Prediction-System.git
cd Heart-Disease-Prediction-System
```

### 2) Build the image
```bash
docker build -t heart-disease-app .
```

### 3) Run the container
```bash
docker run --rm -p 8501:8501 heart-disease-app
```

### 4) Open the app in your browser
```bash
http://localhost:8501
```

---

## pip — secondary

```bash
git clone https://github.com/VaniPant/Heart-Disease-Prediction-System.git
cd Heart-Disease-Prediction-System

python -m venv venv

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# Linux/Mac
# source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

streamlit run app.py
```

Open:
```bash
http://localhost:8501
```

---

## 🏋️ Training (optional)

If you want to retrain models locally:
```bash
python train_model.py
```

Typical outputs:
- Models saved in `models/`
- Plots saved in `figures/`


**Access the app:**  
Open your browser and navigate to `http://localhost:8501`

### Making Predictions

1. Open the web application
2. Navigate to the "🔍 Prediction" tab
3. Enter patient clinical data in the input fields
4. Click "🔮 Predict Heart Disease Risk"
5. View results, risk factors, and recommendations
6. (Optional) Save prediction report

## 📁 Project Structure
```
heart-disease-prediction/
│
├── data/
│   └── heart.csv                    # Dataset (download separately)
│
├── models/
│   ├── best_model.pkl               # Best performing model
│   ├── scaler.pkl                   # Feature scaler
│   ├── logistic_regression.pkl      # Logistic Regression model
│   ├── random_forest.pkl            # Random Forest model
│   ├── xgboost.pkl                  # XGBoost model
│   ├── metadata.json                # Model metadata
│   └── model_comparison.csv         # Model comparison results
│
├── figures/
│   ├── eda_overview.png             # EDA visualizations
│   ├── target_correlation.png       # Feature correlations
│   ├── confusion_matrix.png         # Confusion matrix
│   ├── roc_curves.png               # ROC curves
│   ├── feature_importance.png       # Feature importance
│   └── model_comparison.png         # Model comparison plot
│
├── train_model.py                   # Model training script
├── app.py                           # Streamlit web application
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Docker configuration
├── docker-compose.yml               # Docker Compose configuration
├── .dockerignore                    # Docker ignore file
├── .gitignore                       # Git ignore file
├── setup.sh                         # Setup script (Linux/Mac)
└── README.md                        # This file
```

## 🛠️ Technologies Used

### Core ML Libraries
- **scikit-learn 1.3.2:** Model training and evaluation
- **XGBoost 2.0.3:** Gradient boosting algorithm
- **pandas 2.1.4:** Data manipulation
- **numpy 1.26.2:** Numerical computing

### Visualization
- **matplotlib 3.8.2:** Plotting
- **seaborn 0.13.0:** Statistical visualizations

### Web Application
- **Streamlit 1.29.0:** Interactive web interface

### Deployment
- **Docker:** Containerization
- **Docker Compose:** Multi-container orchestration

## 🔬 Model Details

### Algorithms Implemented

1. **Logistic Regression**
   - Linear classification model
   - Fast training and inference
   - Interpretable coefficients

2. **Random Forest**
   - Ensemble of decision trees
   - Handles non-linear relationships
   - Provides feature importance

3. **XGBoost**
   - Gradient boosting algorithm
   - Often best performance
   - Regularization to prevent overfitting

### Training Process

1. **Data Split:** 80% train, 20% test (stratified)
2. **Feature Scaling:** StandardScaler normalization
3. **Cross-Validation:** 5-fold CV for robust evaluation
4. **Model Selection:** Based on AUC-ROC score
5. **Evaluation:** Multiple metrics (accuracy, AUC, sensitivity, specificity)

### Feature Engineering

- StandardScaler normalization for all features
- No missing values in dataset
- All features used (no selection performed)
- Potential for future feature engineering

## 📊 Visualizations

The project generates multiple visualizations:

1. **EDA Overview:** Target distribution, age distribution, correlation matrix, chest pain analysis
2. **Target Correlation:** Feature correlation with heart disease
3. **Confusion Matrix:** True/False positives and negatives
4. **ROC Curves:** Model comparison using ROC-AUC
5. **Feature Importance:** Most influential features
6. **Model Comparison:** Accuracy and AUC across models

All visualizations are saved in `figures/` and accessible via the web app.

## 🔮 Future Improvements

### Model Enhancements
- [ ] Implement deep learning models (Neural Networks)
- [ ] Hyperparameter tuning with GridSearchCV/RandomizedSearchCV
- [ ] Feature engineering (interaction terms, polynomial features)
- [ ] Handle class imbalance with SMOTE
- [ ] Ensemble methods (stacking, voting)

### Explainability
- [ ] SHAP values for model interpretability
- [ ] LIME for local explanations
- [ ] Partial dependence plots
- [ ] Individual prediction explanations

### Application Features
- [ ] User authentication and patient database
- [ ] Historical predictions tracking
- [ ] PDF report generation
- [ ] Email notifications
- [ ] Multi-language support
- [ ] Mobile responsive design

### Deployment
- [ ] Deploy to cloud (AWS, GCP, Azure)
- [ ] CI/CD pipeline
- [ ] API endpoints (FastAPI/Flask)
- [ ] Monitoring and logging
- [ ] A/B testing framework

### Data
- [ ] Expand to larger datasets
- [ ] Real-time data integration
- [ ] External data sources (EHR systems)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please ensure:
- Code follows PEP 8 style guide
- All tests pass
- Documentation is updated
- Commit messages are descriptive

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Vani Pant**  
Master's Student in Data Science  
University of Luxembourg

- GitHub: https://github.com/VaniPant
- Email: 0242319112@uni.lu

## ⚠️ Disclaimer

**IMPORTANT: This application is for educational and research purposes ONLY.**

- This tool is NOT a substitute for professional medical advice, diagnosis, or treatment
- Always consult with a qualified healthcare provider for medical decisions
- The model's predictions should not be used for clinical decision-making
- The developers assume no liability for any use of this software
- This is a demonstration project and has not been validated for clinical use

**Medical professionals should NOT rely on this tool for patient care.**

---

## 📚 Additional Resources

- [UCI Heart Disease Dataset Documentation](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Docker Documentation](https://docs.docker.com/)

