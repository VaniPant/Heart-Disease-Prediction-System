"""
Heart Disease Prediction Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #e74c3c;
        text-align: center;
        padding: 20px;
    }
    .sub-header {
        font-size: 24px;
        color: #34495e;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3498db;
    }
    .risk-high {
        background-color: #ffe6e6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #e74c3c;
    }
    .risk-low {
        background-color: #e6ffe6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2ecc71;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL AND METADATA
# ============================================================================

@st.cache_resource
def load_model_and_scaler():
    """Load the trained model and scaler"""
    try:
        model = joblib.load('models/best_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        
        # Load metadata
        with open('models/metadata.json', 'r') as f:
            metadata = json.load(f)
        
        return model, scaler, metadata
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please run 'python train_model.py' first.")
        st.stop()

model, scaler, metadata = load_model_and_scaler()

# ============================================================================
# HEADER
# ============================================================================

st.markdown('<div class="main-header">🫀 Heart Disease Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Cardiovascular Risk Assessment</div>', unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/heart-health.png", width=100)
    st.title("ℹ️ About")
    st.info(
        f"""
        **Model:** {metadata['best_model']}
        
        **Performance:**
        - Accuracy: {metadata['best_model_metrics']['test_accuracy']:.2%}
        - AUC-ROC: {metadata['best_model_metrics']['auc_roc']:.3f}
        
        **Training Data:**
        - {metadata['training_samples']} training samples
        - {metadata['test_samples']} test samples
        
        **Features:** {len(metadata['feature_names'])} clinical measurements
        """
    )
    
    st.markdown("---")
    
    st.warning(
        """
        ⚠️ **Medical Disclaimer**
        
        This tool is for educational and research purposes only. 
        It should NOT be used as a substitute for professional medical advice, 
        diagnosis, or treatment.
        
        Always consult with a qualified healthcare provider for medical decisions.
        """
    )
    
    st.markdown("---")
    st.markdown("**Developer:** Vani")
    st.markdown("**Date:** January 2026")
    st.markdown(f"**Last Update:** {datetime.now().strftime('%Y-%m-%d')}")

# ============================================================================
# MAIN INTERFACE
# ============================================================================

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Prediction", "📊 Model Info", "📈 Visualizations", "ℹ️ Feature Guide"])

# ============================================================================
# TAB 1: PREDICTION
# ============================================================================

with tab1:
    st.header("Patient Information Input")
    st.write("Enter the patient's clinical measurements below:")
    
    # Create two columns for input
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Demographics")
        age = st.number_input('Age (years)', min_value=1, max_value=120, value=50, step=1,
                             help="Patient's age in years")
        sex = st.selectbox('Sex', options=[1, 0], 
                          format_func=lambda x: 'Male' if x == 1 else 'Female',
                          help="Biological sex")
        
        st.subheader("Cardiovascular")
        trestbps = st.number_input('Resting Blood Pressure (mm Hg)', 
                                   min_value=80, max_value=200, value=120, step=1,
                                   help="Resting blood pressure on admission to hospital")
        chol = st.number_input('Cholesterol (mg/dl)', 
                              min_value=100, max_value=600, value=200, step=1,
                              help="Serum cholesterol in mg/dl")
        thalach = st.number_input('Max Heart Rate Achieved', 
                                 min_value=60, max_value=220, value=150, step=1,
                                 help="Maximum heart rate achieved during exercise")
    
    with col2:
        st.subheader("Symptoms & Tests")
        cp = st.selectbox('Chest Pain Type', options=[0, 1, 2, 3],
                         format_func=lambda x: {
                             0: '0 - Typical Angina',
                             1: '1 - Atypical Angina',
                             2: '2 - Non-anginal Pain',
                             3: '3 - Asymptomatic'
                         }[x],
                         help="Type of chest pain experienced")
        
        exang = st.selectbox('Exercise Induced Angina', options=[0, 1],
                            format_func=lambda x: 'Yes' if x == 1 else 'No',
                            help="Angina induced by exercise")
        
        oldpeak = st.number_input('ST Depression', 
                                 min_value=0.0, max_value=10.0, value=1.0, step=0.1,
                                 help="ST depression induced by exercise relative to rest")
        
        slope = st.selectbox('Slope of Peak Exercise ST Segment', options=[0, 1, 2],
                            format_func=lambda x: {
                                0: '0 - Upsloping',
                                1: '1 - Flat',
                                2: '2 - Downsloping'
                            }[x],
                            help="Slope of the peak exercise ST segment")
    
    with col3:
        st.subheader("Additional Tests")
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1],
                          format_func=lambda x: 'Yes' if x == 1 else 'No',
                          help="Fasting blood sugar greater than 120 mg/dl")
        
        restecg = st.selectbox('Resting ECG Results', options=[0, 1, 2],
                              format_func=lambda x: {
                                  0: '0 - Normal',
                                  1: '1 - ST-T Wave Abnormality',
                                  2: '2 - Left Ventricular Hypertrophy'
                              }[x],
                              help="Resting electrocardiographic results")
        
        ca = st.selectbox('Number of Major Vessels (0-3)', options=[0, 1, 2, 3],
                         help="Number of major vessels colored by fluoroscopy")
        
        thal = st.selectbox('Thalassemia', options=[0, 1, 2, 3],
                           format_func=lambda x: {
                               0: '0 - Unknown',
                               1: '1 - Normal',
                               2: '2 - Fixed Defect',
                               3: '3 - Reversible Defect'
                           }[x],
                           help="Thalassemia blood disorder")
    
    st.markdown("---")
    
    # Prediction button
    col_button1, col_button2, col_button3 = st.columns([1, 1, 1])
    
    with col_button2:
        predict_button = st.button('🔮 Predict Heart Disease Risk', use_container_width=True, type="primary")
    
    if predict_button:
        # Create input array
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, 
                               thalach, exang, oldpeak, slope, ca, thal]])
        
        # Scale input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # Display results
        st.markdown("---")
        st.header("🎯 Prediction Results")
        
        # Create columns for results
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            if prediction == 1:
                st.markdown(
                    f"""
                    <div class="risk-high">
                    <h2 style="color: #e74c3c;">⚠️ High Risk Detected</h2>
                    <h3>Probability of Heart Disease: {probability[1]:.1%}</h3>
                    <p style="font-size: 16px;">
                    The model indicates an elevated risk of heart disease based on the provided clinical data.
                    </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="risk-low">
                    <h2 style="color: #2ecc71;">✅ Low Risk</h2>
                    <h3>Probability of Heart Disease: {probability[1]:.1%}</h3>
                    <p style="font-size: 16px;">
                    The model indicates a lower risk of heart disease based on the provided clinical data.
                    </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        with result_col2:
            # Probability gauge
            st.metric(
                label="Disease Probability",
                value=f"{probability[1]:.1%}",
                delta=f"{probability[1] - 0.5:.1%} from baseline"
            )
            
            st.metric(
                label="No Disease Probability",
                value=f"{probability[0]:.1%}"
            )
        
        st.markdown("---")
        
        # Risk factor analysis
        st.subheader("📋 Risk Factor Analysis")
        
        risk_factors = []
        protective_factors = []
        
        # Analyze risk factors
        if age > 60:
            risk_factors.append(f"**Advanced Age:** {age} years (>60 increases risk)")
        elif age < 45:
            protective_factors.append(f"**Younger Age:** {age} years")
        
        if sex == 1:
            risk_factors.append("**Sex:** Male (higher baseline risk)")
        
        if chol > 240:
            risk_factors.append(f"**High Cholesterol:** {chol} mg/dl (>240 is high)")
        elif chol < 200:
            protective_factors.append(f"**Healthy Cholesterol:** {chol} mg/dl (<200)")
        
        if trestbps > 140:
            risk_factors.append(f"**High Blood Pressure:** {trestbps} mm Hg (>140 is high)")
        elif trestbps < 120:
            protective_factors.append(f"**Normal Blood Pressure:** {trestbps} mm Hg (<120)")
        
        if fbs == 1:
            risk_factors.append("**Elevated Fasting Blood Sugar:** >120 mg/dl")
        
        if exang == 1:
            risk_factors.append("**Exercise-Induced Angina:** Present")
        
        if thalach < 100:
            risk_factors.append(f"**Low Max Heart Rate:** {thalach} bpm")
        
        if cp in [1, 2, 3]:
            if cp == 1:
                risk_factors.append("**Chest Pain:** Atypical angina")
            elif cp == 2:
                risk_factors.append("**Chest Pain:** Non-anginal pain")
        
        if oldpeak > 2:
            risk_factors.append(f"**Significant ST Depression:** {oldpeak}")
        
        # Display risk factors
        col_risk, col_protect = st.columns(2)
        
        with col_risk:
            if risk_factors:
                st.markdown("**⚠️ Identified Risk Factors:**")
                for rf in risk_factors:
                    st.markdown(f"- {rf}")
            else:
                st.success("No major risk factors identified!")
        
        with col_protect:
            if protective_factors:
                st.markdown("**✅ Protective Factors:**")
                for pf in protective_factors:
                    st.markdown(f"- {pf}")
        
        st.markdown("---")
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        if prediction == 1:
            st.error(
                """
                **Immediate Actions:**
                1. 🏥 Schedule an appointment with a cardiologist for comprehensive evaluation
                2. 📋 Bring all medical records and current medications to the appointment
                3. 🩺 Consider additional diagnostic tests (ECG, stress test, cardiac catheterization)
                4. 💊 Do NOT stop or change medications without consulting your doctor
                
                **Lifestyle Modifications:**
                - Follow a heart-healthy diet (Mediterranean or DASH diet)
                - Engage in regular moderate exercise (consult doctor first)
                - Quit smoking and limit alcohol consumption
                - Manage stress through relaxation techniques
                - Monitor blood pressure and cholesterol regularly
                """
            )
        else:
            st.success(
                """
                **Maintenance Actions:**
                1. ✅ Continue regular check-ups with your primary care physician
                2. 🏃 Maintain regular physical activity (150 min/week moderate exercise)
                3. 🥗 Follow a balanced, heart-healthy diet
                4. 📊 Monitor cardiovascular risk factors periodically
                
                **Prevention Strategies:**
                - Keep blood pressure under control (<120/80 mm Hg)
                - Maintain healthy cholesterol levels (LDL <100 mg/dl)
                - Achieve and maintain healthy body weight
                - Avoid smoking and excessive alcohol
                - Manage diabetes if present
                """
            )
        
        # Save prediction log
        st.markdown("---")
        
        if st.button("💾 Save Prediction Report"):
            # Create prediction report
            report = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'patient_data': {
                    'age': int(age),
                    'sex': 'Male' if sex == 1 else 'Female',
                    'chest_pain_type': int(cp),
                    'resting_bp': int(trestbps),
                    'cholesterol': int(chol),
                    'fasting_blood_sugar': 'High' if fbs == 1 else 'Normal',
                    'resting_ecg': int(restecg),
                    'max_heart_rate': int(thalach),
                    'exercise_angina': 'Yes' if exang == 1 else 'No',
                    'st_depression': float(oldpeak),
                    'slope': int(slope),
                    'major_vessels': int(ca),
                    'thalassemia': int(thal)
                },
                'prediction': {
                    'result': 'High Risk' if prediction == 1 else 'Low Risk',
                    'disease_probability': float(probability[1]),
                    'no_disease_probability': float(probability[0])
                },
                'risk_factors': risk_factors,
                'protective_factors': protective_factors
            }
            
            # Save to JSON
            report_filename = f"prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_filename, 'w') as f:
                json.dump(report, f, indent=4)
            
            st.success(f"✅ Prediction report saved as: {report_filename}")

# ============================================================================
# TAB 2: MODEL INFO
# ============================================================================

with tab2:
    st.header("📊 Model Information & Performance")
    
    # Model details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
            <h3>Model Details</h3>
            <ul>
                <li><strong>Algorithm:</strong> {metadata['best_model']}</li>
                <li><strong>Features:</strong> {len(metadata['feature_names'])}</li>
                <li><strong>Training Samples:</strong> {metadata['training_samples']}</li>
                <li><strong>Test Samples:</strong> {metadata['test_samples']}</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
            <h3>Performance Metrics</h3>
            <ul>
                <li><strong>Test Accuracy:</strong> {metadata['best_model_metrics']['test_accuracy']:.2%}</li>
                <li><strong>AUC-ROC:</strong> {metadata['best_model_metrics']['auc_roc']:.3f}</li>
                <li><strong>CV AUC (mean):</strong> {metadata['best_model_metrics']['cv_auc_mean']:.3f}</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    
    # Feature list
    st.subheader("📝 Model Features")
    
    feature_descriptions = {
        'age': 'Age in years',
        'sex': 'Sex (1 = male, 0 = female)',
        'cp': 'Chest pain type (0-3)',
        'trestbps': 'Resting blood pressure (mm Hg)',
        'chol': 'Serum cholesterol (mg/dl)',
        'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)',
        'restecg': 'Resting electrocardiographic results (0-2)',
        'thalach': 'Maximum heart rate achieved',
        'exang': 'Exercise induced angina (1 = yes, 0 = no)',
        'oldpeak': 'ST depression induced by exercise relative to rest',
        'slope': 'Slope of the peak exercise ST segment (0-2)',
        'ca': 'Number of major vessels colored by fluoroscopy (0-3)',
        'thal': 'Thalassemia (0-3)'
    }
    
    feature_df = pd.DataFrame({
        'Feature': metadata['feature_names'],
        'Description': [feature_descriptions.get(f, 'N/A') for f in metadata['feature_names']]
    })
    
    st.dataframe(feature_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Model comparison
    st.subheader("🏆 Model Comparison")
    
    if os.path.exists('models/model_comparison.csv'):
        comparison_df = pd.read_csv('models/model_comparison.csv')
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    else:
        st.info("Model comparison data not available. Run training script to generate.")

# ============================================================================
# TAB 3: VISUALIZATIONS
# ============================================================================

with tab3:
    st.header("📈 Model Visualizations")
    
    # Check if visualization files exist
    viz_files = {
        'EDA Overview': 'figures/eda_overview.png',
        'Target Correlation': 'figures/target_correlation.png',
        'Confusion Matrix': 'figures/confusion_matrix.png',
        'ROC Curves': 'figures/roc_curves.png',
        'Feature Importance': 'figures/feature_importance.png',
        'Model Comparison': 'figures/model_comparison.png'
    }
    
    available_viz = {k: v for k, v in viz_files.items() if os.path.exists(v)}
    
    if available_viz:
        viz_selection = st.selectbox(
            'Select Visualization',
            options=list(available_viz.keys())
        )
        
        st.image(available_viz[viz_selection], use_container_width=True)
        
        # Download button
        with open(available_viz[viz_selection], 'rb') as file:
            st.download_button(
                label="📥 Download Image",
                data=file,
                file_name=os.path.basename(available_viz[viz_selection]),
                mime="image/png"
            )
    else:
        st.warning("⚠️ Visualizations not found. Please run 'python train_model.py' to generate visualizations.")

# ============================================================================
# TAB 4: FEATURE GUIDE
# ============================================================================

with tab4:
    st.header("ℹ️ Clinical Feature Guide")
    
    st.markdown("""
    This guide provides detailed information about each clinical feature used in the model.
    """)
    
    with st.expander("🧍 Age"):
        st.markdown("""
        **Description:** Patient's age in years
        
        **Range:** 29-77 years (typical in dataset)
        
        **Clinical Significance:**
        - Heart disease risk increases with age
        - Men ≥45 and women ≥55 are at higher risk
        - Age is one of the strongest predictors
        """)
    
    with st.expander("⚧️ Sex"):
        st.markdown("""
        **Description:** Biological sex of the patient
        
        **Values:**
        - 1 = Male
        - 0 = Female
        
        **Clinical Significance:**
        - Men generally have higher risk at younger ages
        - Women's risk increases significantly after menopause
        - Hormonal differences affect cardiovascular health
        """)
    
    with st.expander("💔 Chest Pain Type (cp)"):
        st.markdown("""
        **Description:** Type of chest pain experienced
        
        **Values:**
        - 0 = Typical Angina: Classic heart-related chest pain
        - 1 = Atypical Angina: Chest pain with some unusual features
        - 2 = Non-Anginal Pain: Chest pain unlikely to be heart-related
        - 3 = Asymptomatic: No chest pain
        
        **Clinical Significance:**
        - Typical angina is highly suggestive of coronary artery disease
        - Presence and type of chest pain is crucial for diagnosis
        """)
    
    with st.expander("🩸 Resting Blood Pressure (trestbps)"):
        st.markdown("""
        **Description:** Blood pressure measured at rest (mm Hg)
        
        **Normal Ranges:**
        - Normal: <120 mm Hg
        - Elevated: 120-129 mm Hg
        - High BP Stage 1: 130-139 mm Hg
        - High BP Stage 2: ≥140 mm Hg
        
        **Clinical Significance:**
        - Hypertension is a major cardiovascular risk factor
        - Damages blood vessels over time
        - Often called the "silent killer"
        """)
    
    with st.expander("🧪 Cholesterol (chol)"):
        st.markdown("""
        **Description:** Serum cholesterol in mg/dl
        
        **Ranges:**
        - Desirable: <200 mg/dl
        - Borderline High: 200-239 mg/dl
        - High: ≥240 mg/dl
        
        **Clinical Significance:**
        - High cholesterol leads to plaque buildup in arteries
        - LDL ("bad" cholesterol) is particularly harmful
        - Major modifiable risk factor
        """)
    
    with st.expander("🍬 Fasting Blood Sugar (fbs)"):
        st.markdown("""
        **Description:** Fasting blood sugar level
        
        **Values:**
        - 1 = Fasting blood sugar >120 mg/dl (high)
        - 0 = Fasting blood sugar ≤120 mg/dl (normal)
        
        **Clinical Significance:**
        - High blood sugar indicates diabetes or prediabetes
        - Diabetes significantly increases heart disease risk
        - Damages blood vessels and nerves
        """)
    
    with st.expander("📉 Resting ECG (restecg)"):
        st.markdown("""
        **Description:** Resting electrocardiographic results
        
        **Values:**
        - 0 = Normal
        - 1 = ST-T wave abnormality
        - 2 = Left ventricular hypertrophy
        
        **Clinical Significance:**
        - Abnormal ECG may indicate heart problems
        - Left ventricular hypertrophy suggests chronic high BP
        - Important diagnostic tool
        """)
    
    with st.expander("💓 Maximum Heart Rate (thalach)"):
        st.markdown("""
        **Description:** Maximum heart rate achieved during exercise
        
        **Expected Max HR:** Approximately 220 - age
        
        **Clinical Significance:**
        - Lower than expected max HR may indicate heart problems
        - Important measure of cardiovascular fitness
        - Used to assess exercise capacity
        """)
    
    with st.expander("🏃 Exercise Induced Angina (exang)"):
        st.markdown("""
        **Description:** Chest pain triggered by physical exertion
        
        **Values:**
        - 1 = Yes (angina during exercise)
        - 0 = No (no angina during exercise)
        
        **Clinical Significance:**
        - Strong indicator of coronary artery disease
        - Suggests inadequate blood flow to heart during stress
        - Often leads to further cardiac testing
        """)
    
    with st.expander("📊 ST Depression (oldpeak)"):
        st.markdown("""
        **Description:** ST segment depression on ECG during exercise
        
        **Measured in:** Millimeters
        
        **Clinical Significance:**
        - Indicates reduced blood flow to heart muscle
        - Higher values suggest more severe disease
        - Key diagnostic criterion for CAD
        """)
    
    with st.expander("📈 ST Slope (slope)"):
        st.markdown("""
        **Description:** Slope of peak exercise ST segment
        
        **Values:**
        - 0 = Upsloping (most normal)
        - 1 = Flat
        - 2 = Downsloping (most abnormal)
        
        **Clinical Significance:**
        - Downsloping ST segment is highly suggestive of CAD
        - Used in conjunction with ST depression
        - Important for exercise stress test interpretation
        """)
    
    with st.expander("🔬 Major Vessels (ca)"):
        st.markdown("""
        **Description:** Number of major vessels colored by fluoroscopy
        
        **Range:** 0-3 vessels
        
        **Clinical Significance:**
        - More colored vessels indicates more blockages
        - Directly visualizes coronary artery disease
        - Helps determine severity and treatment approach
        """)
    
    with st.expander("🩺 Thalassemia (thal)"):
        st.markdown("""
        **Description:** Blood disorder affecting hemoglobin
        
        **Values:**
        - 0 = Unknown
        - 1 = Normal
        - 2 = Fixed defect
        - 3 = Reversible defect
        
        **Clinical Significance:**
        - Can affect oxygen-carrying capacity of blood
        - Reversible defects may indicate ischemia
        - Used in nuclear imaging studies
        """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #7f8c8d; padding: 20px;">
        <p><strong>Heart Disease Prediction System v1.0</strong></p>
        <p>Developed by Panos | Data Science Master's Student | University of Luxembourg</p>
        <p>⚠️ For educational and research purposes only - Not for clinical use</p>
    </div>
    """,
    unsafe_allow_html=True
)