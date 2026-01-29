"""
Heart Disease Prediction Model Training Script
Author: Panos
Date: January 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from xgboost import XGBClassifier
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('figures', exist_ok=True)

print("="*60)
print("HEART DISEASE PREDICTION - MODEL TRAINING")
print("="*60)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/7] Loading dataset...")

try:
    df = pd.read_csv('data/heart.csv')
    print(f"✓ Dataset loaded successfully: {df.shape[0]} samples, {df.shape[1]} features")
    
    # ⚠️ CRITICAL: Check and remove duplicates
    print("\n🔍 CHECKING FOR DUPLICATES...")
    original_size = len(df)
    duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates}")
    
    if duplicates > 0:
        print(f"⚠️ WARNING: {duplicates} duplicates found ({duplicates/original_size*100:.1f}% of dataset)")
        print("Removing duplicates...")
        df = df.drop_duplicates()
        print(f"✓ Duplicates removed!")
        print(f"Dataset size: {original_size} → {len(df)} samples")
        print(f"Lost {original_size - len(df)} duplicate rows")
    else:
        print("✓ No duplicates found")
    
except FileNotFoundError:
    print("ERROR: data/heart.csv not found!")
    print("Please download the dataset from:")
    print("https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset")
    print("And place it in the data/ folder")
    exit(1)

# Display basic info
print(f"\nFinal dataset shape: {df.shape}")
print(f"Features: {list(df.columns)}")
print(f"\nTarget distribution:")
print(df['target'].value_counts())
print(f"Class balance: {df['target'].value_counts(normalize=True)}")

# Additional data quality checks
print("\n🔍 DATA QUALITY CHECKS...")

# Check for missing values
missing_values = df.isnull().sum()
if missing_values.sum() > 0:
    print(f"⚠️ Missing values found:")
    print(missing_values[missing_values > 0])
else:
    print("✓ No missing values")

# Check for suspicious correlations (potential leakage)
print("\n🔍 Checking for potential data leakage...")
target_corr = df.corr()['target'].abs().sort_values(ascending=False)
high_corr = target_corr[(target_corr > 0.8) & (target_corr < 1.0)]
if len(high_corr) > 0:
    print(f"⚠️ ALERT: Very high correlations detected (>0.8):")
    print(high_corr)
else:
    print("✓ No suspiciously high correlations detected")

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n[2/7] Performing exploratory data analysis...")

# Create visualizations
print("\nGenerating visualizations...")

# 1. Target distribution and EDA
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Target distribution
axes[0, 0].pie(df['target'].value_counts(), labels=['No Disease', 'Disease'], 
               autopct='%1.1f%%', startangle=90, colors=['#2ecc71', '#e74c3c'])
axes[0, 0].set_title('Target Distribution', fontsize=14, fontweight='bold')

# Age distribution by target
sns.histplot(data=df, x='age', hue='target', kde=True, ax=axes[0, 1], bins=20)
axes[0, 1].set_title('Age Distribution by Heart Disease', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Age')
axes[0, 1].set_ylabel('Count')

# Correlation heatmap
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, ax=axes[1, 0], cbar_kws={'shrink': 0.8})
axes[1, 0].set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')

# Chest pain type vs target
cp_counts = df.groupby(['cp', 'target']).size().unstack()
cp_counts.plot(kind='bar', ax=axes[1, 1], color=['#2ecc71', '#e74c3c'])
axes[1, 1].set_title('Chest Pain Type vs Heart Disease', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Chest Pain Type')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=0)
axes[1, 1].legend(['No Disease', 'Disease'])

plt.tight_layout()
plt.savefig('figures/eda_overview.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figures/eda_overview.png")
plt.close()

# Additional correlation with target
target_corr = df.corr()['target'].sort_values(ascending=False)
plt.figure(figsize=(10, 8))
target_corr[1:].plot(kind='barh', color='steelblue')
plt.title('Feature Correlation with Heart Disease', fontsize=14, fontweight='bold')
plt.xlabel('Correlation Coefficient')
plt.tight_layout()
plt.savefig('figures/target_correlation.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figures/target_correlation.png")
plt.close()

# ============================================================================
# 3. DATA PREPROCESSING
# ============================================================================
print("\n[3/7] Preprocessing data...")

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Verify no overlap between train and test
assert len(set(X_train.index) & set(X_test.index)) == 0, "❌ CRITICAL: Train-test overlap detected!"
print("✓ No overlap between train and test sets")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Features scaled using StandardScaler")

# ============================================================================
# 4. MODEL TRAINING
# ============================================================================
print("\n[4/7] Training models...")

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, 
        random_state=RANDOM_STATE,
        max_depth=10,
        min_samples_split=5
    ),
    'XGBoost': XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric='logloss',
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1
    )
}

results = {}

for name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Training {name}...")
    print(f"{'='*60}")
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    auc = roc_auc_score(y_test, y_test_proba)
    
    # Sanity checks
    print(f"\n🔍 Sanity Checks:")
    print(f"   Train size: {len(y_train)}, Test size: {len(y_test)}")
    print(f"   Train predictions unique values: {np.unique(y_train_pred)}")
    print(f"   Test predictions unique values: {np.unique(y_test_pred)}")
    print(f"   Test predictions distribution: {np.bincount(y_test_pred)}")
    
    results[name] = {
        'model': model,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'auc_roc': auc,
        'cv_auc_mean': cv_scores.mean(),
        'cv_auc_std': cv_scores.std(),
        'y_test_pred': y_test_pred,
        'y_test_proba': y_test_proba
    }
    
    print(f"\n📊 Performance Metrics:")
    print(f"   Train Accuracy: {train_acc:.4f}")
    print(f"   Test Accuracy:  {test_acc:.4f}")
    print(f"   AUC-ROC:        {auc:.4f}")
    print(f"   CV AUC:         {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # Overfitting warnings
    acc_gap = train_acc - test_acc
    if train_acc > 0.99 and test_acc > 0.99:
        print(f"   ⚠️⚠️⚠️ ALERT: Suspiciously high accuracy! Check for data leakage!")
    elif acc_gap > 0.15:
        print(f"   ⚠️ WARNING: Large train-test gap ({acc_gap:.4f}) - model is overfitting!")
    elif acc_gap < 0.02:
        print(f"   ✓ Good generalization (gap: {acc_gap:.4f})")
    else:
        print(f"   ✓ Acceptable generalization (gap: {acc_gap:.4f})")

# ============================================================================
# 5. MODEL EVALUATION
# ============================================================================
print("\n[5/7] Evaluating models...")

# Find best model
best_model_name = max(results, key=lambda x: results[x]['auc_roc'])
best_model = results[best_model_name]['model']

print(f"\n{'='*60}")
print(f"BEST MODEL: {best_model_name}")
print(f"{'='*60}")

# Detailed metrics for best model
print(f"\nDetailed Classification Report:")
print(classification_report(
    y_test, 
    results[best_model_name]['y_test_pred'],
    target_names=['No Disease', 'Disease']
))

# Confusion Matrix
cm = confusion_matrix(y_test, results[best_model_name]['y_test_pred'])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Add accuracy text
accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
plt.text(0.5, -0.1, f'Accuracy: {accuracy:.2%}', 
         ha='center', transform=plt.gca().transAxes, fontsize=12)

plt.tight_layout()
plt.savefig('figures/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: figures/confusion_matrix.png")
plt.close()

# ROC Curves for all models
plt.figure(figsize=(10, 8))
for name, result in results.items():
    fpr, tpr, _ = roc_curve(y_test, result['y_test_proba'])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {result['auc_roc']:.3f})", linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/roc_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figures/roc_curves.png")
plt.close()

# Model Comparison
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Train Accuracy': [results[m]['train_accuracy'] for m in results],
    'Test Accuracy': [results[m]['test_accuracy'] for m in results],
    'AUC-ROC': [results[m]['auc_roc'] for m in results],
    'CV AUC (mean)': [results[m]['cv_auc_mean'] for m in results],
    'Overfitting Gap': [results[m]['train_accuracy'] - results[m]['test_accuracy'] for m in results]
})

print(f"\n{'='*60}")
print("MODEL COMPARISON")
print(f"{'='*60}")
print(comparison_df.to_string(index=False))

# Visualize comparison
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Accuracy comparison
x = np.arange(len(results))
width = 0.35
axes[0].bar(x - width/2, comparison_df['Train Accuracy'], width, 
            label='Train Accuracy', color='skyblue')
axes[0].bar(x + width/2, comparison_df['Test Accuracy'], width, 
            label='Test Accuracy', color='orange')
axes[0].set_xlabel('Model')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Model Accuracy Comparison', fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(comparison_df['Model'], rotation=15, ha='right')
axes[0].legend()
axes[0].set_ylim([0.7, 1.0])
axes[0].grid(axis='y', alpha=0.3)

# Add value labels
for i, (train, test) in enumerate(zip(comparison_df['Train Accuracy'], comparison_df['Test Accuracy'])):
    axes[0].text(i - width/2, train + 0.01, f'{train:.3f}', ha='center', fontsize=8)
    axes[0].text(i + width/2, test + 0.01, f'{test:.3f}', ha='center', fontsize=8)

# AUC comparison
axes[1].bar(comparison_df['Model'], comparison_df['AUC-ROC'], color='green', alpha=0.7)
axes[1].set_xlabel('Model')
axes[1].set_ylabel('AUC-ROC Score')
axes[1].set_title('Model AUC-ROC Comparison', fontweight='bold')
axes[1].set_xticklabels(comparison_df['Model'], rotation=15, ha='right')
axes[1].set_ylim([0.7, 1.0])
axes[1].grid(axis='y', alpha=0.3)

# Add value labels
for i, auc in enumerate(comparison_df['AUC-ROC']):
    axes[1].text(i, auc + 0.01, f'{auc:.3f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('figures/model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figures/model_comparison.png")
plt.close()

# ============================================================================
# 6. FEATURE IMPORTANCE
# ============================================================================
print("\n[6/7] Analyzing feature importance...")

if hasattr(best_model, 'feature_importances_'):
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))
    
    # Visualize feature importance
    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
    plt.title(f'Feature Importance - {best_model_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('figures/feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: figures/feature_importance.png")
    plt.close()
elif hasattr(best_model, 'coef_'):
    # For logistic regression
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'coefficient': best_model.coef_[0]
    }).sort_values('coefficient', key=abs, ascending=False)
    
    print("\nFeature Coefficients (Logistic Regression):")
    print(importance_df.to_string(index=False))
    
    plt.figure(figsize=(10, 8))
    colors = ['red' if x < 0 else 'green' for x in importance_df['coefficient']]
    plt.barh(importance_df['feature'], importance_df['coefficient'], color=colors, alpha=0.7)
    plt.title(f'Feature Coefficients - {best_model_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    plt.savefig('figures/feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: figures/feature_importance.png")
    plt.close()

# ============================================================================
# 7. SAVE MODELS
# ============================================================================
print("\n[7/7] Saving models and artifacts...")

# Save best model
joblib.dump(best_model, 'models/best_model.pkl')
print(f"✓ Saved: models/best_model.pkl ({best_model_name})")

# Save scaler
joblib.dump(scaler, 'models/scaler.pkl')
print("✓ Saved: models/scaler.pkl")

# Save all models
for name, result in results.items():
    filename = f"models/{name.lower().replace(' ', '_')}.pkl"
    joblib.dump(result['model'], filename)
    print(f"✓ Saved: {filename}")

# Save model metadata (convert numpy types to Python native types)
metadata = {
    'best_model': best_model_name,
    'best_model_metrics': {
        'test_accuracy': float(results[best_model_name]['test_accuracy']),
        'auc_roc': float(results[best_model_name]['auc_roc']),
        'cv_auc_mean': float(results[best_model_name]['cv_auc_mean'])
    },
    'feature_names': list(X.columns),
    'training_samples': int(X_train.shape[0]),
    'test_samples': int(X_test.shape[0]),
    'random_state': int(RANDOM_STATE),
    'duplicates_removed': int(duplicates),
    'final_dataset_size': int(len(df))
}

import json
with open('models/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)
print("✓ Saved: models/metadata.json")

# Save comparison results
comparison_df.to_csv('models/model_comparison.csv', index=False)
print("✓ Saved: models/model_comparison.csv")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"\n✓ Best Model: {best_model_name}")
print(f"✓ Test Accuracy: {results[best_model_name]['test_accuracy']:.4f}")
print(f"✓ AUC-ROC: {results[best_model_name]['auc_roc']:.4f}")
print(f"✓ Dataset: {len(df)} unique samples ({duplicates} duplicates removed)")
print(f"\n✓ All models and visualizations saved successfully!")
print(f"✓ Run 'streamlit run app.py' to launch the web application")