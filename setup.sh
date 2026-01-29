#!/bin/bash

echo "================================================"
echo "Heart Disease Prediction - Setup Script"
echo "================================================"

# Create directories
echo ""
echo "[1/5] Creating directories..."
mkdir -p data models figures
echo "✓ Directories created"

# Check if Python is installed
echo ""
echo "[2/5] Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✓ $PYTHON_VERSION installed"
else
    echo "✗ Python 3 not found. Please install Python 3.8 or higher"
    exit 1
fi

# Create virtual environment
echo ""
echo "[3/5] Creating virtual environment..."
python3 -m venv venv
echo "✓ Virtual environment created"

# Activate virtual environment
echo ""
echo "[4/5] Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Install dependencies
echo ""
echo "[5/5] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Dependencies installed"

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Download the heart disease dataset from:"
echo "   https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset"
echo "2. Place 'heart.csv' in the data/ directory"
echo "3. Train the model: python train_model.py"
echo "4. Run the app: streamlit run app.py"
echo ""
echo "OR use Docker:"
echo "  docker-compose up --build"
echo ""
echo "================================================"