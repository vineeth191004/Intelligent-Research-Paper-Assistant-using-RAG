#!/bin/bash
# Startup script for Intelligent Research Paper Assistant

# Ensure the script stops if any command fails
set -e

echo "Activating virtual environment..."
source venv/bin/activate

echo "Configuring Python Path..."
export PYTHONPATH=$(pwd)

echo "Launching Streamlit application..."
streamlit run frontend/app.py
