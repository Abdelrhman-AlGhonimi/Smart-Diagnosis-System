# Smart Diagnosis System - Setup and Usage Guide

## Project Overview

The Smart Diagnosis System is a machine learning-based application that predicts potential diseases based on patient symptoms. The system features a user-friendly GUI built with PyQt5, allowing medical professionals to:

- Select symptoms from categorized lists
- Receive AI-powered disease predictions with confidence scores
- View precautions and treatment suggestions for diagnosed conditions
- Maintain a searchable patient diagnosis history
- Export diagnosis records to Excel

## Required Dependencies

This project requires the following Python libraries:

```
pandas
numpy
matplotlib
scikit-learn
imbalanced-learn (imblearn)
xgboost
PyQt5
joblib
sqlite3 (included with Python)
```

## Installation Guide

1. **Set up Python Environment** (Python 3.7+ recommended)

   Make sure you have Python installed on your system. If not, download and install it from [python.org](https://python.org).

2. **Install Required Libraries**

   Open a command prompt/terminal and run:

   ```bash
   pip install pandas numpy matplotlib scikit-learn imbalanced-learn xgboost PyQt5 joblib
   ```

   For Anaconda users:
   ```bash
   conda install pandas numpy matplotlib scikit-learn imbalanced-learn xgboost PyQt5 joblib
   ```

3. **Download Project Files**

   Ensure all project files are in the same directory:
   - `main.py` (Main application code)
   - `DiseaseAndSymptoms_cleaned.csv` (Training data)
   - `Disease_precaution_cleaned.csv` (Precautions data)

## Running the Application

1. **Navigate to Project Directory**

   ```bash
   cd path/to/project/directory
   ```

2. **Launch the Application**

   ```bash
   python main.py
   ```

   - On first run, the system will automatically train machine learning models, which may take a few minutes.
   - Subsequent runs will load pre-trained models for faster startup.

## Usage Workflow

### 1. Diagnosing a Patient

1. **Enter Patient Information**
   - Input patient ID and name in the fields provided

2. **Select Symptoms**
   - Browse symptoms by category or use the search function
   - Select symptoms from the list and click "Add →" to add them to the selected list
   - Remove any unwanted symptoms using the "← Remove" button

3. **Get Diagnosis**
   - Click the "🧠 Diagnose" button to generate predictions
   - The system will display:
     - Most likely diseases with confidence scores
     - Probability distribution chart of top predictions
     - Recommended precautions for the highest probability disease

4. **Save Diagnosis to History**
   - Click "💾 Save to History" to store the diagnosis in the database
   - Add optional notes if needed

### 2. Viewing and Managing History

1. **Access History Tab**
   - Click on the "History" tab to view past diagnoses

2. **Search and Filter Records**
   - Filter by patient name or ID
   - View records by date range

3. **View Detailed Records**
   - Select any history entry to view full diagnosis details

4. **Export Records**
   - Click "Export to Excel" to save history data as a spreadsheet

## Project Structure

- `main.py`: Contains all application code
- `disease_model.pkl`: Trained machine learning model (generated on first run)
- `label_encoder.pkl`: Label encoder for disease names (generated on first run)
- `symptom_columns.pkl`: List of all symptoms (generated on first run)
- `diagnosis_history.db`: SQLite database storing patient diagnosis history
- `DiseaseAndSymptoms_cleaned.csv`: Training data with disease-symptom mappings
- `Disease_precaution_cleaned.csv`: Precautions for each disease

## Technical Details

The system implements multiple machine learning models, including:
- Random Forest Classifier
- K-Nearest Neighbors Classifier
- XGBoost Classifier
- Neural Network (MLP Classifier)

During initial setup, the system:
1. Trains all models on the symptom dataset
2. Uses SMOTE for handling class imbalance
3. Performs hyperparameter tuning via GridSearchCV
4. Selects the best performing model for production use

## Troubleshooting

- **Missing Files Error**: Ensure all required CSV files are in the same directory as main.py
- **Library Import Errors**: Verify all dependencies are installed correctly
- **Database Errors**: Check if you have write permissions in the project directory
