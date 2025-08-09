import pandas as pd, numpy as np, random, joblib, sqlite3, os, datetime, time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Machine Learning imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

# PyQt imports
import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QListWidget, 
                             QVBoxLayout, QTextEdit, QListWidgetItem, QAbstractItemView, 
                             QHBoxLayout, QLineEdit, QComboBox, QTabWidget, QGroupBox,
                             QRadioButton, QCheckBox, QDialog, QFileDialog, QMessageBox,
                             QTableWidget, QTableWidgetItem, QHeaderView, QSplitter)
from PyQt5.QtCore import Qt, QSize, QTimer, QDate
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor

def setup_database():
    """Create SQLite database for diagnosis history if it doesn't exist"""
    conn = sqlite3.connect('diagnosis_history.db')
    cursor = conn.cursor()
    
    # Check if the diagnoses table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='diagnoses'")
    table_exists = cursor.fetchone() is not None
    
    if not table_exists:
        # Create new table with patient name and ID fields
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS diagnoses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            patient_name TEXT,
            timestamp TEXT,
            symptoms TEXT,
            predicted_disease TEXT,
            confidence REAL,
            notes TEXT
        )
        ''')
    else:
        # Check if we need to alter the existing table to add new columns
        cursor.execute("PRAGMA table_info(diagnoses)")
        columns = [info[1] for info in cursor.fetchall()]
        
        if 'patient_id' not in columns:
            cursor.execute("ALTER TABLE diagnoses ADD COLUMN patient_id TEXT")
        
        if 'patient_name' not in columns:
            cursor.execute("ALTER TABLE diagnoses ADD COLUMN patient_name TEXT")
    
    conn.commit()
    conn.close()

def train_and_save_models():
    print("Loading and preprocessing data...")
    sym_df = pd.read_csv("DiseaseAndSymptoms_cleaned.csv")
    all_sym = pd.unique(sym_df.drop('Disease', axis=1).values.ravel())
    all_sym = [s for s in all_sym if pd.notna(s)]
    
    # Generate more training examples with augmentation
    rows = []
    for _, r in sym_df.iterrows():
        disease = r['Disease']
        syms = r.dropna()[1:].values
        
        # Original symptom set
        base = {s: 1 if s in syms else 0 for s in all_sym}
        base['Disease'] = disease
        rows.append(base)
        
        # Data augmentation - create variations with subsets of symptoms
        for _ in range(30):
            keep = set(syms) - set(random.sample(list(syms), k=min(2, len(syms))))
            art = {s: 1 if s in keep else 0 for s in all_sym}
            art['Disease'] = disease
            rows.append(art)
    
    # Create DataFrame and prepare for training
    df = pd.DataFrame(rows)
    label_enc = LabelEncoder()
    df['y'] = label_enc.fit_transform(df['Disease'])
    X, y = df[all_sym], df['y']
    
    # Split data and apply SMOTE for class balancing
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    print(f"Training data shape before SMOTE: {X_tr.shape}")
    
    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_tr_balanced, y_tr_balanced = smote.fit_resample(X_tr, y_tr)
    print(f"Training data shape after SMOTE: {X_tr_balanced.shape}")
    
    # Train different models with hyperparameter tuning
    models = {}
    best_score = 0
    best_model_name = ""
    
    print("Training Random Forest model with GridSearchCV...")
    # Random Forest with GridSearchCV
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='accuracy', n_jobs=-1)
    rf_grid.fit(X_tr_balanced, y_tr_balanced)
    models['RandomForest'] = rf_grid.best_estimator_
    rf_acc = accuracy_score(y_te, models['RandomForest'].predict(X_te))
    print(f"RandomForest best params: {rf_grid.best_params_}")
    print(f"RandomForest accuracy: {rf_acc:.4f}")
    
    if rf_acc > best_score:
        best_score = rf_acc
        best_model_name = 'RandomForest'
    
    print("Training K-NN model...")
    # K-NN
    knn_params = {
        'n_neighbors': [1, 3, 5, 7],
        'weights': ['uniform', 'distance'],
        'metric': ['jaccard', 'hamming']
    }
    knn = KNeighborsClassifier()
    knn_grid = GridSearchCV(knn, knn_params, cv=3, scoring='accuracy', n_jobs=-1)
    knn_grid.fit(X_tr_balanced, y_tr_balanced)
    models['KNN'] = knn_grid.best_estimator_
    knn_acc = accuracy_score(y_te, models['KNN'].predict(X_te))
    print(f"KNN best params: {knn_grid.best_params_}")
    print(f"KNN accuracy: {knn_acc:.4f}")
    
    if knn_acc > best_score:
        best_score = knn_acc
        best_model_name = 'KNN'
    
    print("Training XGBoost model...")
    # XGBoost
    xgb_params = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.01]
    }
    xgb = XGBClassifier(random_state=42)
    xgb_grid = GridSearchCV(xgb, xgb_params, cv=3, scoring='accuracy', n_jobs=-1)
    xgb_grid.fit(X_tr_balanced, y_tr_balanced)
    models['XGBoost'] = xgb_grid.best_estimator_
    xgb_acc = accuracy_score(y_te, models['XGBoost'].predict(X_te))
    print(f"XGBoost best params: {xgb_grid.best_params_}")
    print(f"XGBoost accuracy: {xgb_acc:.4f}")
    
    if xgb_acc > best_score:
        best_score = xgb_acc
        best_model_name = 'XGBoost'
    
    print("Training Neural Network model...")
    # Neural Network
    mlp_params = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001]
    }
    mlp = MLPClassifier(max_iter=1000, random_state=42)
    mlp_grid = GridSearchCV(mlp, mlp_params, cv=3, scoring='accuracy', n_jobs=-1)
    mlp_grid.fit(X_tr_balanced, y_tr_balanced)
    models['NeuralNetwork'] = mlp_grid.best_estimator_
    mlp_acc = accuracy_score(y_te, models['NeuralNetwork'].predict(X_te))
    print(f"Neural Network best params: {mlp_grid.best_params_}")
    print(f"Neural Network accuracy: {mlp_acc:.4f}")
    
    if mlp_acc > best_score:
        best_score = mlp_acc
        best_model_name = 'NeuralNetwork'
    
    # Save the best model and all models
    print(f"✅ Best model: {best_model_name} with accuracy {best_score:.4f}")
    best_model = models[best_model_name]
    
    # Save models and metadata
    joblib.dump(best_model, "disease_model.pkl")
    joblib.dump(models, "all_models.pkl")
    joblib.dump(label_enc, "label_encoder.pkl")
    joblib.dump(all_sym, "symptom_columns.pkl")
    
    # Save model performance metadata
    model_info = {
        'best_model': best_model_name,
        'accuracies': {
            'RandomForest': rf_acc,
            'KNN': knn_acc,
            'XGBoost': xgb_acc,
            'NeuralNetwork': mlp_acc
        }
    }
    joblib.dump(model_info, "model_info.pkl")
    
    # Create database for history if it doesn't exist
    setup_database()
    
    print("✔️ Saved models, encoders, and set up database")
    return all_sym

class MatplotlibCanvas(FigureCanvas):
    """Matplotlib canvas for displaying charts in PyQt"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(self.fig)

class DiagnosisApp(QWidget):
    def __init__(self, model, label_enc, all_sym, precautions):
        super().__init__()
        self.model = model
        self.label_enc = label_enc
        self.all_sym = all_sym
        self.precautions = precautions
        self.dark_mode = False
        self.selected_symptoms = []
        self.diagnosis_history = []
        
        # Load all models if available
        try:
            self.all_models = joblib.load("all_models.pkl")
            self.model_info = joblib.load("model_info.pkl")
        except:
            self.all_models = {"MainModel": self.model}
            self.model_info = {"best_model": "MainModel"}
        
        # Organize symptoms into categories
        self.setup_symptom_categories()
        self.init_ui()
    
    def setup_symptom_categories(self):
        """Organize symptoms into categories"""
        self.categories = {
            "Respiratory": [s for s in self.all_sym if any(k in s.lower() for k in ['cough', 'breath', 'throat', 'sinus', 'lung'])],
            "Digestive": [s for s in self.all_sym if any(k in s.lower() for k in ['stomach', 'digest', 'vomit', 'nausea', 'diarrhea', 'abdominal'])],
            "Skin": [s for s in self.all_sym if any(k in s.lower() for k in ['skin', 'rash', 'itch', 'sweat'])],
            "Head & Neurological": [s for s in self.all_sym if any(k in s.lower() for k in ['head', 'brain', 'neuro', 'dizz', 'migraine', 'memory'])],
            "Pain & Discomfort": [s for s in self.all_sym if any(k in s.lower() for k in ['pain', 'ache', 'sore', 'discomfort', 'cramp'])],
            "Fever & Infection": [s for s in self.all_sym if any(k in s.lower() for k in ['fever', 'infection', 'swelling', 'inflam'])],
        }
        
        # Add "Other" category for uncategorized symptoms
        categorized = []
        for cat_syms in self.categories.values():
            categorized.extend(cat_syms)
        self.categories["Other"] = [s for s in self.all_sym if s not in categorized]

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle('🩺 Smart Diagnosis System')
        self.setMinimumSize(900, 700)
        
        # Main layout
        main_layout = QVBoxLayout()
        
        # Title and description
        title_label = QLabel('🩺 Smart Diagnosis System')
        title_label.setFont(QFont('Arial', 16, QFont.Bold))
        main_layout.addWidget(title_label)
        
        description = QLabel('Select your symptoms and get an AI-powered diagnosis with precaution tips.')
        main_layout.addWidget(description)
        
        # Create tab widget
        tabs = QTabWidget()
        
        # Tab 1: Diagnosis
        diagnosis_tab = QWidget()
        diagnosis_layout = QVBoxLayout(diagnosis_tab)
        
        # Symptom search and filter section
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel('Search Symptoms:'))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText('Type to search symptoms...')
        self.search_input.textChanged.connect(self.filter_symptoms)
        search_layout.addWidget(self.search_input)
        
        # Category dropdown
        search_layout.addWidget(QLabel('Category:'))
        self.category_combo = QComboBox()
        self.category_combo.addItems(['All'] + list(self.categories.keys()))
        self.category_combo.currentTextChanged.connect(self.on_category_changed)
        search_layout.addWidget(self.category_combo)
        diagnosis_layout.addLayout(search_layout)
        
        # Split the layout for symptom selection and results
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side - symptom selection
        symptom_widget = QWidget()
        symptom_layout = QVBoxLayout(symptom_widget)
        symptom_layout.addWidget(QLabel('Available Symptoms:'))
        
        self.symptom_list = QListWidget()
        self.symptom_list.setSelectionMode(QAbstractItemView.MultiSelection)
        for s in sorted(self.all_sym):
            item = QListWidgetItem(s)
            self.symptom_list.addItem(item)
        symptom_layout.addWidget(self.symptom_list)
        
        # Selected symptoms area
        symptom_layout.addWidget(QLabel('Selected Symptoms:'))
        self.selected_list = QListWidget()
        symptom_layout.addWidget(self.selected_list)
        
        # Buttons for symptoms
        sym_btn_layout = QHBoxLayout()
        self.add_btn = QPushButton('Add →')
        self.add_btn.clicked.connect(self.add_symptom)
        sym_btn_layout.addWidget(self.add_btn)
        
        self.remove_btn = QPushButton('← Remove')
        self.remove_btn.clicked.connect(self.remove_symptom)
        sym_btn_layout.addWidget(self.remove_btn)
        symptom_layout.addLayout(sym_btn_layout)
        
        # Right side - diagnosis results
        result_widget = QWidget()
        result_layout = QVBoxLayout(result_widget)
        
        # Patient information group
        patient_group = QGroupBox("Patient Information")
        patient_layout = QVBoxLayout()
        
        # Patient ID and Name fields
        id_layout = QHBoxLayout()
        id_layout.addWidget(QLabel("Patient ID:"))
        self.patient_id_input = QLineEdit()
        self.patient_id_input.setPlaceholderText("Enter unique ID")
        id_layout.addWidget(self.patient_id_input)
        patient_layout.addLayout(id_layout)
        
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Patient Name:"))
        self.patient_name_input = QLineEdit()
        self.patient_name_input.setPlaceholderText("Enter patient name")
        name_layout.addWidget(self.patient_name_input)
        patient_layout.addLayout(name_layout)
        
        patient_group.setLayout(patient_layout)
        result_layout.addWidget(patient_group)
        
        # Diagnosis buttons
        btn_layout = QHBoxLayout()
        self.diagnose_btn = QPushButton('🧠 Diagnose')
        self.diagnose_btn.setMinimumHeight(40)
        self.diagnose_btn.clicked.connect(self.on_diagnose)
        btn_layout.addWidget(self.diagnose_btn)
        
        self.reset_btn = QPushButton('🔄 Reset')
        self.reset_btn.clicked.connect(self.on_reset)
        btn_layout.addWidget(self.reset_btn)
        result_layout.addLayout(btn_layout)
        
        # Results display
        result_layout.addWidget(QLabel('Diagnosis Results:'))
        self.result_area = QTextEdit()
        self.result_area.setReadOnly(True)
        result_layout.addWidget(self.result_area)
        
        # Visualization area - initially empty
        result_layout.addWidget(QLabel('Probability Distribution:'))
        self.plot_widget = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_widget)
        self.canvas = MatplotlibCanvas(width=5, height=4)
        self.plot_layout.addWidget(self.canvas)
        result_layout.addWidget(self.plot_widget)
        self.plot_widget.setVisible(False)  # Hide until needed
        
        # Button to save to history
        self.save_btn = QPushButton('💾 Save to History')
        self.save_btn.clicked.connect(self.save_to_history)
        self.save_btn.setEnabled(False)
        result_layout.addWidget(self.save_btn)
        
        # Add widgets to splitter
        splitter.addWidget(symptom_widget)
        splitter.addWidget(result_widget)
        splitter.setSizes([400, 500])
        diagnosis_layout.addWidget(splitter)
        
        # Add first tab
        tabs.addTab(diagnosis_tab, '📋 Diagnosis')
        
        # Tab 2: History
        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)
        
        # Search box for patient name and ID
        search_group = QGroupBox("Search Patient Records")
        search_group_layout = QVBoxLayout()
        
        # Instructions label
        search_group_layout.addWidget(QLabel("Search by patient name or ID:"))
        
        # Search input layout
        search_input_layout = QHBoxLayout()
        self.history_search_input = QLineEdit()
        self.history_search_input.setPlaceholderText("Enter patient name or ID...")
        self.history_search_input.textChanged.connect(self.filter_history)
        search_input_layout.addWidget(self.history_search_input)
        
        # Search button
        search_btn = QPushButton('🔍 Search')
        search_btn.clicked.connect(self.filter_history)
        search_input_layout.addWidget(search_btn)
        
        # Clear search button
        clear_search_btn = QPushButton('Clear')
        clear_search_btn.clicked.connect(self.clear_history_search)
        search_input_layout.addWidget(clear_search_btn)
        
        search_group_layout.addLayout(search_input_layout)
        search_group.setLayout(search_group_layout)
        history_layout.addWidget(search_group)
        
        # History table
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(6)
        self.history_table.setHorizontalHeaderLabels([
            "Date & Time", "Patient ID", "Patient Name", "Disease", "Confidence", "Symptoms"
        ])
        self.history_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)  # Symptoms column stretches
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setSelectionBehavior(QTableWidget.SelectRows)  # Select entire rows
        history_layout.addWidget(self.history_table)
        
        # Details view button
        view_btn = QPushButton('🔍 View Selected Record Details')
        view_btn.clicked.connect(self.view_history_details)
        history_layout.addWidget(view_btn)
        
        # Export buttons
        export_layout = QHBoxLayout()
        self.export_excel_btn = QPushButton('📊 Export to Excel')
        self.export_excel_btn.clicked.connect(self.export_to_excel)
        export_layout.addWidget(self.export_excel_btn)
        
        self.refresh_history_btn = QPushButton('🔄 Refresh History')
        self.refresh_history_btn.clicked.connect(self.load_history)
        export_layout.addWidget(self.refresh_history_btn)
        history_layout.addLayout(export_layout)
        
        # Add history tab
        tabs.addTab(history_tab, '📜 History')
        
        # Tab 3: About & Models
        about_tab = QWidget()
        about_layout = QVBoxLayout(about_tab)
        
        # Model information
        about_layout.addWidget(QLabel('<h2>Smart Diagnosis System</h2>'))
        about_layout.addWidget(QLabel('<p>This system uses machine learning to predict diseases based on symptoms.</p>'))
        
        # Model info box
        model_info_group = QGroupBox("Model Information")
        model_info_layout = QVBoxLayout()
        
        # Show best model and accuracies
        try:
            best_model = self.model_info['best_model']
            model_info_layout.addWidget(QLabel(f"<b>Best model:</b> {best_model}"))
            
            accs = self.model_info.get('accuracies', {})
            for model_name, acc in accs.items():
                model_info_layout.addWidget(QLabel(f"{model_name}: {acc:.4f} accuracy"))
        except:
            model_info_layout.addWidget(QLabel("<i>Model information not available</i>"))
        
        model_info_group.setLayout(model_info_layout)
        about_layout.addWidget(model_info_group)
        about_layout.addStretch()
        
        # Add about tab
        tabs.addTab(about_tab, 'ℹ️ About')
        
        # Add tabs to main layout
        main_layout.addWidget(tabs)
        
        # Theme toggle and status bar
        bottom_layout = QHBoxLayout()
        self.theme_btn = QPushButton('🌙 Dark Mode')
        self.theme_btn.clicked.connect(self.toggle_theme)
        bottom_layout.addWidget(self.theme_btn)
        
        # Status label
        self.status_label = QLabel('Ready')
        bottom_layout.addWidget(self.status_label)
        main_layout.addLayout(bottom_layout)
        
        # Set consistent styling for the entire app
        self.setStyleSheet("""
            QWidget { font-family: 'Segoe UI', Arial, sans-serif; }
            QGroupBox { font-weight: bold; border: 1px solid #cccccc; border-radius: 5px; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px 0 5px; }
            QPushButton { background-color: #2980b9; color: white; border: none; padding: 8px; border-radius: 4px; }
            QPushButton:hover { background-color: #3498db; }
            QPushButton:pressed { background-color: #1c6ea4; }
            QLineEdit, QTextEdit, QListWidget, QComboBox { border: 1px solid #cccccc; border-radius: 4px; padding: 4px; }
            QTabWidget::pane { border: 1px solid #cccccc; }
            QTabBar::tab { background-color: #f0f0f0; padding: 8px 16px; margin-right: 2px; }
            QTabBar::tab:selected { background-color: #2980b9; color: white; }
        """)
        
        self.setLayout(main_layout)
        self.resize(1100, 750)

    def filter_symptoms(self):
        """Filter symptoms based on search text and selected category"""
        search_text = self.search_input.text().lower()
        category = self.category_combo.currentText()
        
        # Get the list of symptoms to display based on category
        if category == 'All':
            available_symptoms = self.all_sym
        else:
            available_symptoms = self.categories[category]
        
        # Clear and repopulate the list with filtered symptoms
        self.symptom_list.clear()
        for s in sorted(available_symptoms):
            if search_text in s.lower():
                item = QListWidgetItem(s)
                self.symptom_list.addItem(item)
    
    def on_category_changed(self, category):
        """Handle category selection change"""
        self.filter_symptoms()
    
    def add_symptom(self):
        """Add selected symptoms to the selected list"""
        for item in self.symptom_list.selectedItems():
            symptom = item.text()
            if symptom not in self.selected_symptoms:
                self.selected_symptoms.append(symptom)
                self.selected_list.addItem(symptom)
        self.symptom_list.clearSelection()
    
    def remove_symptom(self):
        """Remove symptoms from the selected list"""
        for item in self.selected_list.selectedItems():
            symptom = item.text()
            self.selected_symptoms.remove(symptom)
            self.selected_list.takeItem(self.selected_list.row(item))
    
    def toggle_theme(self):
        """Switch between light and dark themes"""
        self.dark_mode = not self.dark_mode
        if self.dark_mode:
            self.setStyleSheet("""
                QWidget { background-color: #2D2D30; color: #FFFFFF; font-family: 'Segoe UI', Arial, sans-serif; }
                QLabel { color: #FFFFFF; }
                QPushButton { background-color: #0078D7; color: white; border: none; padding: 8px; border-radius: 4px; }
                QPushButton:hover { background-color: #1C97EA; }
                QPushButton:pressed { background-color: #006CC1; }
                QTextEdit, QListWidget, QComboBox, QLineEdit, QTableWidget { 
                    background-color: #1E1E1E; 
                    color: #FFFFFF; 
                    border: 1px solid #3F3F46; 
                    border-radius: 4px;
                }
                QGroupBox { 
                    font-weight: bold; 
                    border: 1px solid #3F3F46; 
                    border-radius: 5px; 
                    margin-top: 10px; 
                    color: #FFFFFF; 
                }
                QGroupBox::title { 
                    subcontrol-origin: margin; 
                    left: 10px; 
                    padding: 0 5px 0 5px; 
                }
                QTabWidget::pane { border: 1px solid #3F3F46; }
                QTabBar::tab { background-color: #2D2D30; color: #FFFFFF; padding: 8px 16px; margin-right: 2px; }
                QTabBar::tab:selected { background-color: #0078D7; }
                QTableWidget { gridline-color: #3F3F46; }
                QTableWidget::item { padding: 4px; }
                QHeaderView::section { 
                    background-color: #252526; 
                    color: white; 
                    padding: 6px; 
                    border: 1px solid #3F3F46; 
                }
            """)
            self.theme_btn.setText('☀️ Light Mode')
        else:
            self.setStyleSheet("""
                QWidget { font-family: 'Segoe UI', Arial, sans-serif; }
                QGroupBox { font-weight: bold; border: 1px solid #cccccc; border-radius: 5px; margin-top: 10px; }
                QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px 0 5px; }
                QPushButton { background-color: #2980b9; color: white; border: none; padding: 8px; border-radius: 4px; }
                QPushButton:hover { background-color: #3498db; }
                QPushButton:pressed { background-color: #1c6ea4; }
                QLineEdit, QTextEdit, QListWidget, QComboBox, QTableWidget { border: 1px solid #cccccc; border-radius: 4px; padding: 4px; }
                QTabWidget::pane { border: 1px solid #cccccc; }
                QTabBar::tab { background-color: #f0f0f0; padding: 8px 16px; margin-right: 2px; }
                QTabBar::tab:selected { background-color: #2980b9; color: white; }
                QTableWidget { gridline-color: #d4d4d4; alternate-background-color: #f6f6f6; }
                QTableWidget::item { padding: 4px; }
                QHeaderView::section { background-color: #e0e0e0; padding: 6px; border: 1px solid #cccccc; }
            """)
            self.theme_btn.setText('🌙 Dark Mode')
    
    def save_to_history(self):
        """Save the current diagnosis to history and database with patient info"""
        if not hasattr(self, 'last_disease') or not hasattr(self, 'last_confidence'):
            QMessageBox.warning(self, "Warning", "No diagnosis to save")
            return
        
        # Get patient information
        patient_id = self.patient_id_input.text()
        patient_name = self.patient_name_input.text()
        
        # Validate patient information
        if not patient_id or not patient_name:
            QMessageBox.warning(self, "Missing Information", 
                               "Please enter both patient ID and name before saving.")
            return
        
        # Format current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Save to SQLite database
        conn = sqlite3.connect('diagnosis_history.db')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO diagnoses (patient_id, patient_name, timestamp, symptoms, predicted_disease, confidence, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (patient_id, patient_name, timestamp, ", ".join(self.selected_symptoms), 
             self.last_disease, self.last_confidence, "")
        )
        conn.commit()
        conn.close()
        
        # Add to in-memory history
        self.diagnosis_history.append({
            'patient_id': patient_id,
            'patient_name': patient_name,
            'timestamp': timestamp,
            'symptoms': self.selected_symptoms.copy(),
            'disease': self.last_disease,
            'confidence': self.last_confidence
        })
        
        # Show confirmation and update status
        self.status_label.setText(f"Diagnosis for {patient_name} (ID: {patient_id}) saved at {timestamp}")
        QMessageBox.information(self, "Success", f"Diagnosis for {patient_name} saved to history")
        
        # Clear patient fields for next diagnosis
        self.patient_id_input.clear()
        self.patient_name_input.clear()
    
    def on_diagnose(self):
        """Perform diagnosis and show results including visualization"""
        # Get symptoms from the selected list
        self.selected_symptoms = [self.selected_list.item(i).text() for i in range(self.selected_list.count())]
        
        if not self.selected_symptoms:
            self.result_area.setText('⚠️ Please select at least one symptom.')
            self.plot_widget.setVisible(False)
            self.save_btn.setEnabled(False)
            return
        
        # Prepare input data
        inp = pd.DataFrame([[1 if s in self.selected_symptoms else 0 for s in self.all_sym]], columns=self.all_sym)
        
        # Get prediction and probabilities
        pred = self.model.predict(inp)[0]
        self.last_disease = self.label_enc.inverse_transform([pred])[0]
        
        # Get prediction probabilities if model supports it
        try:
            proba = self.model.predict_proba(inp)[0]
            # Get top 5 predictions
            top_indices = proba.argsort()[-5:][::-1]
            top_diseases = self.label_enc.inverse_transform(top_indices)
            top_probas = proba[top_indices]
            self.last_confidence = top_probas[0]
            
            # Create visualization
            self.plot_diseases(top_diseases, top_probas)
        except:
            # If model doesn't support probabilities
            self.last_confidence = 1.0
            self.plot_widget.setVisible(False)
        
        # Get precautions
        if self.last_disease in self.precautions.index:
            tips = self.precautions.loc[self.last_disease].dropna().tolist()
        else:
            tips = ["No precautions available."]
        
        # Format output
        output_text = f"💡 **Predicted Disease: {self.last_disease}**\n\n"
        output_text += f"Confidence: {self.last_confidence:.2%}\n\n"
        output_text += "**Precautions:**\n"
        output_text += ''.join(f"- {tip}\n" for tip in tips)
        
        # Show results
        self.result_area.setText(output_text)
        self.save_btn.setEnabled(True)
        self.status_label.setText(f"Diagnosis complete: {self.last_disease}")
    
    def plot_diseases(self, diseases, probabilities):
        """Create a bar chart of disease probabilities"""
        self.canvas.axes.clear()
        bars = self.canvas.axes.bar(diseases, probabilities, color='#0078D7')
        
        # Set labels and title
        self.canvas.axes.set_title('Disease Probability Distribution')
        self.canvas.axes.set_ylabel('Probability')
        self.canvas.axes.set_xlabel('Disease')
        
        # Rotate x-axis labels for better readability
        self.canvas.axes.set_xticklabels(diseases, rotation=45, ha='right')
        
        # Add percentage labels on top of bars
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            self.canvas.axes.text(bar.get_x() + bar.get_width()/2., height,
                    f'{prob:.1%}', ha='center', va='bottom')
        
        self.canvas.fig.tight_layout()
        self.canvas.draw()
        self.plot_widget.setVisible(True)
    
    def on_reset(self):
        """Clear all selections and results"""
        self.search_input.clear()
        self.symptom_list.clearSelection()
        self.selected_symptoms = []
        self.selected_list.clear()
        self.result_area.clear()
        self.plot_widget.setVisible(False)
        self.save_btn.setEnabled(False)
        self.status_label.setText("Ready")
        self.category_combo.setCurrentIndex(0)
        
        # Keep patient information unless explicitly cleared
        # This makes it easier when diagnosing the same patient multiple times

    def load_history(self):
        """Load diagnosis history from database"""
        self.history_table.setRowCount(0)  # Clear table
        
        try:
            conn = sqlite3.connect('diagnosis_history.db')
            cursor = conn.cursor()
            cursor.execute(
                "SELECT timestamp, patient_id, patient_name, predicted_disease, confidence, symptoms, id "
                "FROM diagnoses ORDER BY timestamp DESC"
            )
            rows = cursor.fetchall()
            conn.close()
            
            # Store the full dataset for filtering
            self.history_data = rows
            
            # Display the data
            self.display_history_data(rows)
            
            self.status_label.setText(f"Loaded {len(rows)} diagnosis records")
        except Exception as e:
            self.status_label.setText(f"Error loading history: {str(e)}")
    
    def display_history_data(self, rows):
        """Display history data in the table"""
        self.history_table.setRowCount(len(rows))
        for i, (timestamp, patient_id, patient_name, disease, confidence, symptoms, record_id) in enumerate(rows):
            self.history_table.setItem(i, 0, QTableWidgetItem(timestamp))
            self.history_table.setItem(i, 1, QTableWidgetItem(patient_id or "N/A"))
            self.history_table.setItem(i, 2, QTableWidgetItem(patient_name or "N/A"))
            self.history_table.setItem(i, 3, QTableWidgetItem(disease))
            
            # Format confidence as percentage
            try:
                conf_value = float(confidence)
                conf_text = f"{conf_value:.2%}"
            except:
                conf_text = confidence or "N/A"
            self.history_table.setItem(i, 4, QTableWidgetItem(conf_text))
            
            self.history_table.setItem(i, 5, QTableWidgetItem(symptoms))
            
            # Store the record ID as hidden data in first column
            self.history_table.item(i, 0).setData(Qt.UserRole, record_id)
    
    def filter_history(self):
        """Filter history by patient name or ID"""
        search_text = self.history_search_input.text().lower()
        
        # If no search text, show all records
        if not search_text:
            self.display_history_data(self.history_data)
            return
        
        # Filter data based on patient name or ID
        filtered_data = []
        for row in self.history_data:
            patient_id = str(row[1] or "").lower()
            patient_name = str(row[2] or "").lower()
            
            if search_text in patient_id or search_text in patient_name:
                filtered_data.append(row)
        
        # Update display with filtered data
        self.display_history_data(filtered_data)
        self.status_label.setText(f"Found {len(filtered_data)} matching records")
    
    def clear_history_search(self):
        """Clear the history search box and show all records"""
        self.history_search_input.clear()
        self.display_history_data(self.history_data)
        self.status_label.setText(f"Showing all {len(self.history_data)} records")
    
    def view_history_details(self):
        """View detailed information for the selected history record"""
        selected_items = self.history_table.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "Selection Required", "Please select a record to view details")
            return
        
        # Get the selected row
        row = selected_items[0].row()
        
        # Get data for the selected record
        timestamp = self.history_table.item(row, 0).text()
        patient_id = self.history_table.item(row, 1).text()
        patient_name = self.history_table.item(row, 2).text()
        disease = self.history_table.item(row, 3).text()
        confidence = self.history_table.item(row, 4).text()
        symptoms = self.history_table.item(row, 5).text()
        
        # Create detailed view dialog
        details_dialog = QDialog(self)
        details_dialog.setWindowTitle(f"Patient Details: {patient_name}")
        details_dialog.setMinimumWidth(500)
        
        # Create layout
        layout = QVBoxLayout(details_dialog)
        
        # Add information fields
        layout.addWidget(QLabel(f"<h3>Record for: {patient_name}</h3>"))
        layout.addWidget(QLabel(f"<b>ID:</b> {patient_id}"))
        layout.addWidget(QLabel(f"<b>Date & Time:</b> {timestamp}"))
        layout.addWidget(QLabel(f"<b>Diagnosis:</b> {disease}"))
        layout.addWidget(QLabel(f"<b>Confidence:</b> {confidence}"))
        
        # Symptoms list
        layout.addWidget(QLabel("<b>Symptoms:</b>"))
        symptoms_text = QTextEdit()
        symptoms_text.setPlainText(symptoms)
        symptoms_text.setReadOnly(True)
        layout.addWidget(symptoms_text)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(details_dialog.close)
        layout.addWidget(close_btn)
        
        # Show dialog
        details_dialog.exec_()
    
    def export_to_excel(self):
        """Export diagnosis history to Excel file"""
        try:
            import pandas as pd
            from datetime import datetime
            
            # Get save location from user
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export History", 
                f"diagnosis_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                "Excel Files (*.xlsx);;All Files (*)"
            )
            
            if not filename:
                return  # User canceled
                
            # Connect to database and get all records
            conn = sqlite3.connect('diagnosis_history.db')
            history_df = pd.read_sql_query(
                "SELECT id, patient_id, patient_name, timestamp, symptoms, predicted_disease, confidence, notes "
                "FROM diagnoses ORDER BY timestamp DESC", 
                conn
            )
            
            # Rename columns for better readability in Excel
            history_df = history_df.rename(columns={
                'patient_id': 'Patient ID',
                'patient_name': 'Patient Name',
                'timestamp': 'Date & Time',
                'symptoms': 'Symptoms',
                'predicted_disease': 'Diagnosis',
                'confidence': 'Confidence',
                'notes': 'Notes'
            })
            
            conn.close()
            
            # Save to Excel
            history_df.to_excel(filename, index=False)
            self.status_label.setText(f"Exported {len(history_df)} records to {filename}")
            QMessageBox.information(self, "Export Successful", f"Exported {len(history_df)} diagnosis records to Excel")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export: {str(e)}")
            self.status_label.setText(f"Export error: {str(e)}")

def launch_pyqt_app():
    """Launch the PyQt application"""
    # Load models and data
    model        = joblib.load("disease_model.pkl")
    label_enc    = joblib.load("label_encoder.pkl")
    all_sym      = joblib.load("symptom_columns.pkl")
    
    # Load precautions data
    precautions  = pd.read_csv("Disease_precaution_cleaned.csv").set_index("Disease")
    
    # Create and show app
    app = QApplication(sys.argv)
    window = DiagnosisApp(model, label_enc, all_sym, precautions)
    window.load_history() # Load history on startup
    window.show()
    sys.exit(app.exec_())

def main():
    # Check for required data files
    required_files = [
        "DiseaseAndSymptoms_cleaned.csv", 
        "Disease_precaution_cleaned.csv"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"Error: Missing required data files: {', '.join(missing_files)}")
        print("Please make sure these files are in the current directory.")
        sys.exit(1)
    
    # Train and save models if not already present
    if not (os.path.exists("disease_model.pkl") and 
            os.path.exists("label_encoder.pkl") and 
            os.path.exists("symptom_columns.pkl")):
        print("Models not found. Training new models...")
        train_and_save_models()
    else:
        print("Loading existing models...")
    
    # Ensure database exists
    setup_database()
    
    # Launch the PyQt app
    try:
        launch_pyqt_app()
    except Exception as e:
        print(f"Error launching application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

