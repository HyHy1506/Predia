Predia - Diabetes Prediction System
Predia is a web-based application built with Flask that uses machine learning to predict the likelihood of diabetes based on user-provided health metrics or CSV data uploads. It employs a Random Forest Classifier trained on a dataset with features like age, hypertension, BMI, HbA1c level, and blood glucose level, utilizing StandardScaler and PCA for preprocessing. The application features a responsive interface styled with Bootstrap and supports both individual and batch predictions.
Screenshots
Home Page

Prediction Input Form

Individual Prediction Result

Batch Prediction Result 

Table of Contents

Overview
Features by Role
Technologies
Project Structure
Setup Instructions
Usage
Contributing
License

Overview
Predia is designed to assist users in assessing their risk of diabetes by providing a simple interface to input health metrics or upload a CSV file with multiple patient records. The system uses a pre-trained Random Forest Classifier, with data preprocessing via StandardScaler and PCA, to predict diabetes probability. It supports two modes: individual predictions for single users and batch predictions for analyzing multiple patients via CSV files.
Features by Role
General User

Individual Prediction: Enter personal health metrics (age, hypertension status, BMI, HbA1c level, blood glucose level) to receive a percentage likelihood of diabetes.
Batch Prediction: Upload a CSV file containing patient data to predict diabetes probabilities for multiple individuals, displayed in a tabular format.
Home Page Information: View educational content about diabetes, including symptoms, complications, and the importance of early detection.
Responsive Interface: Access the application seamlessly on desktop and mobile devices, with a clean, Bootstrap-styled UI.
About Us: Learn about the development team and their mission to improve community health through AI and machine learning.

Developer/Researcher

Model Access: Utilize the pre-trained Random Forest Classifier (model.pkl), StandardScaler (sc.pkl), and PCA (pca.pkl) for custom predictions or further model training.
Dataset Exploration: Access the training dataset (BTD_dataset.csv) and Jupyter notebook (py_diabetes.ipynb) for data analysis and model development.
Custom CSV Processing: Process CSV files with specific headers (age, hypertension, bmi, HbA1c_level, blood_glucose_level) for batch predictions, with error handling for missing features.
Code Reusability: Leverage the modular Flask application structure and preprocessing scripts (py_diabetes_backup_v2.py) for extending functionality.

Technologies
Backend

Flask: Web framework for routing and request handling.
scikit-learn: Machine learning library for Random Forest Classifier, StandardScaler, and PCA.
pandas: Data manipulation for processing CSV files and feature extraction.
numpy: Numerical computations for array operations.
pickle: Serialization of pre-trained model and preprocessing objects.
gunicorn: WSGI server for production deployment.

Frontend

Jinja2: Templating engine for dynamic HTML rendering.
Bootstrap 5: Responsive UI framework for styling.
jQuery: JavaScript library for DOM manipulation and form handling.
HTML/CSS: Structure and styling of web pages.
JavaScript: Client-side interactivity for toggling input fields based on CSV upload selection.

Data Processing

SMOTE: Oversampling technique for handling imbalanced datasets.
StandardScaler: Feature scaling for model input.
PCA: Dimensionality reduction for improved model performance.

Project Structure
Predia/
├── BTD_dataset.csv                # Training dataset
├── Procfile                       # Deployment configuration for gunicorn
├── README.md                      # Project documentation
├── app.py                         # Flask application with routes and prediction logic
├── model.pkl                      # Pre-trained Random Forest Classifier
├── pca.pkl                        # PCA object for dimensionality reduction
├── py_diabetes.ipynb              # Jupyter notebook for data analysis and model training
├── py_diabetes_backup_v2.py       # Backup script for model training and preprocessing
├── requirements.txt               # Python dependencies
├── sc.pkl                         # StandardScaler object for feature scaling
├── static/
│   ├── css/
│   │   ├── predict.css            # Styles for prediction page
│   │   └── style.css              # General styles for the application
│   ├── img/
│   │   ├── stephen-andrews-GwgFPDXiSIs-unsplash.jpg  # Background image
│   │   └── team.jpg               # Team image for About Us section
│   └── js/
│       ├── predict.js             # JavaScript for toggling input fields
│       └── script.js              # General JavaScript functionality
└── templates/
    ├── base.html                  # Base HTML template
    ├── index.html                 # Home page template
    └── predict.html               # Prediction page template

Setup Instructions
Prerequisites

Python 3.8 or later
pip: Python package manager
Git: Version control system
Web Browser: For accessing the application

Application Setup

Clone the repository:git clone https://github.com/HyHy1506/Predia.git
cd Predia


Create and activate a virtual environment:python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:pip install -r requirements.txt


Download pre-trained models and preprocessing objects:
model.pkl: Download
sc.pkl: Download
pca.pkl: Download
Place these files in the project root directory.


Run the Flask application:python app.py


Access the application at http://localhost:5000.

Deployment (Optional)
To deploy on a platform like Heroku:

Ensure the Procfile is correctly set up (web: gunicorn app:app).
Install the Heroku CLI and follow these commands:heroku create
git push heroku main
heroku open



CSV File Format for Batch Predictions
For batch predictions, the CSV file must have the following columns (semicolon ; delimiter):

age: Age of the patient (numeric)
hypertension: 0 (no hypertension) or 1 (has hypertension)
bmi: Body Mass Index (numeric)
HbA1c_level: HbA1c percentage (numeric)
blood_glucose_level: Blood glucose level in mg/dL (numeric)

Example CSV content:
age;hypertension;bmi;HbA1c_level;blood_glucose_level
40;0;20.6;4.5;90
35;1;28.2;7.2;130

Usage

Home Page: Navigate to http://localhost:5000 to view information about diabetes and access the prediction feature.
Prediction Page: Go to /predict to:
Individual Prediction: Enter your age, hypertension status, BMI, HbA1c level, and blood glucose level, then submit to see the diabetes probability.
Batch Prediction: Check the "Use CSV File" option, upload a CSV file with patient data, and submit to view a table of predicted probabilities.


View Results: After submission, results are displayed on the same page, either as a single probability percentage or a table for batch predictions.
Return to Input: Use the "Quay về" button to return to the input form after viewing results.

Contributing
Contributions are welcome! Follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Commit your changes (git commit -m 'Add your feature').
Push to the branch (git push origin feature/your-feature).
Open a Pull Request.

Please ensure code follows PEP 8 guidelines and includes appropriate documentation.
License
This project is licensed under the MIT License - see the LICENSE file for details.
