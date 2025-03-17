**Diabetes Prediction - Streamlit App**

**Overview**

This project is a Diabetes Prediction Tool built using Streamlit. It allows users to input health-related parameters and predicts the likelihood of diabetes using a pre-trained machine learning model.

**Features**

User-friendly interface to input health metrics.

Predicts diabetes likelihood based on user input.

Data preprocessing and feature scaling.

Model trained using Scikit-learn.

Interactive visualizations for data insights.

**Tech Stack**

Python (3.9.6)

Streamlit (Web App Framework)

Scikit-learn (1.3.0) – for ML model

Joblib (1.4.2) – for model serialization

Numpy (1.26.4)

Pandas, Matplotlib, Seaborn (for data analysis & visualization)
**Installation & Setup**

1️⃣ **Clone the Repository**

git clone https://github.com/your-username/your-repo.git

cd your-repo

2️⃣ **Install Dependencies**

pip install -r requirements.txt

3️⃣ **Run Locally**
streamlit run src/app.py

**Deployment on Streamlit Cloud**

Push your code to GitHub.

Go to Streamlit Cloud and connect your repo.

Ensure requirements.txt and runtime.txt are present.

Deploy and monitor logs for any errors.

**Troubleshooting**

If the app runs locally but fails on Streamlit Cloud:

Ensure dependencies are correctly listed in requirements.txt.

Check for OS compatibility issues (Windows vs. Mac).

Add runtime.txt with the correct Python version (python-3.9.6).

Review logs for missing packages or path errors.








