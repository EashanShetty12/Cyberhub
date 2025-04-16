# CyberSec

SecBERT-Based Phishing URL Detection

This project uses SecBERT and machine learning techniques to detect phishing URLs. It leverages transformer-based embeddings and a classifier to determine whether a URL is legitimate or malicious.

🚀 Features

Load and preprocess phishing and benign URL datasets

Generate embeddings using SecBERT

Train a logistic regression model for classification

Evaluate model performance

Predict if a URL is phishing or legitimate

🧠 Model

Tokenizer & Embeddings: jackaduma/SecBERT

Classifier: Logistic Regression using Scikit-learn

📦 Requirements

pip install torch transformers pandas scikit-learn joblib

📂 Dataset

Phishing URLs: PhishTank

Benign URLs: Manually curated from trusted websites (Google, Facebook, etc.)

🛠️ Usage

1. Download the dataset

wget http://data.phishtank.com/data/online-valid.csv

2. Train the model

# train_model.py
# (Contains code to load dataset, extract SecBERT embeddings, train and save the model)

3. Predict new URLs

from predict import check_url
print(check_url("http://secure-paypal-login.com"))
print(check_url("https://www.google.com"))

📈 Output

Phishing
Legitimate

💾 Model File

The trained model is saved as:

url_auth_detector.pkl

📄 License

This project is for educational and research purposes only.

✍️ Author

Built using Hugging Face Transformers, Scikit-learn, and PhishTank data.

Feel free to contribute or extend this with deep learning classifiers, SecRoBERTa, or online deployment!

