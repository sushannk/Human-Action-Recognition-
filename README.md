Human Action Recognition Using CNN-LSTM (UCF50 Dataset)

📌 Project Overview

This project implements a human action recognition system using a CNN-LSTM model trained on the UCF50 dataset. It effectively captures spatial and temporal patterns in video sequences, achieving 86.2% accuracy.

🛠️ Tech Stack

Python

TensorFlow & Keras

OpenCV

Flask

CNN (ResNet50, VGG16) & LSTM

Machine Learning & Deep Learning

🎯 Key Features

Optimized spatiotemporal feature learning by leveraging pretrained CNNs (ResNet50, VGG16) for feature extraction and LSTM for sequential modeling, improving classification.

Achieved 86.2% accuracy on action recognition, effectively detecting human movements across diverse action categories.

Designed and deployed a real-time action recognition system using Flask and OpenCV, enabling live action prediction with dynamic label overlays, evaluated through precision, recall, and F1-score analysis.

🚀 Model Training & Optimization

Preprocessed UCF50 dataset (frame extraction, resizing, normalization).

Applied CNN (ResNet50, VGG16) for feature extraction and LSTM for sequential learning.

Evaluated model using confusion matrix, precision, recall, and F1-score.

📌 Deployment

Built a real-time Flask web app for human action recognition.

Integrated OpenCV for live video input and action prediction.

📂 Project Structure

├── dataset/                 # UCF50 dataset
├── models/                  # Trained models (CNN-LSTM, ResNet50, VGG16)
├── src/
│   ├── preprocess.py        # Data preprocessing
│   ├── train.py             # Model training
│   ├── evaluate.py          # Model evaluation
│   ├── app.py               # Flask-based web app
│   ├── realtime.py          # Real-time action recognition with OpenCV
├── requirements.txt         # Required Python packages
├── README.md                # Project documentation

📊 Results

Accuracy: 86.2%

Loss: Optimized with categorical cross-entropy

Evaluation Metrics: Precision, Recall, and F1-score

📦 Installation & Usage

🔹 Clone the Repository

git clone https://github.com/your-username/action-recognition.git
cd action-recognition

🔹 Install Dependencies

pip install -r requirements.txt

🔹 Train the Model

python src/train.py

🔹 Run Real-Time Action Recognition

python src/realtime.py

🔹 Launch Web App

python src/app.py

📜 License

This project is licensed under the MIT License.

✉️ Contact

For any queries, reach out via LinkedIn or email your.email@example.com.

