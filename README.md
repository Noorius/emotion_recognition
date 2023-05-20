# Emotion_recognition
The project is aimed to detect 5 basic emotions by facial expression.
1. Happy
2. Sad
3. Surprised
4. Angry
5. Neutral

# Project structure
- app - The Flask Application for Model Presentation
- csv
  - MediaPipe - .csv file generated with distances
- image - The folder with the images dataset
- models - The folder with traned models
  - CNN - Convolutional Neural Network Model
  - GradientBoosting - Gradient Boosting Decision Tree Model
  - MLP - Multi-Layer Perceptron Model
- var - Standard Scaler object

# File Description
- dataset preparation - prepare .csv files with distances
- emotion-detection-resnet - trains CNN model based on images
- GradientBoosDecisionTree train Gradient Boosting Decision Tree based on .csv with distances
- gridsearch - Finds best hyperparameters for GradBoost
- MLP - train Multi-Layer Perceptron neural network based on .csv with distances
- PCA - implements Principal Components Analysis for dimensionality reduction for .csv distances
- Project Final - Implements Classical Machine Learning based on .csv with distances
- video - Test CNN model
- video-Tabular - Test GradBoost, MLP, ML models

# How to start Flask?
- Make sure you trained your models, put them in root directory of apps/Emotion Detection App
- Activate you virtual env, or create a new with requirements.txt
- Change directory to flask app root folder: cd "../apps/Emotion Detection App"
- Start the command: python camera_flask_app.py
- Go to http://127.0.0.1:5000/ on your browser

![image](https://github.com/Noorius/emotion_recognition/assets/78252057/9c811d30-2fdd-4b13-93f7-fa4d6d842e08)


*KBTU, 2023* 🎉
