# Emotion_recognition
The project is aimed to detect 5 basic emotions by facial expression.

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
