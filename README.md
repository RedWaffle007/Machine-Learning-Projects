# IMDB Sentiment Analysis

This project performs sentiment analysis on the IMDB movie reviews dataset using various machine learning models and techniques. The models used include:

- **K-Nearest Neighbors (KNN)**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Randomized SearchCV for hyperparameter tuning**
- **Recurrent Neural Network (RNN)** for deep learning-based sentiment classification

## Features
- **Data Preprocessing**: Text cleaning, tokenization, and vectorization of reviews.
- **Model Evaluation**: Accuracy, precision, recall, and F1 score.
- **Hyperparameter Tuning**: Using Randomized SearchCV to optimize models.

## Installation

```bash
git clone https://github.com/RedWaffle007/Machine-Learning-Projects.git
cd Machine-Learning-Projects/nlp_project
pip install -r requirements.txt
```

## Usage

1. Load the dataset and preprocess the reviews.
2. Train models (KNN, Decision Tree, Random Forest).
3. Tune models using RandomizedSearchCV for best performance.
4. Evaluate the models and compare their performance.
5. Use an RNN for deep learning-based sentiment analysis.

## Results

The models' performance is compared in terms of accuracy. After evaluating multiple models, the Random Forest Classifier with Randomized SearchCV performed the best, achieving an accuracy of 84% on the test set. The Recurrent Neural Network (RNN) also showed promising results i.e., 74% on test set but was outperformed by the Random Forest model in this particular dataset.

