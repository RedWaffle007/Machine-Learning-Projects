**Sentiment Analysis on IMDB Movie Reviews with KNN**

__🌟 Project Overview__
This project performs **sentiment analysis on the IMDB Movie Reviews dataset using the K-Nearest Neighbors (KNN) algorithm.** By preprocessing the raw text data manually with SpaCy, I transformed the reviews into a format suitable for machine learning. I applied RandomizedSearchCV for hyperparameter tuning to improve the model’s performance, achieving an accuracy of 59%.
The project demonstrates my proficiency in machine learning, natural language processing (NLP), and model optimization, making it a valuable addition to my skill set.

__🔑 Key Features__
Dataset: IMDB Movie Reviews, preprocessed using SpaCy for text normalization and tokenization.
Model: K-Nearest Neighbors (KNN), suitable for text classification tasks.
Optimization: Hyperparameter tuning via RandomizedSearchCV to improve the model’s performance.
Evaluation: Model accuracy of 59% and detailed performance analysis using metrics like confusion matrix and classification report.

__🛠️ Technologies Used__
Python
SpaCy for advanced text preprocessing
Scikit-learn for building the KNN model and hyperparameter tuning
Pandas and NumPy for data manipulation

__🚀 Getting Started__
Prerequisites
Ensure you have Python 3.8+ and SpaCy installed. Install SpaCy via:

bash
Copy code
pip install spacy
Setup Instructions
Clone the repository:
bash
Copy code
git clone https://github.com/RedWaffle007/Machine-Learning-Projects.git
cd Machine-Learning-Projects
Open the Jupyter Notebook:
bash
Copy code
jupyter lab knnproject.ipynb
Run the Project
Execute all cells in the notebook to preprocess the data, train the model, tune hyperparameters, and analyze the results.

__📊 Results__
Achieved 59% accuracy after hyperparameter tuning, which reflects the performance of the KNN model on the IMDB dataset.
Challenges faced: Text data preprocessing, class imbalance, and the challenge of fine-tuning the KNN hyperparameters for improved performance.
Key Insights: The model successfully identifies sentiment trends in movie reviews, although there is room for improvement in terms of accuracy.

__🎯 Future Scope__
Explore advanced algorithms like Support Vector Machines (SVM) or Neural Networks for higher accuracy.
Experiment with techniques to handle class imbalance and improve model robustness.
Real-world application: Integrate the model into a web app for real-time sentiment analysis of user feedback in movie-related applications.
