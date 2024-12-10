
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

class SpamDetectionModel:
    def __init__(self):
        self.df = pd.read_csv('/mnt/data/project/data/Cleaned_Data.csv')
        self.df['Email'] = self.df.Email.apply(lambda email: str(email))
        self.Data = self.df.Email
        self.Labels = self.df.Label
        self.training_data, self.testing_data, self.training_labels, self.testing_labels = train_test_split(
            self.Data, self.Labels, random_state=10
        )
        self.vectorizer = TfidfVectorizer()
        self.training_vectors = self.vectorizer.fit_transform(self.training_data.to_list())
        
        # Initialize models
        self.models = {
            "Naive Bayes": MultinomialNB(),
            "Random Forest": RandomForestClassifier(n_estimators=19),
            "Logistic Regression": LogisticRegression(),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=9),
            "Support Vector Machines": SVC(probability=True)
        }
        
        # Train models
        for model in self.models.values():
            model.fit(self.training_vectors, self.training_labels)

    def get_prediction(self, vector):
        preds = [model.predict(vector)[0] for model in self.models.values()]
        spam_counts = preds.count(1)
        return 'Spam' if spam_counts >= 3 else 'Non-Spam'

    def get_probabilities(self, vector):
        probs = [model.predict_proba(vector)[0] * 100 for model in self.models.values()]
        return probs

    def get_vector(self, text):
        return self.vectorizer.transform([text])
