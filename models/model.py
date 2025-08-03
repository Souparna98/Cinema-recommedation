import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

class CinemaRecommender:
    def __init__(self):
        self.model = SVD()
        self.trainset = None

    def load_data(self, file_path):
        # Load dataset
        df = pd.read_csv(file_path)
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
        self.trainset = data.build_full_trainset()

    def train(self):
        # Train the model
        self.model.fit(self.trainset)

    def predict(self, user_id, movie_id):
        # Predict rating for a specific user and movie
        return self.model.predict(user_id, movie_id).est

    def evaluate(self, testset):
        # Evaluate the model
        predictions = self.model.test(testset)
        return accuracy.rmse(predictions)
