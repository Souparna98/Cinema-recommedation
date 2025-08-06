import pandas as pd
from models.model import CinemaRecommender
from surprise import Dataset, Reader

def main():
    recommender = CinemaRecommender()
    recommender.load_data('data/raw/movies.csv')
    
    # Split data into train and test sets
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(pd.read_csv('data/raw/movies.csv')[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2)

    recommender.train()
    rmse = recommender.evaluate(testset)
    print(f"RMSE: {rmse}")

if __name__ == "__main__":
    main()
