import pandas as pd
from models.model import CinemaRecommender

def main():
    recommender = CinemaRecommender()
    recommender.load_data('data/raw/movies.csv')
    recommender.train()
    print("Model trained successfully!")

if __name__ == "__main__":
    main()

