import sys
import os

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from recsys.graph_recommender import GraphRecommender

def test_recommender():
    # Path to the ratings file
    ratings_path = 'data/processed_full/user_item_ratings.csv'
    
    print("Initializing GraphRecommender...")
    recommender = GraphRecommender(ratings_path)
    
    # Test with a user ID we saw earlier (e.g., 23)
    user_id = 23
    print(f"\nGenerating recommendations for User {user_id}...")
    recommendations = recommender.recommend(user_id, top_k=5)
    
    print(f"Top 5 recommendations for User {user_id}: {recommendations}")
    
    if not recommendations:
        print("No recommendations generated.")
    else:
        print("Success! Recommendations generated.")

if __name__ == "__main__":
    test_recommender()
