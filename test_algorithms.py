"""
DISCLAIMER THIS IS GENERATED CODE I WANTED TO TRY OUT THE RECSYS FRAMEWORK AND SEE IF IT WORKS
Test script for Popularity and Content-Based recommendation algorithms.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from recsys.individual.algorithms.popularity import PopularityAlgo
from recsys.individual.algorithms.content_based import ContentBasedAlgo

def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV data and ensure correct column types."""
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df):,} ratings from {file_path}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Users: {df['user'].nunique():,}, Items: {df['item'].nunique():,}")
    return df

def train_test_split(df: pd.DataFrame, test_ratio: float = 0.2, random_state: int = 42):
    """Split data into train/test ensuring each user has items in both sets."""
    np.random.seed(random_state)
    
    # For each user, randomly select test_ratio of their ratings for test
    train_list, test_list = [], []
    
    for user, group in df.groupby('user'):
        n = len(group)
        if n == 1:
            # Users with only 1 rating go to train
            train_list.append(group)
        else:
            # Split the rest
            n_test = max(1, int(n * test_ratio))
            indices = np.random.permutation(n)
            test_idx = group.iloc[indices[:n_test]]
            train_idx = group.iloc[indices[n_test:]]
            train_list.append(train_idx)
            test_list.append(test_idx)
    
    train = pd.concat(train_list, ignore_index=True)
    test = pd.concat(test_list, ignore_index=True)
    
    print(f"\nTrain: {len(train):,} ratings ({len(train)/len(df)*100:.1f}%)")
    print(f"Test:  {len(test):,} ratings ({len(test)/len(df)*100:.1f}%)")
    
    return train, test

def evaluate_recommendations(model, test: pd.DataFrame, train: pd.DataFrame, k: int = 10):
    """
    Evaluate recommendation quality using Hit Rate and NDCG@k.
    For each user in test, recommend k items and check if any test items are in recommendations.
    """
    hits = 0
    total_users = 0
    ndcg_sum = 0.0
    
    # Get all items to use as candidates
    all_items = set(train['item'].unique())
    
    test_by_user = test.groupby('user')['item'].apply(set).to_dict()
    
    for user, true_items in test_by_user.items():
        total_users += 1
        
        # Get items user already rated in training (to exclude)
        train_items = set(train[train['user'] == user]['item'])
        candidates = list(all_items - train_items)
        
        if not candidates:
            continue
            
        try:
            # Get recommendations
            recs = model.predict_for_user(user, candidates=candidates, n=k)
            
            if recs.empty:
                continue
                
            rec_items = set(recs['item'].tolist())
            
            # Hit Rate: did we recommend any item the user actually liked?
            if len(rec_items & true_items) > 0:
                hits += 1
            
            # NDCG: calculate normalized discounted cumulative gain
            dcg = 0.0
            for i, item in enumerate(recs['item'].tolist()):
                if item in true_items:
                    dcg += 1.0 / np.log2(i + 2)  # +2 because positions start at 0
            
            # Ideal DCG (if all k items were hits)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(true_items))))
            
            if idcg > 0:
                ndcg_sum += dcg / idcg
                
        except Exception as e:
            # Skip users that cause errors (e.g., cold start)
            continue
    
    hit_rate = hits / total_users if total_users > 0 else 0.0
    ndcg = ndcg_sum / total_users if total_users > 0 else 0.0
    
    return {
        'hit_rate': hit_rate,
        'ndcg': ndcg,
        'total_users': total_users,
        'users_with_hits': hits
    }

def main():
    print("=" * 70)
    print("TESTING RECOMMENDATION ALGORITHMS")
    print("=" * 70)
    
    # Load data
    data_path = "data/filtered/hotelrec_filtered_00001.csv"
    if not Path(data_path).exists():
        print(f"\nError: Data file not found at {data_path}")
        print("Please run the dataloader first to generate the filtered data.")
        return
    
    df = load_data(data_path)
    
    # Split data
    train, test = train_test_split(df, test_ratio=0.2)
    
    # Test Popularity Algorithm
    print("\n" + "=" * 70)
    print("TESTING POPULARITY ALGORITHM (count mode)")
    print("=" * 70)
    
    pop_count = PopularityAlgo(mode="count", min_count=5)
    print("Training...")
    pop_count.fit(train)
    print(f"Items in model: {len(pop_count.item_scores_):,}")
    
    print("\nTop 10 most popular items:")
    print(pop_count.item_scores_.head(10))
    
    print("\nEvaluating on test set...")
    metrics = evaluate_recommendations(pop_count, test, train, k=10)
    print(f"Hit Rate@10: {metrics['hit_rate']:.4f}")
    print(f"NDCG@10: {metrics['ndcg']:.4f}")
    print(f"Users evaluated: {metrics['total_users']:,}")
    print(f"Users with hits: {metrics['users_with_hits']:,}")
    
    # Test Popularity Algorithm (mean mode)
    print("\n" + "=" * 70)
    print("TESTING POPULARITY ALGORITHM (mean mode)")
    print("=" * 70)
    
    pop_mean = PopularityAlgo(mode="mean", min_count=5)
    print("Training...")
    pop_mean.fit(train)
    
    print("\nTop 10 highest rated items:")
    print(pop_mean.item_scores_.head(10))
    
    print("\nEvaluating on test set...")
    metrics = evaluate_recommendations(pop_mean, test, train, k=10)
    print(f"Hit Rate@10: {metrics['hit_rate']:.4f}")
    print(f"NDCG@10: {metrics['ndcg']:.4f}")
    
    # Test Content-Based Algorithm (if text data is available)
    print("\n" + "=" * 70)
    print("TESTING CONTENT-BASED ALGORITHM")
    print("=" * 70)
    
    if 'text' not in df.columns:
        print("Warning: No 'text' column in data. Content-based will only use properties if available.")
    
    try:
        cb = ContentBasedAlgo(min_df=3, max_features=5000, use_props=True, like_threshold=3.5)
        print("Training...")
        cb.fit(train)
        print(f"Items in model: {len(cb.items_):,}")
        print(f"Has text features: {cb.has_text_}")
        print(f"Has property features: {cb.has_props_}")
        
        if cb.has_text_:
            print(f"Vocabulary size: {len(cb.vectorizer_.vocabulary_):,}")
        
        # Test on a sample user
        sample_user = test['user'].iloc[0]
        print(f"\nSample recommendations for user: {sample_user}")
        recs = cb.predict_for_user(sample_user, n=10)
        print(recs)
        
        print("\nEvaluating on test set...")
        metrics = evaluate_recommendations(cb, test, train, k=10)
        print(f"Hit Rate@10: {metrics['hit_rate']:.4f}")
        print(f"NDCG@10: {metrics['ndcg']:.4f}")
        print(f"Users evaluated: {metrics['total_users']:,}")
        print(f"Users with hits: {metrics['users_with_hits']:,}")
        
    except Exception as e:
        print(f"Error testing Content-Based algorithm: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("TESTING COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
