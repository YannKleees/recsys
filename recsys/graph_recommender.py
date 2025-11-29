import pandas as pd
import networkx as nx
from collections import Counter

class GraphRecommender:
    def __init__(self, ratings_path):
        """
        Initialize the GraphRecommender.
        
        Args:
            ratings_path (str): Path to the user-item ratings CSV file.
        """
        self.ratings_path = ratings_path
        self.graph = None
        self.user_nodes = set()
        self.item_nodes = set()
        self._load_data_and_build_graph()

    def _load_data_and_build_graph(self):
        """
        Load data from CSV and build the bipartite graph.
        """
        print(f"Loading data from {self.ratings_path}...")
        # Read only necessary columns
        df = pd.read_csv(self.ratings_path, usecols=['user_id', 'item_id', 'rating'])
        
        # Initialize graph
        self.graph = nx.Graph()
        
        # Add nodes and edges
        # We prefix nodes to distinguish between users and items in case of ID collision
        # though usually they are distinct. Let's keep them raw for now but assume
        # the user handles ID overlap or we can use a type attribute.
        # To be safe, let's use string prefixes internally.
        
        print("Building graph...")
        for _, row in df.iterrows():
            user_node = f"u_{int(row['user_id'])}"
            item_node = f"i_{int(row['item_id'])}"
            rating = row['rating']
            
            self.graph.add_node(user_node, type='user')
            self.graph.add_node(item_node, type='item')
            self.graph.add_edge(user_node, item_node, weight=rating)
            
            self.user_nodes.add(user_node)
            self.item_nodes.add(item_node)
            
        print(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

    def recommend(self, user_id, top_k=10):
        """
        Generate recommendations for a user.
        
        Args:
            user_id (int): The user ID to generate recommendations for.
            top_k (int): Number of recommendations to return.
            
        Returns:
            list: List of recommended item IDs.
        """
        user_node = f"u_{user_id}"
        
        if user_node not in self.graph:
            print(f"User {user_id} not found in graph.")
            return []
        
        # 1. Get items rated by this user
        rated_items = set(self.graph.neighbors(user_node))
        
        # 2. Find similar users (neighbors of these items)
        similar_users = []
        for item in rated_items:
            similar_users.extend(self.graph.neighbors(item))
            
        # Filter out the user themselves
        similar_users = [u for u in similar_users if u != user_node]
        
        # 3. Find items rated by similar users
        candidate_items = []
        for u in similar_users:
            # We can weight this by the rating the similar user gave, 
            # or just frequency. Let's use simple frequency for now.
            candidate_items.extend(self.graph.neighbors(u))
            
        # 4. Score items
        item_counts = Counter(candidate_items)
        
        # 5. Filter out already rated items
        recommendations = []
        for item, count in item_counts.most_common():
            if item not in rated_items:
                # Convert back to original item ID
                item_id = int(item.split('_')[1])
                recommendations.append(item_id)
                if len(recommendations) >= top_k:
                    break
                    
        return recommendations
