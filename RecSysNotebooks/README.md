# Algorithm 1:
### matrix factorization Basic
1. Concept:
Matrix Factorization algorithms decompose the sparse user-item rating matrix into two lower-dimensional matrices of latent factors: a user-latent feature matrix and an item-latent feature matrix. The product of these two matrices approximates the original rating matrix. Popular techniques include Singular Value Decomposition (SVD), Alternating Least Squares (ALS), and Non-negative Matrix Factorization (NMF).

2. Pros:
Scalability: Generally scales well to large datasets and can handle high sparsity efficiently by learning latent representations.
Performance: Often achieves higher accuracy than memory-based methods due to its ability to uncover hidden patterns and generalize from sparse data.
Sparsity Robustness: More robust to sparsity compared to memory-based CF.
Cold Start (Mitigation): Can provide reasonable recommendations even for users/items with few ratings by inferring latent features.

3. Cons:
Lack of Interpretability: Latent factors are abstract and do not directly correspond to human-understandable features, making recommendations harder to explain.
Computational Cost (Training): Training can still be computationally intensive for very large datasets, though often less so than memory-based methods.
Hyperparameter Tuning: Requires careful tuning of hyperparameters (e.g., number of latent factors, regularization parameters).
Cold Start (Still Present): While mitigated, entirely new users/items with no ratings still pose a challenge.

### Hybrid MF + content-based utilizing user subratings
1. Concept:
Hybrid approaches combine elements from different recommender system paradigms to leverage their respective strengths. A common hybrid approach is to combine collaborative filtering with content-based features. In this case, subratings can be treated as content features of the hotels or as part of the user's explicit preferences.

2. Pros:
Leverages All Data: Can effectively utilize both collaborative rating patterns and explicit subrating information.
Addresses Cold Start: Content features from subratings can help recommend items even if they have no ratings, by matching them with user preferences for specific attributes.
Improved Accuracy: Often outperforms pure CF or content-based methods by mitigating their individual weaknesses.

3. Cons:
Increased Complexity: More complex to design, implement, and tune than single-paradigm systems.
Feature Engineering: Requires careful feature engineering from subratings to make them useful for content-based components.
Potential for Feature Overload: Too many features derived from subratings could lead to increased dimensionality and computational challenges if not managed properly.