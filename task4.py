import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# 🎯 Sample dataset (user-item ratings)
data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 4, 4, 5],
    'item_id': ['A', 'B', 'C', 'A', 'C', 'B', 'D', 'C', 'D', 'A'],
    'rating': [5, 3, 2, 4, 3, 4, 5, 2, 4, 3]
}

df = pd.DataFrame(data)

# 🧱 Create user-item matrix
user_item_matrix = df.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)

# 🔍 Compute user-user similarity
similarity_matrix = cosine_similarity(user_item_matrix)
similarity_df = pd.DataFrame(similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)

# 👤 Pick a user to recommend for
target_user = 1
target_ratings = user_item_matrix.loc[target_user]
unrated_items = target_ratings[target_ratings == 0].index

# 🔮 Predict ratings for unrated items
predicted_ratings = {}
for item in unrated_items:
    weighted_sum = 0
    sim_sum = 0
    for other_user in user_item_matrix.index:
        if other_user != target_user and user_item_matrix.loc[other_user, item] > 0:
            similarity = similarity_df.loc[target_user, other_user]
            rating = user_item_matrix.loc[other_user, item]
            weighted_sum += similarity * rating
            sim_sum += similarity
    if sim_sum > 0:
        predicted_ratings[item] = weighted_sum / sim_sum

# 📝 Sort and recommend top items
recommended_items = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)

# 📢 Output recommendations
print(f"Top recommendations for user {target_user}:")
for item, score in recommended_items:
    print(f"Item {item} - Predicted Rating: {score:.2f}")
