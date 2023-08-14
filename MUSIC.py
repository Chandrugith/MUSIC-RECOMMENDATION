from surprise import Dataset, Reader, KNNBasic
import pandas as pd

# Sample data (user_id, song_id, rating)
data = {
    'user_id': ['user1', 'user1', 'user2', 'user3', 'user4'],
    'song_id': ['song1', 'song2', 'song1', 'song3', 'song2'],
    'rating': [5, 4, 3, 4, 5]
}

# Convert data to a pandas DataFrame
df = pd.DataFrame(data)

# Define the rating scale (from 1 to 5)
reader = Reader(rating_scale=(1, 5))

# Load the data into the Surprise Dataset
data = Dataset.load_from_df(df[['user_id', 'song_id', 'rating']], reader)

# Build the collaborative filtering model (KNNBasic)
model = KNNBasic()

# Train the model on the data
trainset = data.build_full_trainset()
model.fit(trainset)

# Get recommendations for a specific user
user_id = 'user1'
user_idx = trainset.to_inner_uid(user_id)
recommendations = model.get_neighbors(user_idx, k=3)  # Get top 3 similar users

print(f"Recommendations for user {user_id}:")
for rec_idx in recommendations:
    raw_uid = trainset.to_raw_uid(rec_idx)
    print(f"Song: {df[df['user_id'] == raw_uid]['song_id'].values[0]}")
