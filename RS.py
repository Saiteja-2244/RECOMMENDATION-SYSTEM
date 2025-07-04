# Install the surprise library if not already installed:
# pip install scikit-surprise

from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Example data: userId, itemId, rating
import pandas as pd
ratings_dict = {
    "userId": [1, 1, 1, 2, 2, 3, 3, 3],
    "itemId": [1, 2, 3, 2, 3, 1, 2, 3],
    "rating": [5, 3, 2, 4, 1, 2, 5, 4],
}
df = pd.DataFrame(ratings_dict)

# Define a Reader and load the data
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[["userId", "itemId", "rating"]], reader)

# Split into train and test sets
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

# Use SVD for matrix factorization
algo = SVD()
algo.fit(trainset)

# Predict ratings for the test set
predictions = algo.test(testset)

# Evaluate the model
print("RMSE:", accuracy.rmse(predictions))

# Example: Predict rating for user 1 on item 3
pred = algo.predict(uid=1, iid=3)
print(f"Predicted rating for user 1 on item 3: {pred.est:.2f}")
