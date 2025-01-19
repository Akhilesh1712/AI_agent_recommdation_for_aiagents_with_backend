# src/recommendation.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def recommend_agents(user_query, dataset):
    """
    Recommends AI agents based on user input using TF-IDF and cosine similarity.

    Args:
        user_query (str): The user's input describing their requirements.
        dataset (pd.DataFrame): The dataset of AI agents.

    Returns:
        pd.DataFrame: Top recommended agents with their details and similarity scores.
    """
    # Combine tags and category for feature vectorization
    dataset['features'] = dataset['tags'] + ", " + dataset['category']

    # Initialize the TF-IDF vectorizer and fit the dataset features
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(dataset['features'])

    # Transform the user query into a vector
    user_vector = vectorizer.transform([user_query])

    # Compute similarity scores between user query and dataset
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix)

    # Add similarity scores to the dataset
    dataset['similarity'] = similarity_scores[0]

    # Sort the dataset by similarity score in descending order and return the top 3 recommendations
    top_recommendations = dataset[['name', 'category', 'description', 'similarity' , 'imagelink']].sort_values(by='similarity', ascending=False).head(3)

    return top_recommendations
