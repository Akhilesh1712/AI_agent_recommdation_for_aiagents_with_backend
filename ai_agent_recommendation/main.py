# main.py

from ai_agent_recommendation.backend.dataset import get_dataset
from ai_agent_recommendation.backend.recommendation import recommend_agents

def main():
    # Load dataset
    dataset = get_dataset()

    # User query
    user_query = input("Enter your requirements (e.g., 'I need help with academic writing and planning'): ")

    # Get recommendations
    recommendations = recommend_agents(user_query, dataset)

    # Display recommendations
    print("\nTop Recommendations:")
    print(recommendations)

if __name__ == "__main__":
    main()
