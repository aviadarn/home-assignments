import os
import sys
import time
import json
import numpy as np
from pymongo import MongoClient
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

def perform_rag(query, collection, model, llm):
    print("Mode: Similarity Search / RAG")
    print("Embedding query...")
    query_embedding = model.encode(query)
    
    print("Fetching users with embeddings...")
    # Fetch only users that have the 'identity_embedding' field
    users = list(collection.find({"identity_embedding": {"$exists": True}}))
    
    if not users:
        print("No users with 'identity_embedding' found. Cannot perform RAG.")
        return

    print(f"Scanning {len(users)} users for similarity...")
    scored_users = []
    for user in users:
        user_embedding = np.array(user["identity_embedding"])
        score = cosine_similarity(query_embedding, user_embedding)
        scored_users.append((score, user))
    
    # Sort by score descending
    scored_users.sort(key=lambda x: x[0], reverse=True)
    top_k = 5
    top_users = scored_users[:top_k]
    
    print(f"\n--- Top {top_k} Similar Users ---")
    context_list = []
    for score, user in top_users:
        print(f"Score: {score:.4f} | ID: {user.get('id')} | Age: {user.get('age')} | Loc: {user.get('location')} | Duration: {user.get('duration')}")
        context_list.append(f"- User {user.get('id')}: Age {user.get('age')}, Gender {user.get('gender')}, Location {user.get('location')}, Home {user.get('home_location')}, Duration {user.get('duration')} mins.")
    
    context_text = "\n".join(context_list)
    
    rag_template = """
    You are an intelligent assistant. Use the following retrieved user data to answer the user request.
    
    Context (Similar Users):
    {context}
    
    User Query: "{query}"
    
    Answer:
    """
    prompt = PromptTemplate.from_template(rag_template)
    chain = prompt | llm
    print("\nGenerating answer...")
    response = chain.invoke({"context": context_text, "query": query})
    print("\n--- Final Answer ---")
    print(response)
    print("--------------------")

def perform_aggregation(query, collection, llm):
    print("Mode: Aggregation Analysis")
    schema_desc = """
    Collection 'users' has the following schema:
    - user_id: string (UUID)
    - location: string (Example: "NYC", "London", "Tel Aviv", "SF", "Tokyo")
    - home_location: string (Example: "NYC", "London", "Tel Aviv")
    - age: int (18-65)
    - gender: string ("male", "female")
    - duration: int (minutes spent at location)
    - timestamp: datetime
    
    Note: "middle age" usually means 35-55. 
    Note: "NY" likely refers to "NYC".
    """

    template = """
    You are a MongoDB expert. Given the following schema for a 'users' collection, translate the natural language query into a valid Python PyMongo aggregation pipeline list.
    
    Schema:
    {schema}
    
    Query: "{query}"
    
    Return ONLY the list of aggregation stages as a JSON array. Do not include any explanation, markdown formatting, or '```'. 
    IMPORTANT: Ensure all keys (like $match, $group, $sum, etc.) are enclosed in DOUBLE QUOTES.
    Example output format:
    [
        {{"$match": {{"age": {{"$gte": 30}}}}}},
        {{"$group": {{"_id": "$location", "total": {{"$sum": "$duration"}}}}}}
    ]
    """
    
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm
    response = chain.invoke({"schema": schema_desc, "query": query})
    
    # Clean response
    clean_response = response.strip()
    if clean_response.startswith("```json"):
        clean_response = clean_response[7:]
    if clean_response.startswith("```"):
        clean_response = clean_response[3:]
    if clean_response.endswith("```"):
        clean_response = clean_response[:-3]
    clean_response = clean_response.strip()

    print(f"Generated Pipeline:\n{clean_response}")
    try:
        pipeline = json.loads(clean_response)
        results = list(collection.aggregate(pipeline))
        print("\n--- Query Results ---")
        for res in results:
            print(res)
        print("---------------------")
    except Exception as e:
        print(f"Aggregation Failed: {e}")


def main():
    mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
    db_name = os.environ.get("DB_NAME", "user_data_db")
    collection_name = os.environ.get("COLLECTION_NAME", "users")
    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    query = os.environ.get("QUERY", "Where are middle age from NY are hanging most of the time?")
    model_name = os.environ.get("MODEL_NAME", "llama3")

    print("Initializing resources...")
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]
    
    # Load Embedding Model
    print("Loading embedding model 'all-MiniLM-L6-v2'...")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Load LLM
    print(f"Initializing Ollama '{model_name}'...")
    llm = OllamaLLM(model=model_name, base_url=ollama_url)
    
    print(f"Processing query: {query}")
    
    # Simple routing logic
    rag_keywords = ["similar", "like", "find users", "who matches", "resemble"]
    if any(k in query.lower() for k in rag_keywords):
        perform_rag(query, collection, embed_model, llm)
    else:
        perform_aggregation(query, collection, llm)

    client.close()

if __name__ == "__main__":
    main()
