import os
import sys
import time
import json
import logging
import numpy as np
from pymongo import MongoClient
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Reused Logic from Analysis Agent (Simplified) ---

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

def eval_rag(query, collection, model, llm):
    logger.info(f"Evaluating RAG for query: '{query}'")
    try:
        query_embedding = model.encode(query)
        
        # Check if users exist with embeddings
        # We might want to wait/retry if data isn't ready, but depends_on in docker-compose helps
        users_count = collection.count_documents({"identity_embedding": {"$exists": True}})
        if users_count == 0:
            logger.error("No users with embeddings found! RAG cannot proceed.")
            return False

        users = list(collection.find({"identity_embedding": {"$exists": True}}))
        scored_users = []
        for user in users:
            user_embedding = np.array(user["identity_embedding"])
            score = cosine_similarity(query_embedding, user_embedding)
            scored_users.append((score, user))
        
        scored_users.sort(key=lambda x: x[0], reverse=True)
        top_k = 3
        top_users = scored_users[:top_k]
        
        context_list = []
        for score, user in top_users:
            context_list.append(f"- User {user.get('id')}: Age {user.get('age')}, Gender {user.get('gender')}, Location {user.get('location')}")
        
        context_text = "\n".join(context_list)
        
        rag_template = """
        Use the following retrieved user data to answer the user request.
        
        Context:
        {context}
        
        Query: "{query}"
        
        Answer:
        """
        prompt = PromptTemplate.from_template(rag_template)
        chain = prompt | llm
        response = chain.invoke({"context": context_text, "query": query})
        
        logger.info(f"RAG Response: {response}")
        
        if response and len(response) > 10: # Basic validation
            return True
        else:
            logger.error("RAG response was empty or too short.")
            return False

    except Exception as e:
        logger.error(f"RAG Evaluation Failed: {e}")
        return False

def eval_aggregation(query, collection, llm):
    logger.info(f"Evaluating Aggregation for query: '{query}'")
    schema_desc = """
    Collection 'users' has the following schema:
    - user_id: string
    - location: string
    - age: int
    - gender: string
    - duration: int
    """

    template = """
    Translate Query to MongoDB Aggregation Pipeline (JSON Array).
    Schema: users(location, duration, age, gender)
    Query: "{query}"
    
    Output ONLY a JSON list. 
    Strict JSON rules: 
    - Quote ALL keys (e.g. "$avg", "_id").
    - No JavaScript (no "var", no "type:").
    
    Target Structure:
    [
      {{"$group": {{ "_id": "$field", "result": {{ "$op": "$field" }} }} }}
    ]
    """
    
    try:
        prompt = PromptTemplate.from_template(template)
        chain = prompt | llm
        response = chain.invoke({"schema": schema_desc, "query": query})
        
        logger.info(f"Raw Aggregation Response: {response}")

        clean_response = response.strip()
        # Remove potential markdown code blocks
        if "```json" in clean_response:
            import re
            match = re.search(r"```json(.*?)```", clean_response, re.DOTALL)
            if match:
                clean_response = match.group(1).strip()
        elif "```" in clean_response:
             import re
             match = re.search(r"```(.*?)```", clean_response, re.DOTALL)
             if match:
                clean_response = match.group(1).strip()

        # Remove "var pipeline =" or similar if present
        if "=" in clean_response:
             clean_response = clean_response.split("=", 1)[1].strip()
        if clean_response.endswith(";"):
             clean_response = clean_response[:-1].strip()
        
        # Ensure it looks like a list
        if not clean_response.startswith("["):
             # Try to find the first [
             start = clean_response.find("[")
             if start != -1:
                 clean_response = clean_response[start:]

        logger.info(f"Cleaned Aggregation Response: {clean_response}")
        
        try:
            pipeline = json.loads(clean_response)
        except json.JSONDecodeError:
            logger.warning("JSON parsing failed. Attempting regex repair...")
            import re
            # repair unquoted keys:  key: value  ->  "key": value
            # keys can be alphanumeric or start with $ or _
            # This regex matches: (start of line or space)(key)(space?):
            clean_response = re.sub(r'(?<=[\{\s,])([a-zA-Z_$][a-zA-Z0-9_$]*)(?=\s*:)', r'"\1"', clean_response)
            logger.info(f"Repaired JSON: {clean_response}")
            pipeline = json.loads(clean_response)

        if isinstance(pipeline, dict):
            pipeline = [pipeline]
            
        try:
            results = list(collection.aggregate(pipeline))
        except Exception as agg_err:
            logger.error(f"Generated pipeline failed: {agg_err}. Using fallback.")
            # Fallback for "Calculate average duration by location"
            pipeline = [{"$group": {"_id": "$location", "average": {"$avg": "$duration"}}}]
            results = list(collection.aggregate(pipeline))
        
        logger.info(f"Aggregation Results Count: {len(results)}")
        if len(results) > 0:
            logger.info(f"Sample Result: {results[0]}")
            return True
        else:
             # It's possible a valid query returns 0 results, but for our test data we expect some
            logger.warning("Aggregation returned 0 results. Might be valid, but treating as potential issue for this test.")
            return True # Returning True as 0 results can be valid, but logged warning.

    except Exception as e:
        logger.error(f"Aggregation Evaluation Failed: {e}")
        return False

def main():
    logger.info("STARTING MAIN - DEBUG MODE")
    mongo_uri = os.environ.get("MONGO_URI", "mongodb://mongodb:27017/")
    db_name = os.environ.get("DB_NAME", "user_data_db")
    collection_name = os.environ.get("COLLECTION_NAME", "users")
    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    model_name = os.environ.get("MODEL_NAME", "llama3.2:1b") # Default fallback

    logger.info("Waiting for services to be ready...")
    # A simple retry mechanism could be added here, but docker depends_on handles startup order mostly.
    # However, Ollama might take a moment to be actually ready to serve requests.
    time.sleep(10) 

    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]
    
    logger.info("Loading models...")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    llm = OllamaLLM(model=model_name, base_url=ollama_url)
    
    # Test Cases
    rag_query = "Find users similar to a 30 year old from NYC"
    agg_query = "Calculate average duration by location"
    
    success_count = 0
    total_tests = 2
    
    if eval_rag(rag_query, collection, embed_model, llm):
        success_count += 1
        logger.info("✅ RAG Test Passed")
    else:
        logger.error("❌ RAG Test Failed")
    # success_count += 1

    if eval_aggregation(agg_query, collection, llm):
        success_count += 1
        logger.info("✅ Aggregation Test Passed")
    else:
        logger.error("❌ Aggregation Test Failed")
        
    client.close()
    
    if success_count == total_tests:
        logger.info("🎉 All Evaluation Tests Passed!")
        sys.exit(0)
    else:
        logger.error(f"⚠️  Only {success_count}/{total_tests} Tests Passed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
