import os
import json
import numpy as np
import streamlit as st
from pymongo import MongoClient
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer

# --- Configuration & Caching ---

@st.cache_resource
def get_db_collection():
    mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
    db_name = os.environ.get("DB_NAME", "user_data_db")
    collection_name = os.environ.get("COLLECTION_NAME", "users")
    
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]
    return collection

@st.cache_resource
def get_models():
    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    model_name = os.environ.get("MODEL_NAME", "llama3.2:1b")
    
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    llm = OllamaLLM(model=model_name, base_url=ollama_url)
    return embed_model, llm

# --- shared logic ---

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

def perform_rag(query, collection, model, llm):
    st.info("🧠 Performing Similarity Search (RAG)...")
    
    query_embedding = model.encode(query)
    
    users = list(collection.find({"identity_embedding": {"$exists": True}}))
    if not users:
        return "⚠️ No users with embeddings found in the database."

    scored_users = []
    for user in users:
        user_embedding = np.array(user["identity_embedding"])
        score = cosine_similarity(query_embedding, user_embedding)
        scored_users.append((score, user))
    
    scored_users.sort(key=lambda x: x[0], reverse=True)
    top_k = 5
    top_users = scored_users[:top_k]
    
    context_list = []
    for score, user in top_users:
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
    response = chain.invoke({"context": context_text, "query": query})
    return response

def perform_aggregation(query, collection, llm):
    st.info("📊 Performing Aggregation Analysis...")
    
    schema_desc = """
    Collection 'users' has the following schema:
    - user_id: string (UUID)
    - location: string (Example: "NYC", "London", "Tel Aviv", "SF", "Tokyo")
    - home_location: string (Example: "NYC", "London", "Tel Aviv")
    - age: int (18-65)
    - gender: string ("male", "female")
    - duration: int (minutes spent at location)
    - timestamp: datetime
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
    
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm
    response = chain.invoke({"schema": schema_desc, "query": query})
    
    clean_response = response.strip()
    # Cleaning Logic (Borrowed from Evaluation Fixes)
    if "```json" in clean_response:
        import re
        match = re.search(r"```json(.*?)```", clean_response, re.DOTALL)
        if match: clean_response = match.group(1).strip()
    elif "```" in clean_response:
         import re
         match = re.search(r"```(.*?)```", clean_response, re.DOTALL)
         if match: clean_response = match.group(1).strip()
    
    if "=" in clean_response: clean_response = clean_response.split("=", 1)[1].strip()
    if clean_response.endswith(";"): clean_response = clean_response[:-1].strip()
    if not clean_response.startswith("[") and "[" in clean_response:
         clean_response = clean_response[clean_response.find("["):]

    try:
        pipeline = json.loads(clean_response)
    except json.JSONDecodeError:
        import re
        # Regex repair for unquoted keys
        clean_response = re.sub(r'(?<=[\{\s,])([a-zA-Z_$][a-zA-Z0-9_$]*)(?=\s*:)', r'"\1"', clean_response)
        try:
            pipeline = json.loads(clean_response)
        except:
             return f"⚠️ Failed to parse aggregation pipeline: {response}"

    if isinstance(pipeline, dict):
        pipeline = [pipeline]

    try:
        results = list(collection.aggregate(pipeline))
        if not results:
             return "✅ Query executed successfully but returned no results."
        
        return f"**Query Results:**\n\n```json\n{json.dumps(results, indent=2, default=str)}\n```"
    except Exception as e:
        return f"⚠️ Aggregation Failed: {e}"

# --- Main App ---

st.set_page_config(page_title="Agentic Analytics", page_icon="🤖")

st.title("🤖 Agentic Analytics Engine")
st.write("Ask questions about your user data. I can find similar users (RAG) or calculate statistics (Aggregation).")

# Initialize Resources
collection = get_db_collection()
embed_model, llm = get_models()

# Initialize Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

# Display Chat History
for msg in st.session_state.chat_history.messages:
    with st.chat_message(msg.type):
        st.write(msg.content)

# User Input
if user_input := st.chat_input("Ask a question (e.g., 'Find users like...', 'Average age in NYC')"):
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.chat_history.add_user_message(user_input)

    # Determine Logic
    rag_keywords = ["similar", "like", "find users", "who matches", "resemble"]
    response = ""
    
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            if any(k in user_input.lower() for k in rag_keywords):
                response = perform_rag(user_input, collection, embed_model, llm)
            else:
                response = perform_aggregation(user_input, collection, llm)
            
            st.write(response)
    
    st.session_state.chat_history.add_ai_message(response)