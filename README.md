# MongoDB User Analysis & RAG Agent

This project demonstrates a full data pipeline and intelligent agent system using **MongoDB**, **Docker**, **Sentence Transformers**, and **Ollama (LLM)**.

It generates synthetic user location data, calculates vector embeddings for that data, and provides an AI agent capable of answering questions via both **Aggregation Pipelines** (Analytics) and **RAG** (Similarity Search).

## Components

### 1. Data Generator (`init_data/`)
- **Purpose**: specific Python script to seed the database.
- **Function**: Generates synthetic user profiles with:
  - `user_id`, `age`, `gender`.
  - `location`, `home_location`, `duration`.
  - `timestamp`.
- **Docker**: Runs once and exits after checking/inserting 100 users.

### 2. Embedding Service (`create_embedding/`)
- **Purpose**: Vector processing.
- **Function**: 
  - Monitors the `users` collection.
  - Calculates embeddings (`all-MiniLM-L6-v2`) for locations and user identity.
  - Updates documents with `location_embedding` and `identity_embedding`.
- **Tech**: `sentence-transformers`, `pymongo`.

### 3. Agent Service (`agents/`)
- **Purpose**: Intelligent Query Interface.
- **Function**: Translates natural language questions into answers using two modes:
  - **Aggregation Mode**: Translates natural language to **MongoDB Aggregation Pipelines** for analytical queries (e.g., "Where are middle-aged people from NY hanging out?").
  - **RAG / Similarity Mode**: Uses **Cosine Similarity** to find users matching a description (e.g., "Find users similar to a 30yo male from London") and summarizes them using the LLM.
- **Tech**: `LangChain`, `Ollama`, `NumPy`.

## Prerequisites

1. **Docker & Docker Compose**: Installed and running.
2. **Ollama**: Installed on your host machine to serve the LLM.
   - [Download Ollama](https://ollama.com)
   - Run: `ollama serve`
   - Pull the model (default `llama3` or `llama3.2:1b`):
     ```bash
     ollama pull llama3.2:1b
     ```

## How to Run

1. **Clone/Open** the repository.
2. **Start the Stack**:
   ```bash
   docker-compose up --build
   ```
   This will:
   - Start MongoDB.
   - Generate 100 users.
   - Generate embeddings for those users.
   - Start the Agent to answer the default query.

3. **Check Results**:
   - The `agent-service` logs will show the answer to the default query.
   - You can see the generated pipeline or the RAG context in the logs.

## Custom Queries

You can change the query by setting the `QUERY` environment variable.

**Example 1: Analytics (Aggregation)**
```bash
QUERY="What is the average duration for users in Tel Aviv?" docker-compose up agent-service
```

**Example 2: Similarity Search (RAG)**
Use keywords like "similar", "like", "find users" to trigger RAG mode.
```bash
QUERY="Find users similar to a female from Tokyo staying for short duration" docker-compose up agent-service
```
