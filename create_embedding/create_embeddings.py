import os
import time
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

def main():
    mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
    db_name = os.environ.get("DB_NAME", "user_data_db")
    collection_name = os.environ.get("COLLECTION_NAME", "users")
    
    print("Waiting for MongoDB to fully start...")
    time.sleep(10) # Simple wait for Mongo to be up, or retry loop

    try:
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]
        
        # Test connection
        client.admin.command('ping')
        print("Connected to MongoDB successfully.")
        
        print("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded.")

        users = list(collection.find({"identity_embedding": {"$exists": False}}))
        print(f"Found {len(users)} users needing embeddings.")

        if not users:
            print("No users found to process.")
            return

        count = 0
        for user in users:
            location = user.get("location")
            home_location = user.get("home_location")
            age = user.get("age")
            gender = user.get("gender") 
            duration = user.get("duration")
            if location:
                identity_embedding = model.encode(f"{age} {gender} {duration} {home_location} {location}")
                collection.update_one(
                    {"_id": user["_id"]},
                    {"$set": {
                    "identity_embedding": identity_embedding.tolist()}}
                )
                count += 1
                if count % 10 == 0:
                    print(f"Processed {count}/{len(users)} users...")

        print(f"Finished generating embeddings for {count} users.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'client' in locals():
            client.close()

if __name__ == "__main__":
    main()
