import argparse
import random
import uuid
from datetime import datetime
import sys

try:
    from pymongo import MongoClient
except ImportError:
    print("Error: pymongo not installed. Please run 'pip install -r requirements.txt'")
    sys.exit(1)

def generate_user_data(locations):
    """Generates a single user data dictionary."""
    location = random.choice(locations)
    home_location = random.choice(locations)
    # Generate a timestamp within the last 30 days for variety, or just now.
    # Requirement was generic "timestamp". Let's simply use now() for simplicity 
    # but maybe add a small random offset if needed. For now, strict 'now' is fine 
    # or maybe a random time in the last 24 hours to simulate activity.
    # Let's stick to simple efficient generation:
    timestamp = datetime.now() 
    
    return {
        "id": str(uuid.uuid4()),
        "location": location,
        "timestamp": timestamp,
        "age": random.randint(18, 65),
        "gender": random.choice(["male", "female"]),
        "home_location": home_location,
        "duration": random.randint(1, 120)  # duration in minutes
    }

import os

def main():

    mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
    db_name = os.environ.get("DB_NAME", "user_data_db")
    collection_name = os.environ.get("COLLECTION_NAME", "users")
    count = int(os.environ.get("USER_COUNT", 10))
    locations_str = os.environ.get("LOCATIONS", "NYC,London,Tel Aviv,SF,Tokyo")
    locations_list = [loc.strip() for loc in locations_str.split(",") if loc.strip()]
    dry_run = os.environ.get("DRY_RUN", "false").lower() == "true"


    print(f"Generating {count} users...")
    print(f"Locations: {locations_list}")

    data_buffer = []
    
    # Generate data
    for _ in range(count):
        data_buffer.append(generate_user_data(locations_list))

    if dry_run:
        print("\n--- Dry Run Output (First 5 records) ---")
        for doc in data_buffer[:5]:
            print(doc)
        print(f"\nTotal records generated: {len(data_buffer)}")
        print("Skipping DB insertion.")
        return

    # Database Insertion
    try:
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]
        
        # Test connection
        client.admin.command('ping')
        print("Connected to MongoDB successfully.")

        if data_buffer:
            result = collection.insert_many(data_buffer)
            print(f"Successfully inserted {len(result.inserted_ids)} documents.")
        else:
            print("No data to insert.")

    except Exception as e:
        print(f"Error connecting to MongoDB or inserting data: {e}")
        sys.exit(1)
    finally:
        if 'client' in locals():
            client.close()

if __name__ == "__main__":
    main()
