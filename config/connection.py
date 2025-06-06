"""
MongoDB connection configuration module.
This module provides functions to connect to MongoDB v7.0 and v8.0 clusters.
"""

import os
import time
from typing import Dict, Any, Optional
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# MongoDB connection strings
MONGO_V7_URI = os.getenv("MONGO_V7_URI", "TODO - MONGODB V7 CONNECTION STRING")
MONGO_V8_URI = os.getenv("MONGO_V8_URI", "TODO - MONGODB V8 CONNECTION STRING")

# Database names
MONGO_V7_DB = os.getenv("MONGO_V7_DB", "TODO - MONGODB V7 DATABASE NAME")
MONGO_V8_DB = os.getenv("MONGO_V8_DB", "TODO - MONGODB V8 DATABASE NAME")

# Connection timeouts
CONNECTION_TIMEOUT_MS = 5000
SERVER_SELECTION_TIMEOUT_MS = 5000

# Connection pool settings
MAX_POOL_SIZE = 100
MIN_POOL_SIZE = 10

# Clients cache
_clients = {}


def get_client(version: str) -> MongoClient:
    """
    Get a MongoDB client for the specified version.
    
    Args:
        version (str): MongoDB version ('v7' or 'v8')
        
    Returns:
        MongoClient: MongoDB client
        
    Raises:
        ValueError: If version is not 'v7' or 'v8'
        ConnectionFailure: If connection to MongoDB fails
    """
    if version not in ['v7', 'v8']:
        raise ValueError("Version must be 'v7' or 'v8'")
    
    # Return cached client if it exists
    if version in _clients:
        return _clients[version]
    
    # Get connection URI based on version
    uri = MONGO_V7_URI if version == 'v7' else MONGO_V8_URI
    
    # Create client with connection settings
    client = MongoClient(
        uri,
        connectTimeoutMS=CONNECTION_TIMEOUT_MS,
        serverSelectionTimeoutMS=SERVER_SELECTION_TIMEOUT_MS,
        maxPoolSize=MAX_POOL_SIZE,
        minPoolSize=MIN_POOL_SIZE
    )
    
    # Test connection
    try:
        # The ismaster command is cheap and does not require auth
        client.admin.command('ismaster')
        print(f"Successfully connected to MongoDB {version}")
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        raise ConnectionFailure(f"Failed to connect to MongoDB {version}: {e}")
    
    # Cache client
    _clients[version] = client
    return client


def get_database(version: str) -> Any:
    """
    Get a MongoDB database for the specified version.
    
    Args:
        version (str): MongoDB version ('v7' or 'v8')
        
    Returns:
        Database: MongoDB database
        
    Raises:
        ValueError: If version is not 'v7' or 'v8'
        ConnectionFailure: If connection to MongoDB fails
    """
    client = get_client(version)
    db_name = MONGO_V7_DB if version == 'v7' else MONGO_V8_DB
    return client[db_name]


def close_connections() -> None:
    """Close all MongoDB connections."""
    for version, client in _clients.items():
        print(f"Closing connection to MongoDB {version}")
        client.close()
    _clients.clear()


def with_retry(func, max_retries=3, retry_delay=1.0):
    """
    Decorator to retry a MongoDB operation on failure.
    
    Args:
        func: Function to retry
        max_retries (int): Maximum number of retries
        retry_delay (float): Delay between retries in seconds
        
    Returns:
        Function result or raises the last exception
    """
    def wrapper(*args, **kwargs):
        last_exception = None
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                last_exception = e
                if attempt < max_retries - 1:
                    print(f"Retry {attempt + 1}/{max_retries} after error: {e}")
                    time.sleep(retry_delay)
                    # Increase delay for next retry (exponential backoff)
                    retry_delay *= 2
        raise last_exception
    return wrapper