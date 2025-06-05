"""
Data loader module for MongoDB.
This module provides functions to load generated data into MongoDB.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import concurrent.futures
from tqdm import tqdm

from pymongo import MongoClient, InsertMany
from pymongo.errors import BulkWriteError, PyMongoError

from ..config.connection import get_database, with_retry
from ..config.test_config import COLLECTIONS, DATA_GENERATION, INDEXES
from .faker_generator import generate_dataset, generate_batch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_indexes(version: str) -> None:
    """
    Create indexes for all collections.
    
    Args:
        version (str): MongoDB version ('v7' or 'v8')
    """
    db = get_database(version)
    
    for collection_name, indexes in INDEXES.items():
        collection = db[collection_name]
        
        for index_config in indexes:
            fields = index_config["fields"]
            unique = index_config.get("unique", False)
            
            index_name = "_".join([f"{field}_{direction}" for field, direction in fields])
            
            logger.info(f"Creating index {index_name} on {collection_name} for MongoDB {version}")
            
            try:
                collection.create_index(
                    fields,
                    unique=unique,
                    background=True
                )
                logger.info(f"Successfully created index {index_name}")
            except PyMongoError as e:
                logger.error(f"Failed to create index {index_name}: {e}")


@with_retry
def bulk_insert(
    version: str,
    collection_name: str,
    documents: List[Dict[str, Any]],
    ordered: bool = False
) -> Tuple[int, float]:
    """
    Perform a bulk insert operation.
    
    Args:
        version (str): MongoDB version ('v7' or 'v8')
        collection_name (str): Collection name
        documents (List[Dict[str, Any]]): Documents to insert
        ordered (bool): Whether to perform an ordered insert
        
    Returns:
        Tuple[int, float]: Number of documents inserted and time taken in seconds
    """
    if not documents:
        return 0, 0.0
    
    db = get_database(version)
    collection = db[collection_name]
    
    start_time = time.time()
    
    try:
        result = collection.insert_many(documents, ordered=ordered)
        inserted_count = len(result.inserted_ids)
    except BulkWriteError as e:
        inserted_count = e.details.get('nInserted', 0)
        logger.warning(f"Bulk write error: {e.details}")
    
    end_time = time.time()
    time_taken = end_time - start_time
    
    return inserted_count, time_taken


def load_initial_dataset(
    version: str,
    num_customers: int = DATA_GENERATION["customers"]["count"],
    batch_size: int = DATA_GENERATION["customers"]["batch_size"]
) -> Dict[str, int]:
    """
    Load the initial dataset into MongoDB.
    
    Args:
        version (str): MongoDB version ('v7' or 'v8')
        num_customers (int): Number of customers to generate
        batch_size (int): Batch size for bulk inserts
        
    Returns:
        Dict[str, int]: Dictionary with counts of inserted documents
    """
    logger.info(f"Generating and loading initial dataset for MongoDB {version}")
    
    # Create indexes first
    create_indexes(version)
    
    # Generate the complete dataset
    logger.info(f"Generating dataset with {num_customers} customers")
    dataset = generate_dataset(num_customers=num_customers)
    
    # Insert data in batches
    counts = {}
    
    for collection_name, documents in dataset.items():
        logger.info(f"Loading {len(documents)} documents into {collection_name}")
        
        total_inserted = 0
        total_time = 0.0
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            
            inserted, time_taken = bulk_insert(
                version=version,
                collection_name=collection_name,
                documents=batch
            )
            
            total_inserted += inserted
            total_time += time_taken
            
            logger.info(
                f"Inserted {inserted} {collection_name} "
                f"({i+inserted}/{len(documents)}, {time_taken:.2f}s)"
            )
        
        counts[collection_name] = total_inserted
        
        if total_inserted > 0:
            logger.info(
                f"Completed loading {total_inserted} {collection_name} "
                f"in {total_time:.2f}s ({total_inserted/total_time:.2f} docs/s)"
            )
    
    return counts


def parallel_bulk_insert(
    version: str,
    collection_name: str,
    batches: List[List[Dict[str, Any]]],
    max_workers: int = 4
) -> Tuple[int, float]:
    """
    Perform parallel bulk insert operations.
    
    Args:
        version (str): MongoDB version ('v7' or 'v8')
        collection_name (str): Collection name
        batches (List[List[Dict[str, Any]]]): Batches of documents to insert
        max_workers (int): Maximum number of worker threads
        
    Returns:
        Tuple[int, float]: Total number of documents inserted and total time taken
    """
    total_inserted = 0
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batches for processing
        future_to_batch = {
            executor.submit(bulk_insert, version, collection_name, batch): i
            for i, batch in enumerate(batches)
        }
        
        # Process results as they complete
        for future in tqdm(
            concurrent.futures.as_completed(future_to_batch),
            total=len(batches),
            desc=f"Inserting {collection_name}"
        ):
            batch_index = future_to_batch[future]
            try:
                inserted, _ = future.result()
                total_inserted += inserted
            except Exception as e:
                logger.error(f"Batch {batch_index} generated an exception: {e}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return total_inserted, total_time


def generate_and_load_batch(
    version: str,
    collection_name: str,
    batch_size: int,
    customer_ids: Optional[List[str]] = None,
    account_ids: Optional[List[str]] = None
) -> Tuple[int, float]:
    """
    Generate and load a batch of documents for a specific collection.
    
    Args:
        version (str): MongoDB version ('v7' or 'v8')
        collection_name (str): Collection name
        batch_size (int): Number of documents to generate and insert
        customer_ids (List[str], optional): List of customer IDs for generating related documents
        account_ids (List[str], optional): List of account IDs for generating transactions
        
    Returns:
        Tuple[int, float]: Number of documents inserted and time taken
    """
    # Generate batch
    batch = generate_batch(
        collection_name=collection_name,
        batch_size=batch_size,
        customer_ids=customer_ids,
        account_ids=account_ids
    )
    
    # Insert batch
    return bulk_insert(
        version=version,
        collection_name=collection_name,
        documents=batch
    )


def get_sample_ids(version: str, collection_name: str, limit: int = 1000) -> List[str]:
    """
    Get a sample of IDs from a collection.
    
    Args:
        version (str): MongoDB version ('v7' or 'v8')
        collection_name (str): Collection name
        limit (int): Maximum number of IDs to retrieve
        
    Returns:
        List[str]: List of IDs
    """
    db = get_database(version)
    collection = db[collection_name]
    
    # Determine ID field based on collection name
    id_field = f"{collection_name[:-1]}_id"  # Remove trailing 's' from collection name
    
    # Get sample documents
    pipeline = [
        {"$project": {"_id": 0, id_field: 1}},
        {"$limit": limit}
    ]
    
    result = list(collection.aggregate(pipeline))
    
    # Extract IDs
    return [doc[id_field] for doc in result if id_field in doc]


def clear_collections(version: str) -> None:
    """
    Clear all collections in the database.
    
    Args:
        version (str): MongoDB version ('v7' or 'v8')
    """
    db = get_database(version)
    
    for collection_name in COLLECTIONS.values():
        logger.info(f"Clearing collection {collection_name} for MongoDB {version}")
        db[collection_name].delete_many({})


def get_collection_counts(version: str) -> Dict[str, int]:
    """
    Get document counts for all collections.
    
    Args:
        version (str): MongoDB version ('v7' or 'v8')
        
    Returns:
        Dict[str, int]: Dictionary with collection counts
    """
    db = get_database(version)
    counts = {}
    
    for collection_name in COLLECTIONS.values():
        counts[collection_name] = db[collection_name].count_documents({})
    
    return counts