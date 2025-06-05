"""
Read operation performance tests for MongoDB.
This module provides functions to test read operation performance.
"""

import os
import sys
# Add the current directory to sys.path to fix import issues
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import logging
import time
import random
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import concurrent.futures

from pymongo import MongoClient
from bson import ObjectId

from config.connection import get_database, close_connections
from config.test_config import TEST_PARAMETERS, COLLECTIONS
from data_generation import get_sample_ids
from utils import (
    PerformanceMetrics,
    run_with_metrics,
    compare_results
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_single_document_reads(
    version: str,
    collection_name: str,
    num_reads: int = TEST_PARAMETERS["read"]["single_reads_count"],
    iterations: int = TEST_PARAMETERS["read"]["iterations"]
) -> Tuple[Dict[str, Any], PerformanceMetrics]:
    """
    Test single document read performance.
    
    Args:
        version (str): MongoDB version ('v7' or 'v8')
        collection_name (str): Collection name
        num_reads (int): Number of reads per iteration
        iterations (int): Number of iterations
        
    Returns:
        Tuple[Dict[str, Any], PerformanceMetrics]: Test results and performance metrics
    """
    test_name = f"single_document_reads_{collection_name}"
    logger.info(f"Running {test_name} test for MongoDB {version}")
    
    # Get database and collection
    db = get_database(version)
    collection = db[collection_name]
    
    # Get sample document IDs
    id_field = f"{collection_name[:-1]}_id"  # Remove trailing 's' from collection name
    sample_ids = []
    
    # Get a sample of document IDs
    cursor = collection.find({}, {id_field: 1}).limit(num_reads)
    for doc in cursor:
        if id_field in doc:
            sample_ids.append(doc[id_field])
    
    if not sample_ids:
        logger.error(f"No {id_field} values found in {collection_name}. Please load data first.")
        return {
            "test_name": test_name,
            "version": version,
            "error": f"No {id_field} values found"
        }, PerformanceMetrics(test_name, version)
    
    # Initialize metrics
    metrics = PerformanceMetrics(test_name, version)
    metrics.start()
    
    results = {
        "test_name": test_name,
        "version": version,
        "collection_name": collection_name,
        "num_reads": num_reads,
        "iterations": iterations,
        "total_reads": 0,
        "total_time": 0.0,
        "avg_throughput": 0.0
    }
    
    try:
        # Run test iterations
        for i in range(iterations):
            logger.info(f"Iteration {i+1}/{iterations}")
            
            iteration_start_time = time.time()
            reads_completed = 0
            
            # Perform reads
            for _ in range(num_reads):
                # Select a random ID
                doc_id = random.choice(sample_ids)
                
                # Perform read
                start_time = time.time()
                doc = collection.find_one({id_field: doc_id})
                end_time = time.time()
                
                # Record metrics
                if doc:
                    reads_completed += 1
                    response_time = end_time - start_time
                    metrics.record_response_time(response_time)
                    metrics.record_operation("find", 1)
            
            iteration_end_time = time.time()
            iteration_time = iteration_end_time - iteration_start_time
            
            # Update results
            results["total_reads"] += reads_completed
            results["total_time"] += iteration_time
            
            logger.info(
                f"Completed {reads_completed} reads in {iteration_time:.2f}s "
                f"({reads_completed/iteration_time:.2f} reads/s)"
            )
        
        # Calculate average throughput
        if results["total_time"] > 0:
            results["avg_throughput"] = results["total_reads"] / results["total_time"]
        
        return results, metrics
    
    finally:
        # Stop metrics collection
        metrics.stop()


def test_filtered_reads(
    version: str,
    collection_name: str,
    num_reads: int = TEST_PARAMETERS["read"]["filtered_reads_count"],
    iterations: int = TEST_PARAMETERS["read"]["iterations"]
) -> Tuple[Dict[str, Any], PerformanceMetrics]:
    """
    Test filtered read performance.
    
    Args:
        version (str): MongoDB version ('v7' or 'v8')
        collection_name (str): Collection name
        num_reads (int): Number of reads per iteration
        iterations (int): Number of iterations
        
    Returns:
        Tuple[Dict[str, Any], PerformanceMetrics]: Test results and performance metrics
    """
    test_name = f"filtered_reads_{collection_name}"
    logger.info(f"Running {test_name} test for MongoDB {version}")
    
    # Get database and collection
    db = get_database(version)
    collection = db[collection_name]
    
    # Define filter criteria based on collection
    filter_criteria = []
    
    if collection_name == "customers":
        # Get a sample of credit scores
        cursor = collection.find({}, {"credit_score": 1}).limit(100)
        credit_scores = [doc.get("credit_score") for doc in cursor if "credit_score" in doc]
        
        if credit_scores:
            for _ in range(num_reads):
                min_score = random.randint(300, 700)
                max_score = min_score + random.randint(50, 150)
                filter_criteria.append({"credit_score": {"$gte": min_score, "$lte": max_score}})
    
    elif collection_name == "accounts":
        # Get a sample of customer IDs
        customer_ids = get_sample_ids(version, "customers", limit=100)
        
        if customer_ids:
            for _ in range(num_reads):
                customer_id = random.choice(customer_ids)
                filter_criteria.append({"customer_id": customer_id})
    
    elif collection_name == "transactions":
        # Get a sample of account IDs
        account_ids = get_sample_ids(version, "accounts", limit=100)
        
        if account_ids:
            for _ in range(num_reads):
                account_id = random.choice(account_ids)
                filter_criteria.append({"account_id": account_id})
    
    elif collection_name == "loans":
        # Get a sample of customer IDs
        customer_ids = get_sample_ids(version, "customers", limit=100)
        
        if customer_ids:
            for _ in range(num_reads):
                customer_id = random.choice(customer_ids)
                filter_criteria.append({"customer_id": customer_id})
    
    if not filter_criteria:
        logger.error(f"Could not generate filter criteria for {collection_name}. Please load data first.")
        return {
            "test_name": test_name,
            "version": version,
            "error": "Could not generate filter criteria"
        }, PerformanceMetrics(test_name, version)
    
    # Initialize metrics
    metrics = PerformanceMetrics(test_name, version)
    metrics.start()
    
    results = {
        "test_name": test_name,
        "version": version,
        "collection_name": collection_name,
        "num_reads": num_reads,
        "iterations": iterations,
        "total_reads": 0,
        "total_time": 0.0,
        "avg_throughput": 0.0
    }
    
    try:
        # Run test iterations
        for i in range(iterations):
            logger.info(f"Iteration {i+1}/{iterations}")
            
            iteration_start_time = time.time()
            reads_completed = 0
            
            # Perform reads
            for filter_query in filter_criteria:
                # Perform read
                start_time = time.time()
                cursor = collection.find(filter_query).limit(10)
                docs = list(cursor)
                end_time = time.time()
                
                # Record metrics
                reads_completed += 1
                response_time = end_time - start_time
                metrics.record_response_time(response_time)
                metrics.record_operation("find", 1)
            
            iteration_end_time = time.time()
            iteration_time = iteration_end_time - iteration_start_time
            
            # Update results
            results["total_reads"] += reads_completed
            results["total_time"] += iteration_time
            
            logger.info(
                f"Completed {reads_completed} filtered reads in {iteration_time:.2f}s "
                f"({reads_completed/iteration_time:.2f} reads/s)"
            )
        
        # Calculate average throughput
        if results["total_time"] > 0:
            results["avg_throughput"] = results["total_reads"] / results["total_time"]
        
        return results, metrics
    
    finally:
        # Stop metrics collection
        metrics.stop()


def test_complex_reads(
    version: str,
    collection_name: str,
    num_reads: int = TEST_PARAMETERS["read"]["complex_reads_count"],
    iterations: int = TEST_PARAMETERS["read"]["iterations"]
) -> Tuple[Dict[str, Any], PerformanceMetrics]:
    """
    Test complex read performance.
    
    Args:
        version (str): MongoDB version ('v7' or 'v8')
        collection_name (str): Collection name
        num_reads (int): Number of reads per iteration
        iterations (int): Number of iterations
        
    Returns:
        Tuple[Dict[str, Any], PerformanceMetrics]: Test results and performance metrics
    """
    test_name = f"complex_reads_{collection_name}"
    logger.info(f"Running {test_name} test for MongoDB {version}")
    
    # Get database and collection
    db = get_database(version)
    collection = db[collection_name]
    
    # Define complex filter criteria based on collection
    filter_criteria = []
    
    if collection_name == "customers":
        # Complex customer queries
        for _ in range(num_reads):
            min_score = random.randint(300, 700)
            max_score = min_score + random.randint(50, 150)
            
            filter_criteria.append({
                "$and": [
                    {"credit_score": {"$gte": min_score, "$lte": max_score}},
                    {"address.country": "US"},
                    {"created_at": {"$gte": datetime.now() - timedelta(days=365)}}
                ]
            })
    
    elif collection_name == "accounts":
        # Complex account queries
        account_types = ["CHECKING", "SAVINGS"]
        statuses = ["ACTIVE", "INACTIVE"]
        
        for _ in range(num_reads):
            min_balance = random.randint(1000, 10000)
            max_balance = min_balance + random.randint(5000, 50000)
            
            filter_criteria.append({
                "$and": [
                    {"account_type": {"$in": account_types}},
                    {"status": {"$in": statuses}},
                    {"balance": {"$gte": min_balance, "$lte": max_balance}},
                    {"created_at": {"$gte": datetime.now() - timedelta(days=365)}}
                ]
            })
    
    elif collection_name == "transactions":
        # Complex transaction queries
        transaction_types = ["DEPOSIT", "WITHDRAWAL", "PAYMENT"]
        
        for _ in range(num_reads):
            min_amount = random.randint(10, 500)
            max_amount = min_amount + random.randint(100, 1000)
            
            start_date = datetime.now() - timedelta(days=random.randint(30, 365))
            end_date = start_date + timedelta(days=random.randint(7, 30))
            
            filter_criteria.append({
                "$and": [
                    {"transaction_type": {"$in": transaction_types}},
                    {"amount": {"$gte": min_amount, "$lte": max_amount}},
                    {"timestamp": {"$gte": start_date, "$lte": end_date}},
                    {"status": "COMPLETED"}
                ]
            })
    
    elif collection_name == "loans":
        # Complex loan queries
        loan_types = ["PERSONAL", "AUTO", "MORTGAGE"]
        
        for _ in range(num_reads):
            min_amount = random.randint(5000, 50000)
            max_amount = min_amount + random.randint(10000, 100000)
            
            filter_criteria.append({
                "$and": [
                    {"loan_type": {"$in": loan_types}},
                    {"amount": {"$gte": min_amount, "$lte": max_amount}},
                    {"status": {"$in": ["ACTIVE", "APPROVED"]}},
                    {"interest_rate": {"$gte": 3.0, "$lte": 8.0}}
                ]
            })
    
    if not filter_criteria:
        logger.error(f"Could not generate complex filter criteria for {collection_name}. Please load data first.")
        return {
            "test_name": test_name,
            "version": version,
            "error": "Could not generate complex filter criteria"
        }, PerformanceMetrics(test_name, version)
    
    # Initialize metrics
    metrics = PerformanceMetrics(test_name, version)
    metrics.start()
    
    results = {
        "test_name": test_name,
        "version": version,
        "collection_name": collection_name,
        "num_reads": num_reads,
        "iterations": iterations,
        "total_reads": 0,
        "total_time": 0.0,
        "avg_throughput": 0.0
    }
    
    try:
        # Run test iterations
        for i in range(iterations):
            logger.info(f"Iteration {i+1}/{iterations}")
            
            iteration_start_time = time.time()
            reads_completed = 0
            
            # Perform reads
            for filter_query in filter_criteria:
                # Perform read
                start_time = time.time()
                cursor = collection.find(filter_query).limit(20)
                docs = list(cursor)
                end_time = time.time()
                
                # Record metrics
                reads_completed += 1
                response_time = end_time - start_time
                metrics.record_response_time(response_time)
                metrics.record_operation("find", 1)
            
            iteration_end_time = time.time()
            iteration_time = iteration_end_time - iteration_start_time
            
            # Update results
            results["total_reads"] += reads_completed
            results["total_time"] += iteration_time
            
            logger.info(
                f"Completed {reads_completed} complex reads in {iteration_time:.2f}s "
                f"({reads_completed/iteration_time:.2f} reads/s)"
            )
        
        # Calculate average throughput
        if results["total_time"] > 0:
            results["avg_throughput"] = results["total_reads"] / results["total_time"]
        
        return results, metrics
    
    finally:
        # Stop metrics collection
        metrics.stop()


def test_paginated_reads(
    version: str,
    collection_name: str,
    page_sizes: List[int] = TEST_PARAMETERS["read"]["pagination"]["page_sizes"],
    pages_count: int = TEST_PARAMETERS["read"]["pagination"]["pages_count"],
    iterations: int = TEST_PARAMETERS["read"]["iterations"]
) -> Tuple[Dict[str, Any], PerformanceMetrics]:
    """
    Test paginated read performance.
    
    Args:
        version (str): MongoDB version ('v7' or 'v8')
        collection_name (str): Collection name
        page_sizes (List[int]): List of page sizes to test
        pages_count (int): Number of pages to read
        iterations (int): Number of iterations
        
    Returns:
        Tuple[Dict[str, Any], PerformanceMetrics]: Test results and performance metrics
    """
    test_name = f"paginated_reads_{collection_name}"
    logger.info(f"Running {test_name} test for MongoDB {version}")
    
    # Get database and collection
    db = get_database(version)
    collection = db[collection_name]
    
    # Initialize metrics
    metrics = PerformanceMetrics(test_name, version)
    metrics.start()
    
    results = {
        "test_name": test_name,
        "version": version,
        "collection_name": collection_name,
        "page_sizes": page_sizes,
        "pages_count": pages_count,
        "iterations": iterations,
        "total_reads": 0,
        "total_time": 0.0,
        "avg_throughput": 0.0,
        "page_size_results": {}
    }
    
    try:
        # Test each page size
        for page_size in page_sizes:
            logger.info(f"Testing page size: {page_size}")
            
            page_size_results = {
                "page_size": page_size,
                "total_reads": 0,
                "total_time": 0.0,
                "avg_throughput": 0.0
            }
            
            # Run test iterations
            for i in range(iterations):
                logger.info(f"Iteration {i+1}/{iterations}")
                
                iteration_start_time = time.time()
                reads_completed = 0
                
                # Perform paginated reads
                for page in range(pages_count):
                    skip = page * page_size
                    
                    # Perform read
                    start_time = time.time()
                    cursor = collection.find().skip(skip).limit(page_size)
                    docs = list(cursor)
                    end_time = time.time()
                    
                    # Record metrics
                    reads_completed += 1
                    response_time = end_time - start_time
                    metrics.record_response_time(response_time)
                    metrics.record_operation("find", 1)
                
                iteration_end_time = time.time()
                iteration_time = iteration_end_time - iteration_start_time
                
                # Update results
                page_size_results["total_reads"] += reads_completed
                page_size_results["total_time"] += iteration_time
                
                logger.info(
                    f"Completed {reads_completed} paginated reads (page size: {page_size}) "
                    f"in {iteration_time:.2f}s ({reads_completed/iteration_time:.2f} reads/s)"
                )
            
            # Calculate average throughput for this page size
            if page_size_results["total_time"] > 0:
                page_size_results["avg_throughput"] = (
                    page_size_results["total_reads"] / page_size_results["total_time"]
                )
            
            # Update overall results
            results["total_reads"] += page_size_results["total_reads"]
            results["total_time"] += page_size_results["total_time"]
            results["page_size_results"][str(page_size)] = page_size_results
        
        # Calculate overall average throughput
        if results["total_time"] > 0:
            results["avg_throughput"] = results["total_reads"] / results["total_time"]
        
        return results, metrics
    
    finally:
        # Stop metrics collection
        metrics.stop()


def run_read_tests(version: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, PerformanceMetrics]]:
    """
    Run all read tests for a specific MongoDB version.
    
    Args:
        version (str): MongoDB version ('v7' or 'v8')
        
    Returns:
        Tuple[Dict[str, Dict[str, Any]], Dict[str, PerformanceMetrics]]: 
            Dictionary with test results and dictionary with performance metrics
    """
    results = {}
    metrics_dict = {}
    
    # Test single document reads for each collection
    for collection_name in COLLECTIONS.values():
        test_name = f"single_document_reads_{collection_name}"
        test_result, metrics = test_single_document_reads(version, collection_name)
        results[test_name] = test_result
        metrics_dict[test_name] = metrics
    
    # Test filtered reads for each collection
    for collection_name in COLLECTIONS.values():
        test_name = f"filtered_reads_{collection_name}"
        test_result, metrics = test_filtered_reads(version, collection_name)
        results[test_name] = test_result
        metrics_dict[test_name] = metrics
    
    # Test complex reads for each collection
    for collection_name in COLLECTIONS.values():
        test_name = f"complex_reads_{collection_name}"
        test_result, metrics = test_complex_reads(version, collection_name)
        results[test_name] = test_result
        metrics_dict[test_name] = metrics
    
    # Test paginated reads for each collection
    for collection_name in COLLECTIONS.values():
        test_name = f"paginated_reads_{collection_name}"
        test_result, metrics = test_paginated_reads(version, collection_name)
        results[test_name] = test_result
        metrics_dict[test_name] = metrics
    
    return results, metrics_dict


def compare_read_performance() -> Dict[str, Any]:
    """
    Compare read performance between MongoDB v7.0 and v8.0.
    
    Returns:
        Dict[str, Any]: Comparison results
    """
    # Run tests for MongoDB v7.0
    logger.info("Running read tests for MongoDB v7.0")
    v7_results, v7_metrics = run_read_tests('v7')
    
    # Run tests for MongoDB v8.0
    logger.info("Running read tests for MongoDB v8.0")
    v8_results, v8_metrics = run_read_tests('v8')
    
    # Compare results
    comparison = {}
    
    for test_name in v7_results:
        if test_name in v8_results:
            v7_result = v7_results[test_name]
            v8_result = v8_results[test_name]
            
            # Calculate improvement
            if v7_result.get("avg_throughput", 0) > 0:
                throughput_improvement = (
                    (v8_result.get("avg_throughput", 0) - v7_result.get("avg_throughput", 0)) /
                    v7_result.get("avg_throughput", 0)
                ) * 100
            else:
                throughput_improvement = 0
            
            comparison[test_name] = {
                "v7_throughput": v7_result.get("avg_throughput", 0),
                "v8_throughput": v8_result.get("avg_throughput", 0),
                "throughput_improvement_pct": throughput_improvement
            }
            
            # Generate report
            if test_name in v7_metrics and test_name in v8_metrics:
                report_files = compare_results(
                    test_name=test_name,
                    v7_metrics=v7_metrics[test_name],
                    v8_metrics=v8_metrics[test_name]
                )
                comparison[test_name]["report_files"] = report_files
    
    return comparison


if __name__ == "__main__":
    # Run comparison
    comparison = compare_read_performance()
    
    # Print summary
    print("\nRead Performance Comparison (MongoDB v8.0 vs v7.0):")
    print("=" * 80)
    print(f"{'Test':<40} {'v7.0 (ops/s)':<15} {'v8.0 (ops/s)':<15} {'Improvement':<15}")
    print("-" * 80)
    
    for test_name, result in comparison.items():
        print(
            f"{test_name:<40} "
            f"{result['v7_throughput']:<15.2f} "
            f"{result['v8_throughput']:<15.2f} "
            f"{result['throughput_improvement_pct']:<15.2f}%"
        )
    
    print("=" * 80)
    
    # Close connections
    close_connections()