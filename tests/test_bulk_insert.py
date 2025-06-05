"""
Bulk insert performance tests for MongoDB.
This module provides functions to test bulk insert performance.
"""

import logging
import time
from typing import Dict, List, Any, Tuple
import random
from concurrent.futures import ThreadPoolExecutor

from config.connection import get_database, close_connections
from config.test_config import TEST_PARAMETERS, COLLECTIONS
from data_generation import (
    generate_batch,
    get_sample_ids,
    bulk_insert,
    parallel_bulk_insert
)
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


def test_bulk_insert_customers(
    version: str,
    batch_size: int,
    iterations: int = TEST_PARAMETERS["bulk_insert"]["iterations"]
) -> Tuple[Dict[str, Any], PerformanceMetrics]:
    """
    Test bulk insert performance for customers.
    
    Args:
        version (str): MongoDB version ('v7' or 'v8')
        batch_size (int): Batch size for bulk inserts
        iterations (int): Number of iterations
        
    Returns:
        Tuple[Dict[str, Any], PerformanceMetrics]: Test results and performance metrics
    """
    test_name = f"bulk_insert_customers_{batch_size}"
    logger.info(f"Running {test_name} test for MongoDB {version}")
    
    # Initialize metrics
    metrics = PerformanceMetrics(test_name, version)
    metrics.start()
    
    results = {
        "test_name": test_name,
        "version": version,
        "batch_size": batch_size,
        "iterations": iterations,
        "total_documents": 0,
        "total_time": 0.0,
        "avg_throughput": 0.0
    }
    
    try:
        # Run test iterations
        for i in range(iterations):
            logger.info(f"Iteration {i+1}/{iterations}")
            
            # Generate and insert batch
            batch = generate_batch(
                collection_name="customers",
                batch_size=batch_size
            )
            
            start_time = time.time()
            inserted_count, _ = bulk_insert(
                version=version,
                collection_name="customers",
                documents=batch
            )
            end_time = time.time()
            
            # Record metrics
            iteration_time = end_time - start_time
            metrics.record_response_time(iteration_time)
            metrics.record_operation("insert", inserted_count)
            
            # Update results
            results["total_documents"] += inserted_count
            results["total_time"] += iteration_time
            
            logger.info(
                f"Inserted {inserted_count} customers in {iteration_time:.2f}s "
                f"({inserted_count/iteration_time:.2f} docs/s)"
            )
        
        # Calculate average throughput
        if results["total_time"] > 0:
            results["avg_throughput"] = results["total_documents"] / results["total_time"]
        
        return results, metrics
    
    finally:
        # Stop metrics collection
        metrics.stop()


def test_bulk_insert_accounts(
    version: str,
    batch_size: int,
    iterations: int = TEST_PARAMETERS["bulk_insert"]["iterations"]
) -> Tuple[Dict[str, Any], PerformanceMetrics]:
    """
    Test bulk insert performance for accounts.
    
    Args:
        version (str): MongoDB version ('v7' or 'v8')
        batch_size (int): Batch size for bulk inserts
        iterations (int): Number of iterations
        
    Returns:
        Tuple[Dict[str, Any], PerformanceMetrics]: Test results and performance metrics
    """
    test_name = f"bulk_insert_accounts_{batch_size}"
    logger.info(f"Running {test_name} test for MongoDB {version}")
    
    # Get sample customer IDs
    customer_ids = get_sample_ids(version, "customers", limit=1000)
    
    if not customer_ids:
        logger.error("No customer IDs found. Please load customer data first.")
        return {
            "test_name": test_name,
            "version": version,
            "error": "No customer IDs found"
        }, PerformanceMetrics(test_name, version)
    
    # Initialize metrics
    metrics = PerformanceMetrics(test_name, version)
    metrics.start()
    
    results = {
        "test_name": test_name,
        "version": version,
        "batch_size": batch_size,
        "iterations": iterations,
        "total_documents": 0,
        "total_time": 0.0,
        "avg_throughput": 0.0
    }
    
    try:
        # Run test iterations
        for i in range(iterations):
            logger.info(f"Iteration {i+1}/{iterations}")
            
            # Generate and insert batch
            batch = generate_batch(
                collection_name="accounts",
                batch_size=batch_size,
                customer_ids=customer_ids
            )
            
            start_time = time.time()
            inserted_count, _ = bulk_insert(
                version=version,
                collection_name="accounts",
                documents=batch
            )
            end_time = time.time()
            
            # Record metrics
            iteration_time = end_time - start_time
            metrics.record_response_time(iteration_time)
            metrics.record_operation("insert", inserted_count)
            
            # Update results
            results["total_documents"] += inserted_count
            results["total_time"] += iteration_time
            
            logger.info(
                f"Inserted {inserted_count} accounts in {iteration_time:.2f}s "
                f"({inserted_count/iteration_time:.2f} docs/s)"
            )
        
        # Calculate average throughput
        if results["total_time"] > 0:
            results["avg_throughput"] = results["total_documents"] / results["total_time"]
        
        return results, metrics
    
    finally:
        # Stop metrics collection
        metrics.stop()


def test_bulk_insert_transactions(
    version: str,
    batch_size: int,
    iterations: int = TEST_PARAMETERS["bulk_insert"]["iterations"]
) -> Tuple[Dict[str, Any], PerformanceMetrics]:
    """
    Test bulk insert performance for transactions.
    
    Args:
        version (str): MongoDB version ('v7' or 'v8')
        batch_size (int): Batch size for bulk inserts
        iterations (int): Number of iterations
        
    Returns:
        Tuple[Dict[str, Any], PerformanceMetrics]: Test results and performance metrics
    """
    test_name = f"bulk_insert_transactions_{batch_size}"
    logger.info(f"Running {test_name} test for MongoDB {version}")
    
    # Get sample account IDs
    account_ids = get_sample_ids(version, "accounts", limit=1000)
    
    if not account_ids:
        logger.error("No account IDs found. Please load account data first.")
        return {
            "test_name": test_name,
            "version": version,
            "error": "No account IDs found"
        }, PerformanceMetrics(test_name, version)
    
    # Initialize metrics
    metrics = PerformanceMetrics(test_name, version)
    metrics.start()
    
    results = {
        "test_name": test_name,
        "version": version,
        "batch_size": batch_size,
        "iterations": iterations,
        "total_documents": 0,
        "total_time": 0.0,
        "avg_throughput": 0.0
    }
    
    try:
        # Run test iterations
        for i in range(iterations):
            logger.info(f"Iteration {i+1}/{iterations}")
            
            # Generate and insert batch
            batch = generate_batch(
                collection_name="transactions",
                batch_size=batch_size,
                account_ids=account_ids
            )
            
            start_time = time.time()
            inserted_count, _ = bulk_insert(
                version=version,
                collection_name="transactions",
                documents=batch
            )
            end_time = time.time()
            
            # Record metrics
            iteration_time = end_time - start_time
            metrics.record_response_time(iteration_time)
            metrics.record_operation("insert", inserted_count)
            
            # Update results
            results["total_documents"] += inserted_count
            results["total_time"] += iteration_time
            
            logger.info(
                f"Inserted {inserted_count} transactions in {iteration_time:.2f}s "
                f"({inserted_count/iteration_time:.2f} docs/s)"
            )
        
        # Calculate average throughput
        if results["total_time"] > 0:
            results["avg_throughput"] = results["total_documents"] / results["total_time"]
        
        return results, metrics
    
    finally:
        # Stop metrics collection
        metrics.stop()


def test_bulk_insert_loans(
    version: str,
    batch_size: int,
    iterations: int = TEST_PARAMETERS["bulk_insert"]["iterations"]
) -> Tuple[Dict[str, Any], PerformanceMetrics]:
    """
    Test bulk insert performance for loans.
    
    Args:
        version (str): MongoDB version ('v7' or 'v8')
        batch_size (int): Batch size for bulk inserts
        iterations (int): Number of iterations
        
    Returns:
        Tuple[Dict[str, Any], PerformanceMetrics]: Test results and performance metrics
    """
    test_name = f"bulk_insert_loans_{batch_size}"
    logger.info(f"Running {test_name} test for MongoDB {version}")
    
    # Get sample customer IDs
    customer_ids = get_sample_ids(version, "customers", limit=1000)
    
    if not customer_ids:
        logger.error("No customer IDs found. Please load customer data first.")
        return {
            "test_name": test_name,
            "version": version,
            "error": "No customer IDs found"
        }, PerformanceMetrics(test_name, version)
    
    # Initialize metrics
    metrics = PerformanceMetrics(test_name, version)
    metrics.start()
    
    results = {
        "test_name": test_name,
        "version": version,
        "batch_size": batch_size,
        "iterations": iterations,
        "total_documents": 0,
        "total_time": 0.0,
        "avg_throughput": 0.0
    }
    
    try:
        # Run test iterations
        for i in range(iterations):
            logger.info(f"Iteration {i+1}/{iterations}")
            
            # Generate and insert batch
            batch = generate_batch(
                collection_name="loans",
                batch_size=batch_size,
                customer_ids=customer_ids
            )
            
            start_time = time.time()
            inserted_count, _ = bulk_insert(
                version=version,
                collection_name="loans",
                documents=batch
            )
            end_time = time.time()
            
            # Record metrics
            iteration_time = end_time - start_time
            metrics.record_response_time(iteration_time)
            metrics.record_operation("insert", inserted_count)
            
            # Update results
            results["total_documents"] += inserted_count
            results["total_time"] += iteration_time
            
            logger.info(
                f"Inserted {inserted_count} loans in {iteration_time:.2f}s "
                f"({inserted_count/iteration_time:.2f} docs/s)"
            )
        
        # Calculate average throughput
        if results["total_time"] > 0:
            results["avg_throughput"] = results["total_documents"] / results["total_time"]
        
        return results, metrics
    
    finally:
        # Stop metrics collection
        metrics.stop()


def test_parallel_bulk_insert(
    version: str,
    collection_name: str,
    batch_size: int,
    num_batches: int = 10,
    max_workers: int = 4
) -> Tuple[Dict[str, Any], PerformanceMetrics]:
    """
    Test parallel bulk insert performance.
    
    Args:
        version (str): MongoDB version ('v7' or 'v8')
        collection_name (str): Collection name
        batch_size (int): Batch size for bulk inserts
        num_batches (int): Number of batches to insert
        max_workers (int): Maximum number of worker threads
        
    Returns:
        Tuple[Dict[str, Any], PerformanceMetrics]: Test results and performance metrics
    """
    test_name = f"parallel_bulk_insert_{collection_name}_{batch_size}x{num_batches}"
    logger.info(f"Running {test_name} test for MongoDB {version}")
    
    # Get sample IDs for related collections
    customer_ids = None
    account_ids = None
    
    if collection_name in ["accounts", "loans"]:
        customer_ids = get_sample_ids(version, "customers", limit=1000)
        if not customer_ids:
            logger.error("No customer IDs found. Please load customer data first.")
            return {
                "test_name": test_name,
                "version": version,
                "error": "No customer IDs found"
            }, PerformanceMetrics(test_name, version)
    
    if collection_name == "transactions":
        account_ids = get_sample_ids(version, "accounts", limit=1000)
        if not account_ids:
            logger.error("No account IDs found. Please load account data first.")
            return {
                "test_name": test_name,
                "version": version,
                "error": "No account IDs found"
            }, PerformanceMetrics(test_name, version)
    
    # Initialize metrics
    metrics = PerformanceMetrics(test_name, version)
    metrics.start()
    
    results = {
        "test_name": test_name,
        "version": version,
        "collection_name": collection_name,
        "batch_size": batch_size,
        "num_batches": num_batches,
        "max_workers": max_workers,
        "total_documents": 0,
        "total_time": 0.0,
        "avg_throughput": 0.0
    }
    
    try:
        # Generate batches
        batches = []
        for _ in range(num_batches):
            batch = generate_batch(
                collection_name=collection_name,
                batch_size=batch_size,
                customer_ids=customer_ids,
                account_ids=account_ids
            )
            batches.append(batch)
        
        # Insert batches in parallel
        start_time = time.time()
        inserted_count, _ = parallel_bulk_insert(
            version=version,
            collection_name=collection_name,
            batches=batches,
            max_workers=max_workers
        )
        end_time = time.time()
        
        # Record metrics
        total_time = end_time - start_time
        metrics.record_response_time(total_time)
        metrics.record_operation("insert", inserted_count)
        
        # Update results
        results["total_documents"] = inserted_count
        results["total_time"] = total_time
        
        # Calculate throughput
        if total_time > 0:
            results["avg_throughput"] = inserted_count / total_time
        
        logger.info(
            f"Inserted {inserted_count} {collection_name} in {total_time:.2f}s "
            f"({inserted_count/total_time:.2f} docs/s)"
        )
        
        return results, metrics
    
    finally:
        # Stop metrics collection
        metrics.stop()


def run_bulk_insert_tests(version: str) -> Dict[str, Dict[str, Any]]:
    """
    Run all bulk insert tests for a specific MongoDB version.
    
    Args:
        version (str): MongoDB version ('v7' or 'v8')
        
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary with test results
    """
    results = {}
    metrics_dict = {}
    
    # Test bulk insert for customers with different batch sizes
    for batch_size in TEST_PARAMETERS["bulk_insert"]["batch_sizes"]["customers"]:
        test_name = f"bulk_insert_customers_{batch_size}"
        test_result, metrics = test_bulk_insert_customers(version, batch_size)
        results[test_name] = test_result
        metrics_dict[test_name] = metrics
    
    # Test bulk insert for accounts with different batch sizes
    for batch_size in TEST_PARAMETERS["bulk_insert"]["batch_sizes"]["accounts"]:
        test_name = f"bulk_insert_accounts_{batch_size}"
        test_result, metrics = test_bulk_insert_accounts(version, batch_size)
        results[test_name] = test_result
        metrics_dict[test_name] = metrics
    
    # Test bulk insert for transactions with different batch sizes
    for batch_size in TEST_PARAMETERS["bulk_insert"]["batch_sizes"]["transactions"]:
        test_name = f"bulk_insert_transactions_{batch_size}"
        test_result, metrics = test_bulk_insert_transactions(version, batch_size)
        results[test_name] = test_result
        metrics_dict[test_name] = metrics
    
    # Test bulk insert for loans with different batch sizes
    for batch_size in TEST_PARAMETERS["bulk_insert"]["batch_sizes"]["loans"]:
        test_name = f"bulk_insert_loans_{batch_size}"
        test_result, metrics = test_bulk_insert_loans(version, batch_size)
        results[test_name] = test_result
        metrics_dict[test_name] = metrics
    
    # Test parallel bulk insert for each collection
    for collection_name in COLLECTIONS.values():
        batch_size = TEST_PARAMETERS["bulk_insert"]["batch_sizes"].get(
            collection_name, [1000]
        )[0]
        
        test_name = f"parallel_bulk_insert_{collection_name}_{batch_size}x10"
        test_result, metrics = test_parallel_bulk_insert(
            version=version,
            collection_name=collection_name,
            batch_size=batch_size,
            num_batches=10,
            max_workers=4
        )
        results[test_name] = test_result
        metrics_dict[test_name] = metrics
    
    return results, metrics_dict


def compare_bulk_insert_performance() -> Dict[str, Any]:
    """
    Compare bulk insert performance between MongoDB v7.0 and v8.0.
    
    Returns:
        Dict[str, Any]: Comparison results
    """
    # Run tests for MongoDB v7.0
    logger.info("Running bulk insert tests for MongoDB v7.0")
    v7_results, v7_metrics = run_bulk_insert_tests('v7')
    
    # Run tests for MongoDB v8.0
    logger.info("Running bulk insert tests for MongoDB v8.0")
    v8_results, v8_metrics = run_bulk_insert_tests('v8')
    
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
    comparison = compare_bulk_insert_performance()
    
    # Print summary
    print("\nBulk Insert Performance Comparison (MongoDB v8.0 vs v7.0):")
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