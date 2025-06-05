"""
Mixed workload performance tests for MongoDB.
This module provides functions to test mixed workload performance.
"""

import logging
import time
import random
import threading
from typing import Dict, List, Any, Tuple, Callable
from datetime import datetime, timedelta
import concurrent.futures

from pymongo import MongoClient
from bson import ObjectId

from config.connection import get_database, close_connections
from config.test_config import TEST_PARAMETERS, COLLECTIONS
from data_generation import (
    generate_batch,
    get_sample_ids,
    bulk_insert,
    generate_and_load_batch
)
from utils import (
    PerformanceMetrics,
    run_with_metrics,
    compare_results,
    ResourceMonitor
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MixedWorkloadRunner:
    """Class for running mixed workload tests."""
    
    def __init__(
        self,
        version: str,
        read_write_ratio: float = TEST_PARAMETERS["mixed_workload"]["read_write_ratio"],
        duration_seconds: int = TEST_PARAMETERS["mixed_workload"]["duration_seconds"]
    ):
        """
        Initialize a new MixedWorkloadRunner instance.
        
        Args:
            version (str): MongoDB version ('v7' or 'v8')
            read_write_ratio (float): Ratio of read operations to write operations
            duration_seconds (int): Duration of the test in seconds
        """
        self.version = version
        self.read_write_ratio = read_write_ratio
        self.duration_seconds = duration_seconds
        self.db = get_database(version)
        self.stop_event = threading.Event()
        self.metrics = PerformanceMetrics(f"mixed_workload_{version}", version)
        self.operation_counts = {
            "read": 0,
            "write": 0,
            "total": 0
        }
        
        # Get sample IDs for each collection
        self.sample_ids = {}
        for collection_name in COLLECTIONS.values():
            id_field = f"{collection_name[:-1]}_id"  # Remove trailing 's' from collection name
            self.sample_ids[collection_name] = []
            
            # Get a sample of document IDs
            cursor = self.db[collection_name].find({}, {id_field: 1}).limit(1000)
            for doc in cursor:
                if id_field in doc:
                    self.sample_ids[collection_name].append(doc[id_field])
    
    def run(self, concurrency: int = 1) -> Dict[str, Any]:
        """
        Run the mixed workload test.
        
        Args:
            concurrency (int): Number of concurrent clients
            
        Returns:
            Dict[str, Any]: Test results
        """
        test_name = f"mixed_workload_{self.version}_concurrency_{concurrency}"
        logger.info(f"Running {test_name} test")
        
        # Initialize metrics
        self.metrics = PerformanceMetrics(test_name, self.version)
        self.metrics.start()
        
        # Reset operation counts
        self.operation_counts = {
            "read": 0,
            "write": 0,
            "total": 0
        }
        
        # Reset stop event
        self.stop_event.clear()
        
        # Create thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            # Submit client tasks
            futures = []
            for i in range(concurrency):
                future = executor.submit(self._client_task, i)
                futures.append(future)
            
            # Wait for duration
            time.sleep(self.duration_seconds)
            
            # Stop clients
            self.stop_event.set()
            
            # Wait for all clients to complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Client task generated an exception: {e}")
        
        # Stop metrics collection
        self.metrics.stop()
        
        # Calculate throughput
        throughput = self.operation_counts["total"] / self.duration_seconds
        
        # Prepare results
        results = {
            "test_name": test_name,
            "version": self.version,
            "concurrency": concurrency,
            "duration_seconds": self.duration_seconds,
            "read_write_ratio": self.read_write_ratio,
            "operations": self.operation_counts,
            "throughput": throughput
        }
        
        return results, self.metrics
    
    def _client_task(self, client_id: int) -> None:
        """
        Client task for mixed workload.
        
        Args:
            client_id (int): Client ID
        """
        logger.info(f"Client {client_id} started")
        
        while not self.stop_event.is_set():
            # Determine operation type (read or write)
            if random.random() < self.read_write_ratio:
                # Read operation
                self._perform_read_operation()
                self.operation_counts["read"] += 1
            else:
                # Write operation
                self._perform_write_operation()
                self.operation_counts["write"] += 1
            
            self.operation_counts["total"] += 1
            
            # Sleep for a short time to avoid overwhelming the database
            time.sleep(0.01)
        
        logger.info(f"Client {client_id} stopped")
    
    def _perform_read_operation(self) -> None:
        """Perform a random read operation."""
        # Choose a random collection
        collection_name = random.choice(list(COLLECTIONS.values()))
        collection = self.db[collection_name]
        
        # Choose a random read operation type
        operation_type = random.choice([
            "single_document",
            "filtered",
            "complex",
            "aggregation"
        ])
        
        start_time = time.time()
        
        if operation_type == "single_document":
            # Single document read
            if collection_name in self.sample_ids and self.sample_ids[collection_name]:
                id_field = f"{collection_name[:-1]}_id"  # Remove trailing 's' from collection name
                doc_id = random.choice(self.sample_ids[collection_name])
                doc = collection.find_one({id_field: doc_id})
        
        elif operation_type == "filtered":
            # Filtered read
            if collection_name == "customers":
                min_score = random.randint(300, 700)
                max_score = min_score + random.randint(50, 150)
                cursor = collection.find({"credit_score": {"$gte": min_score, "$lte": max_score}}).limit(10)
                docs = list(cursor)
            
            elif collection_name == "accounts" and "customers" in self.sample_ids and self.sample_ids["customers"]:
                customer_id = random.choice(self.sample_ids["customers"])
                cursor = collection.find({"customer_id": customer_id}).limit(10)
                docs = list(cursor)
            
            elif collection_name == "transactions" and "accounts" in self.sample_ids and self.sample_ids["accounts"]:
                account_id = random.choice(self.sample_ids["accounts"])
                cursor = collection.find({"account_id": account_id}).limit(10)
                docs = list(cursor)
            
            elif collection_name == "loans" and "customers" in self.sample_ids and self.sample_ids["customers"]:
                customer_id = random.choice(self.sample_ids["customers"])
                cursor = collection.find({"customer_id": customer_id}).limit(10)
                docs = list(cursor)
        
        elif operation_type == "complex":
            # Complex read
            if collection_name == "customers":
                min_score = random.randint(300, 700)
                max_score = min_score + random.randint(50, 150)
                
                cursor = collection.find({
                    "$and": [
                        {"credit_score": {"$gte": min_score, "$lte": max_score}},
                        {"address.country": "US"},
                        {"created_at": {"$gte": datetime.now() - timedelta(days=365)}}
                    ]
                }).limit(10)
                docs = list(cursor)
            
            elif collection_name == "accounts":
                account_types = ["CHECKING", "SAVINGS"]
                statuses = ["ACTIVE", "INACTIVE"]
                
                min_balance = random.randint(1000, 10000)
                max_balance = min_balance + random.randint(5000, 50000)
                
                cursor = collection.find({
                    "$and": [
                        {"account_type": {"$in": account_types}},
                        {"status": {"$in": statuses}},
                        {"balance": {"$gte": min_balance, "$lte": max_balance}},
                        {"created_at": {"$gte": datetime.now() - timedelta(days=365)}}
                    ]
                }).limit(10)
                docs = list(cursor)
            
            elif collection_name == "transactions":
                transaction_types = ["DEPOSIT", "WITHDRAWAL", "PAYMENT"]
                
                min_amount = random.randint(10, 500)
                max_amount = min_amount + random.randint(100, 1000)
                
                start_date = datetime.now() - timedelta(days=random.randint(30, 365))
                end_date = start_date + timedelta(days=random.randint(7, 30))
                
                cursor = collection.find({
                    "$and": [
                        {"transaction_type": {"$in": transaction_types}},
                        {"amount": {"$gte": min_amount, "$lte": max_amount}},
                        {"timestamp": {"$gte": start_date, "$lte": end_date}},
                        {"status": "COMPLETED"}
                    ]
                }).limit(10)
                docs = list(cursor)
            
            elif collection_name == "loans":
                loan_types = ["PERSONAL", "AUTO", "MORTGAGE"]
                
                min_amount = random.randint(5000, 50000)
                max_amount = min_amount + random.randint(10000, 100000)
                
                cursor = collection.find({
                    "$and": [
                        {"loan_type": {"$in": loan_types}},
                        {"amount": {"$gte": min_amount, "$lte": max_amount}},
                        {"status": {"$in": ["ACTIVE", "APPROVED"]}},
                        {"interest_rate": {"$gte": 3.0, "$lte": 8.0}}
                    ]
                }).limit(10)
                docs = list(cursor)
        
        elif operation_type == "aggregation":
            # Simple aggregation
            if collection_name == "customers":
                pipeline = [{"$group": {"_id": "$address.state", "count": {"$sum": 1}}}]
                result = list(collection.aggregate(pipeline))
            
            elif collection_name == "accounts":
                pipeline = [{"$group": {"_id": "$account_type", "count": {"$sum": 1}, "total_balance": {"$sum": "$balance"}}}]
                result = list(collection.aggregate(pipeline))
            
            elif collection_name == "transactions":
                pipeline = [{"$group": {"_id": "$transaction_type", "count": {"$sum": 1}, "total_amount": {"$sum": "$amount"}}}]
                result = list(collection.aggregate(pipeline))
            
            elif collection_name == "loans":
                pipeline = [{"$group": {"_id": "$loan_type", "count": {"$sum": 1}, "total_amount": {"$sum": "$amount"}}}]
                result = list(collection.aggregate(pipeline))
        
        end_time = time.time()
        
        # Record metrics
        response_time = end_time - start_time
        self.metrics.record_response_time(response_time)
        
        if operation_type == "aggregation":
            self.metrics.record_operation("aggregate", 1)
        else:
            self.metrics.record_operation("find", 1)
    
    def _perform_write_operation(self) -> None:
        """Perform a random write operation."""
        # Choose a random collection
        collection_name = random.choice(list(COLLECTIONS.values()))
        collection = self.db[collection_name]
        
        # Choose a random write operation type
        operation_type = random.choice([
            "insert",
            "update",
            "delete"
        ])
        
        start_time = time.time()
        
        if operation_type == "insert":
            # Insert operation
            if collection_name == "customers":
                batch = generate_batch(collection_name="customers", batch_size=1)
                result = collection.insert_one(batch[0])
            
            elif collection_name == "accounts" and "customers" in self.sample_ids and self.sample_ids["customers"]:
                customer_id = random.choice(self.sample_ids["customers"])
                batch = generate_batch(collection_name="accounts", batch_size=1, customer_ids=[customer_id])
                result = collection.insert_one(batch[0])
            
            elif collection_name == "transactions" and "accounts" in self.sample_ids and self.sample_ids["accounts"]:
                account_id = random.choice(self.sample_ids["accounts"])
                batch = generate_batch(collection_name="transactions", batch_size=1, account_ids=[account_id])
                result = collection.insert_one(batch[0])
            
            elif collection_name == "loans" and "customers" in self.sample_ids and self.sample_ids["customers"]:
                customer_id = random.choice(self.sample_ids["customers"])
                batch = generate_batch(collection_name="loans", batch_size=1, customer_ids=[customer_id])
                result = collection.insert_one(batch[0])
        
        elif operation_type == "update":
            # Update operation
            if collection_name in self.sample_ids and self.sample_ids[collection_name]:
                id_field = f"{collection_name[:-1]}_id"  # Remove trailing 's' from collection name
                doc_id = random.choice(self.sample_ids[collection_name])
                
                if collection_name == "customers":
                    update = {"$set": {"credit_score": random.randint(300, 850)}}
                elif collection_name == "accounts":
                    update = {"$set": {"balance": random.uniform(100, 10000)}}
                elif collection_name == "transactions":
                    update = {"$set": {"status": random.choice(["COMPLETED", "PENDING", "FAILED"])}}
                elif collection_name == "loans":
                    update = {"$set": {"status": random.choice(["ACTIVE", "PAID_OFF", "DEFAULTED"])}}
                
                result = collection.update_one({id_field: doc_id}, update)
        
        elif operation_type == "delete":
            # Delete operation (with low probability to avoid depleting the database)
            if random.random() < 0.1 and collection_name in self.sample_ids and self.sample_ids[collection_name]:
                id_field = f"{collection_name[:-1]}_id"  # Remove trailing 's' from collection name
                doc_id = random.choice(self.sample_ids[collection_name])
                result = collection.delete_one({id_field: doc_id})
                
                # Remove ID from sample IDs
                if doc_id in self.sample_ids[collection_name]:
                    self.sample_ids[collection_name].remove(doc_id)
        
        end_time = time.time()
        
        # Record metrics
        response_time = end_time - start_time
        self.metrics.record_response_time(response_time)
        
        if operation_type == "insert":
            self.metrics.record_operation("insert", 1)
        elif operation_type == "update":
            self.metrics.record_operation("update", 1)
        elif operation_type == "delete":
            self.metrics.record_operation("delete", 1)


def test_mixed_workload(
    version: str,
    concurrency: int,
    read_write_ratio: float = TEST_PARAMETERS["mixed_workload"]["read_write_ratio"],
    duration_seconds: int = TEST_PARAMETERS["mixed_workload"]["duration_seconds"]
) -> Tuple[Dict[str, Any], PerformanceMetrics]:
    """
    Test mixed workload performance.
    
    Args:
        version (str): MongoDB version ('v7' or 'v8')
        concurrency (int): Number of concurrent clients
        read_write_ratio (float): Ratio of read operations to write operations
        duration_seconds (int): Duration of the test in seconds
        
    Returns:
        Tuple[Dict[str, Any], PerformanceMetrics]: Test results and performance metrics
    """
    test_name = f"mixed_workload_{version}_concurrency_{concurrency}"
    logger.info(f"Running {test_name} test")
    
    # Initialize workload runner
    runner = MixedWorkloadRunner(
        version=version,
        read_write_ratio=read_write_ratio,
        duration_seconds=duration_seconds
    )
    
    # Initialize resource monitor
    monitor = ResourceMonitor()
    monitor.start()
    
    try:
        # Run test
        results, metrics = runner.run(concurrency=concurrency)
        
        # Add resource utilization to results
        resource_stats = monitor.get_summary()
        results["resource_utilization"] = resource_stats
        
        return results, metrics
    
    finally:
        # Stop resource monitor
        monitor.stop()


def run_mixed_workload_tests(
    version: str,
    concurrency_levels: List[int] = TEST_PARAMETERS["mixed_workload"]["concurrency"]
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, PerformanceMetrics]]:
    """
    Run mixed workload tests with different concurrency levels.
    
    Args:
        version (str): MongoDB version ('v7' or 'v8')
        concurrency_levels (List[int]): List of concurrency levels to test
        
    Returns:
        Tuple[Dict[str, Dict[str, Any]], Dict[str, PerformanceMetrics]]: 
            Dictionary with test results and dictionary with performance metrics
    """
    results = {}
    metrics_dict = {}
    
    for concurrency in concurrency_levels:
        test_name = f"mixed_workload_{version}_concurrency_{concurrency}"
        test_result, metrics = test_mixed_workload(version, concurrency)
        results[test_name] = test_result
        metrics_dict[test_name] = metrics
    
    return results, metrics_dict


def compare_mixed_workload_performance(
    concurrency_levels: List[int] = TEST_PARAMETERS["mixed_workload"]["concurrency"]
) -> Dict[str, Any]:
    """
    Compare mixed workload performance between MongoDB v7.0 and v8.0.
    
    Args:
        concurrency_levels (List[int]): List of concurrency levels to test
        
    Returns:
        Dict[str, Any]: Comparison results
    """
    # Run tests for MongoDB v7.0
    logger.info("Running mixed workload tests for MongoDB v7.0")
    v7_results, v7_metrics = run_mixed_workload_tests('v7', concurrency_levels)
    
    # Run tests for MongoDB v8.0
    logger.info("Running mixed workload tests for MongoDB v8.0")
    v8_results, v8_metrics = run_mixed_workload_tests('v8', concurrency_levels)
    
    # Compare results
    comparison = {}
    
    for concurrency in concurrency_levels:
        v7_test_name = f"mixed_workload_v7_concurrency_{concurrency}"
        v8_test_name = f"mixed_workload_v8_concurrency_{concurrency}"
        
        if v7_test_name in v7_results and v8_test_name in v8_results:
            v7_result = v7_results[v7_test_name]
            v8_result = v8_results[v8_test_name]
            
            # Calculate improvement
            if v7_result.get("throughput", 0) > 0:
                throughput_improvement = (
                    (v8_result.get("throughput", 0) - v7_result.get("throughput", 0)) /
                    v7_result.get("throughput", 0)
                ) * 100
            else:
                throughput_improvement = 0
            
            comparison[f"concurrency_{concurrency}"] = {
                "v7_throughput": v7_result.get("throughput", 0),
                "v8_throughput": v8_result.get("throughput", 0),
                "throughput_improvement_pct": throughput_improvement,
                "v7_operations": v7_result.get("operations", {}),
                "v8_operations": v8_result.get("operations", {})
            }
            
            # Generate report
            if v7_test_name in v7_metrics and v8_test_name in v8_metrics:
                report_files = compare_results(
                    test_name=f"mixed_workload_concurrency_{concurrency}",
                    v7_metrics=v7_metrics[v7_test_name],
                    v8_metrics=v8_metrics[v8_test_name]
                )
                comparison[f"concurrency_{concurrency}"]["report_files"] = report_files
    
    return comparison


if __name__ == "__main__":
    # Run comparison
    comparison = compare_mixed_workload_performance()
    
    # Print summary
    print("\nMixed Workload Performance Comparison (MongoDB v8.0 vs v7.0):")
    print("=" * 80)
    print(f"{'Concurrency':<15} {'v7.0 (ops/s)':<15} {'v8.0 (ops/s)':<15} {'Improvement':<15}")
    print("-" * 80)
    
    for concurrency_key, result in comparison.items():
        concurrency = concurrency_key.split("_")[1]
        print(
            f"{concurrency:<15} "
            f"{result['v7_throughput']:<15.2f} "
            f"{result['v8_throughput']:<15.2f} "
            f"{result['throughput_improvement_pct']:<15.2f}%"
        )
    
    print("=" * 80)
    
    # Close connections
    close_connections()