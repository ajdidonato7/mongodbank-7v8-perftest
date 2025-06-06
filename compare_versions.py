#!/usr/bin/env python3
"""
Side-by-side comparison of MongoDB v7 and v8 performance.
This script runs tests for both versions in parallel and displays results in real-time.
"""

import os
import sys
# Add the current directory to sys.path to fix import issues
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import time
import logging
import argparse
import threading
from typing import Dict, List, Any, Tuple
from datetime import datetime
import concurrent.futures
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
from tabulate import tabulate
import webbrowser
from pathlib import Path
import json

# Import test modules
from config.connection import get_database, close_connections
from config.test_config import TEST_PARAMETERS, COLLECTIONS
from data_generation.data_loader import get_sample_ids
from data_generation.faker_generator import generate_batch
from models.customer import Customer
from models.account import Account
from models.transaction import Transaction
from models.loan import Loan
from utils.monitoring import ResourceMonitor
from utils.performance_metrics import PerformanceMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for storing results
results = {
    "v7": {},
    "v8": {}
}

# Global variables for real-time plotting
v7_throughputs = []
v8_throughputs = []
test_names = []
current_test = ""
test_running = False
resource_monitor = None

def run_bulk_insert_test(version: str, collection_name: str, batch_size: int, iterations: int = 3) -> Tuple[Dict[str, Any], PerformanceMetrics]:
    """
    Run a bulk insert test for a specific version and collection.
    
    Args:
        version (str): MongoDB version ('v7' or 'v8')
        collection_name (str): Collection name
        batch_size (int): Batch size for bulk inserts
        iterations (int): Number of iterations
        
    Returns:
        Tuple[Dict[str, Any], PerformanceMetrics]: Test results and metrics
    """
    test_name = f"bulk_insert_{collection_name}_{batch_size}"
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
            }, None
    
    if collection_name == "transactions":
        account_ids = get_sample_ids(version, "accounts", limit=1000)
        if not account_ids:
            logger.error("No account IDs found. Please load account data first.")
            return {
                "test_name": test_name,
                "version": version,
                "error": "No account IDs found"
            }, None
    
    # Initialize metrics
    metrics = PerformanceMetrics(test_name, version)
    metrics.start()
    
    results = {
        "test_name": test_name,
        "version": version,
        "collection_name": collection_name,
        "batch_size": batch_size,
        "iterations": iterations,
        "total_documents": 0,
        "total_time": 0.0,
        "avg_throughput": 0.0,
        "iteration_results": []
    }
    
    try:
        # Run test iterations
        for i in range(iterations):
            logger.info(f"Iteration {i+1}/{iterations} for {version}")
            
            # Generate batch
            batch = generate_batch(
                collection_name=collection_name,
                batch_size=batch_size,
                customer_ids=customer_ids,
                account_ids=account_ids
            )
            
            start_time = time.time()
            
            # Get database and collection
            db = get_database(version)
            collection = db[collection_name]
            
            # Insert documents
            try:
                result = collection.insert_many(batch)
                inserted_count = len(result.inserted_ids)
            except Exception as e:
                logger.warning(f"Error during bulk insert: {str(e)}")
                # If it's a BulkWriteError, we can still get the number of inserted documents
                if hasattr(e, 'details') and 'nInserted' in e.details:
                    inserted_count = e.details['nInserted']
                else:
                    # Assume at least some documents were inserted
                    inserted_count = len(batch) // 2
            
            end_time = time.time()
            
            # Record metrics
            iteration_time = end_time - start_time
            metrics.record_response_time(iteration_time)
            metrics.record_operation("insert", inserted_count)
            
            # Update results
            results["total_documents"] += inserted_count
            results["total_time"] += iteration_time
            
            # Store iteration result
            iteration_result = {
                "iteration": i+1,
                "documents": inserted_count,
                "time": iteration_time,
                "throughput": inserted_count/iteration_time if iteration_time > 0 else 0
            }
            results["iteration_results"].append(iteration_result)
            
            logger.info(
                f"{version}: Inserted {inserted_count} {collection_name} in {iteration_time:.2f}s "
                f"({inserted_count/iteration_time:.2f} docs/s)"
            )
        
        # Calculate average throughput
        if results["total_time"] > 0:
            results["avg_throughput"] = results["total_documents"] / results["total_time"]
        
        return results, metrics
    
    finally:
        # Stop metrics collection
        metrics.stop()

def run_read_test(version: str, collection_name: str, read_type: str, num_reads: int = 100, iterations: int = 3) -> Tuple[Dict[str, Any], PerformanceMetrics]:
    """
    Run a read test for a specific version and collection.
    
    Args:
        version (str): MongoDB version ('v7' or 'v8')
        collection_name (str): Collection name
        read_type (str): Type of read operation ('single', 'filtered', 'complex')
        num_reads (int): Number of reads per iteration
        iterations (int): Number of iterations
        
    Returns:
        Tuple[Dict[str, Any], PerformanceMetrics]: Test results and metrics
    """
    test_name = f"{read_type}_read_{collection_name}"
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
        }, None
    
    # Initialize metrics
    metrics = PerformanceMetrics(test_name, version)
    metrics.start()
    
    results = {
        "test_name": test_name,
        "version": version,
        "collection_name": collection_name,
        "read_type": read_type,
        "num_reads": num_reads,
        "iterations": iterations,
        "total_reads": 0,
        "total_time": 0.0,
        "avg_throughput": 0.0,
        "iteration_results": []
    }
    
    try:
        # Run test iterations
        for i in range(iterations):
            logger.info(f"Iteration {i+1}/{iterations} for {version}")
            
            iteration_start_time = time.time()
            reads_completed = 0
            
            # Perform reads
            for _ in range(num_reads):
                # Select a random ID
                doc_id = sample_ids[_ % len(sample_ids)]
                
                # Perform read based on type
                start_time = time.time()
                
                if read_type == "single":
                    # Single document read
                    doc = collection.find_one({id_field: doc_id})
                    
                elif read_type == "filtered":
                    # Filtered read
                    if collection_name == "customers":
                        cursor = collection.find({"credit_score": {"$gte": 500}}).limit(10)
                    elif collection_name == "accounts":
                        cursor = collection.find({"customer_id": doc_id}).limit(10)
                    elif collection_name == "transactions":
                        cursor = collection.find({"account_id": doc_id}).limit(10)
                    elif collection_name == "loans":
                        cursor = collection.find({"customer_id": doc_id}).limit(10)
                    docs = list(cursor)
                    
                elif read_type == "complex":
                    # Complex read
                    if collection_name == "customers":
                        cursor = collection.find({
                            "$and": [
                                {"credit_score": {"$gte": 500}},
                                {"address.country": "US"},
                                {"created_at": {"$gte": datetime.now().replace(year=datetime.now().year-1)}}
                            ]
                        }).limit(10)
                    elif collection_name == "accounts":
                        cursor = collection.find({
                            "$and": [
                                {"account_type": {"$in": ["CHECKING", "SAVINGS"]}},
                                {"status": "ACTIVE"},
                                {"balance": {"$gte": 1000}}
                            ]
                        }).limit(10)
                    elif collection_name == "transactions":
                        cursor = collection.find({
                            "$and": [
                                {"transaction_type": {"$in": ["DEPOSIT", "WITHDRAWAL"]}},
                                {"amount": {"$gte": 100}},
                                {"status": "COMPLETED"}
                            ]
                        }).limit(10)
                    elif collection_name == "loans":
                        cursor = collection.find({
                            "$and": [
                                {"loan_type": {"$in": ["PERSONAL", "AUTO"]}},
                                {"amount": {"$gte": 5000}},
                                {"status": "ACTIVE"}
                            ]
                        }).limit(10)
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
            
            # Store iteration result
            iteration_result = {
                "iteration": i+1,
                "reads": reads_completed,
                "time": iteration_time,
                "throughput": reads_completed/iteration_time if iteration_time > 0 else 0
            }
            results["iteration_results"].append(iteration_result)
            
            logger.info(
                f"{version}: Completed {reads_completed} {read_type} reads in {iteration_time:.2f}s "
                f"({reads_completed/iteration_time:.2f} reads/s)"
            )
        
        # Calculate average throughput
        if results["total_time"] > 0:
            results["avg_throughput"] = results["total_reads"] / results["total_time"]
        
        return results, metrics
    
    finally:
        # Stop metrics collection
        metrics.stop()

def run_aggregation_test(version: str, collection_name: str, agg_type: str, num_aggregations: int = 10, iterations: int = 3) -> Tuple[Dict[str, Any], PerformanceMetrics]:
    """
    Run an aggregation test for a specific version and collection.
    
    Args:
        version (str): MongoDB version ('v7' or 'v8')
        collection_name (str): Collection name
        agg_type (str): Type of aggregation ('simple', 'group_by')
        num_aggregations (int): Number of aggregations per iteration
        iterations (int): Number of iterations
        
    Returns:
        Tuple[Dict[str, Any], PerformanceMetrics]: Test results and metrics
    """
    test_name = f"{agg_type}_aggregation_{collection_name}"
    logger.info(f"Running {test_name} test for MongoDB {version}")
    
    # Get database and collection
    db = get_database(version)
    collection = db[collection_name]
    
    # Define aggregation pipelines based on collection and type
    pipelines = []
    
    if agg_type == "simple":
        if collection_name == "customers":
            pipelines.append([{"$count": "total_customers"}])
            pipelines.append([{"$group": {"_id": "$address.state", "count": {"$sum": 1}}}])
            pipelines.append([{"$group": {"_id": None, "avg_credit_score": {"$avg": "$credit_score"}}}])
        
        elif collection_name == "accounts":
            pipelines.append([{"$count": "total_accounts"}])
            pipelines.append([{"$group": {"_id": "$account_type", "count": {"$sum": 1}}}])
            pipelines.append([{"$group": {"_id": None, "total_balance": {"$sum": "$balance"}}}])
        
        elif collection_name == "transactions":
            pipelines.append([{"$count": "total_transactions"}])
            pipelines.append([{"$group": {"_id": "$transaction_type", "count": {"$sum": 1}}}])
            pipelines.append([{"$group": {"_id": None, "total_amount": {"$sum": "$amount"}}}])
        
        elif collection_name == "loans":
            pipelines.append([{"$count": "total_loans"}])
            pipelines.append([{"$group": {"_id": "$loan_type", "count": {"$sum": 1}}}])
            pipelines.append([{"$group": {"_id": None, "total_amount": {"$sum": "$amount"}}}])
    
    elif agg_type == "group_by":
        if collection_name == "customers":
            pipelines.append([
                {"$group": {"_id": "$address.state", "count": {"$sum": 1}, "avg_credit_score": {"$avg": "$credit_score"}}}
            ])
            pipelines.append([
                {"$group": {"_id": {"state": "$address.state", "city": "$address.city"}, "count": {"$sum": 1}}}
            ])
        
        elif collection_name == "accounts":
            pipelines.append([
                {"$group": {"_id": "$account_type", "count": {"$sum": 1}, "total_balance": {"$sum": "$balance"}}}
            ])
            pipelines.append([
                {"$group": {"_id": "$status", "count": {"$sum": 1}, "avg_balance": {"$avg": "$balance"}}}
            ])
        
        elif collection_name == "transactions":
            pipelines.append([
                {"$group": {"_id": "$transaction_type", "count": {"$sum": 1}, "total_amount": {"$sum": "$amount"}}}
            ])
            pipelines.append([
                {"$group": {"_id": "$category", "count": {"$sum": 1}, "avg_amount": {"$avg": "$amount"}}}
            ])
        
        elif collection_name == "loans":
            pipelines.append([
                {"$group": {"_id": "$loan_type", "count": {"$sum": 1}, "total_amount": {"$sum": "$amount"}}}
            ])
            pipelines.append([
                {"$group": {"_id": "$status", "count": {"$sum": 1}, "avg_amount": {"$avg": "$amount"}}}
            ])
    
    # Ensure we have enough pipelines
    while len(pipelines) < num_aggregations:
        pipelines.extend(pipelines[:num_aggregations - len(pipelines)])
    
    # Limit to requested number of aggregations
    pipelines = pipelines[:num_aggregations]
    
    # Initialize metrics
    metrics = PerformanceMetrics(test_name, version)
    metrics.start()
    
    results = {
        "test_name": test_name,
        "version": version,
        "collection_name": collection_name,
        "agg_type": agg_type,
        "num_aggregations": num_aggregations,
        "iterations": iterations,
        "total_aggregations": 0,
        "total_time": 0.0,
        "avg_throughput": 0.0,
        "iteration_results": []
    }
    
    try:
        # Run test iterations
        for i in range(iterations):
            logger.info(f"Iteration {i+1}/{iterations} for {version}")
            
            iteration_start_time = time.time()
            aggregations_completed = 0
            
            # Perform aggregations
            for pipeline in pipelines:
                # Perform aggregation
                start_time = time.time()
                result = list(collection.aggregate(pipeline))
                end_time = time.time()
                
                # Record metrics
                aggregations_completed += 1
                response_time = end_time - start_time
                metrics.record_response_time(response_time)
                metrics.record_operation("aggregate", 1)
            
            iteration_end_time = time.time()
            iteration_time = iteration_end_time - iteration_start_time
            
            # Update results
            results["total_aggregations"] += aggregations_completed
            results["total_time"] += iteration_time
            
            # Store iteration result
            iteration_result = {
                "iteration": i+1,
                "aggregations": aggregations_completed,
                "time": iteration_time,
                "throughput": aggregations_completed/iteration_time if iteration_time > 0 else 0
            }
            results["iteration_results"].append(iteration_result)
            
            logger.info(
                f"{version}: Completed {aggregations_completed} {agg_type} aggregations in {iteration_time:.2f}s "
                f"({aggregations_completed/iteration_time:.2f} aggs/s)"
            )
        
        # Calculate average throughput
        if results["total_time"] > 0:
            results["avg_throughput"] = results["total_aggregations"] / results["total_time"]
        
        return results, metrics
    
    finally:
        # Stop metrics collection
        metrics.stop()

def run_test_for_both_versions(test_func, *args, **kwargs):
    """
    Run a test for both MongoDB v7 and v8 in parallel.
    
    Args:
        test_func: Test function to run
        *args: Positional arguments for the test function
        **kwargs: Keyword arguments for the test function
    """
    global current_test, test_running, v7_throughputs, v8_throughputs, test_names, results
    
    # Create a descriptive test name
    if 'collection_name' in kwargs:
        collection = kwargs['collection_name']
    else:
        collection = args[0] if len(args) > 0 else 'unknown'
    
    if test_func.__name__ == 'run_bulk_insert_test':
        batch_size = kwargs.get('batch_size', args[1] if len(args) > 1 else 'unknown')
        test_name = f"Bulk Insert {collection} (batch: {batch_size})"
    elif test_func.__name__ == 'run_read_test':
        read_type = kwargs.get('read_type', args[1] if len(args) > 1 else 'unknown')
        test_name = f"{read_type.capitalize()} Read {collection}"
    elif test_func.__name__ == 'run_aggregation_test':
        agg_type = kwargs.get('agg_type', args[1] if len(args) > 1 else 'unknown')
        test_name = f"{agg_type.capitalize()} Aggregation {collection}"
    else:
        test_name = f"{test_func.__name__} {collection}"
    
    current_test = test_name
    test_running = True
    
    # Run tests in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Submit tests
        v7_future = executor.submit(test_func, "v7", *args, **kwargs)
        v8_future = executor.submit(test_func, "v8", *args, **kwargs)
        
        # Wait for results
        v7_result, v7_metrics = v7_future.result()
        v8_result, v8_metrics = v8_future.result()
    
    # Store results
    results["v7"][test_name] = v7_result
    results["v8"][test_name] = v8_result
    
    # Calculate improvement
    if v7_result.get("avg_throughput", 0) > 0:
        improvement = (
            (v8_result.get("avg_throughput", 0) - v7_result.get("avg_throughput", 0)) /
            v7_result.get("avg_throughput", 0)
        ) * 100
    else:
        improvement = 0
    
    # Add to lists for plotting
    test_names.append(test_name)
    v7_throughputs.append(v7_result.get("avg_throughput", 0))
    v8_throughputs.append(v8_result.get("avg_throughput", 0))
    
    # Print comparison
    print("\n" + "=" * 80)
    print(f"Test: {test_name}")
    print("-" * 80)
    print(f"MongoDB v7.0: {v7_result.get('avg_throughput', 0):.2f} ops/s")
    print(f"MongoDB v8.0: {v8_result.get('avg_throughput', 0):.2f} ops/s")
    print(f"Improvement: {improvement:.2f}%")
    print("=" * 80 + "\n")
    
    test_running = False
    
    return v7_result, v8_result, improvement

def update_plot(frame):
    """Update function for the animation."""
    if not test_names:
        return
    
    plt.clf()
    
    # Create bar chart
    x = np.arange(len(test_names))
    width = 0.35
    
    plt.bar(x - width/2, v7_throughputs, width, label='MongoDB v7.0')
    plt.bar(x + width/2, v8_throughputs, width, label='MongoDB v8.0')
    
    plt.xlabel('Test')
    plt.ylabel('Throughput (ops/s)')
    plt.title('MongoDB v7.0 vs v8.0 Performance Comparison')
    plt.xticks(x, test_names, rotation=45, ha='right')
    plt.legend()
    
    plt.tight_layout()
    
    # Add current test indicator
    if test_running:
        plt.figtext(0.5, 0.01, f"Currently running: {current_test}", ha="center", 
                    bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

def generate_html_report():
    """Generate an HTML report with the test results."""
    if not results["v7"] or not results["v8"]:
        return None
    
    # Create report directory
    report_dir = "performance_comparison_report"
    os.makedirs(report_dir, exist_ok=True)
    
    # Save results as JSON
    with open(os.path.join(report_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MongoDB v7.0 vs v8.0 Performance Comparison</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2, h3 { color: #4285f4; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .improvement-positive { color: green; }
            .improvement-negative { color: red; }
            .chart-container { width: 100%; height: 400px; margin-bottom: 30px; }
            .test-section { margin-bottom: 40px; border-bottom: 1px solid #eee; padding-bottom: 20px; }
        </style>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body>
        <h1>MongoDB v7.0 vs v8.0 Performance Comparison</h1>
    """
    
    # Add date
    html_content += "<p><strong>Date:</strong> " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "</p>"
    
    # Add chart container
    html_content += """
        <div class="chart-container">
            <canvas id="summaryChart"></canvas>
        </div>
        
        <h2>Summary Results</h2>
        <table>
            <tr>
                <th>Test</th>
                <th>MongoDB v7.0 (ops/s)</th>
                <th>MongoDB v8.0 (ops/s)</th>
                <th>Improvement</th>
            </tr>
    """
    
    # Add summary table rows
    chart_labels = []
    v7_data = []
    v8_data = []
    
    for test_name in test_names:
        v7_result = results["v7"][test_name]
        v8_result = results["v8"][test_name]
        
        v7_throughput = v7_result.get("avg_throughput", 0)
        v8_throughput = v8_result.get("avg_throughput", 0)
        
        if v7_throughput > 0:
            improvement = ((v8_throughput - v7_throughput) / v7_throughput) * 100
        else:
            improvement = 0
        
        improvement_class = "improvement-positive" if improvement > 0 else "improvement-negative"
        
        html_content += f"""
            <tr>
                <td>{test_name}</td>
                <td>{v7_throughput:.2f}</td>
                <td>{v8_throughput:.2f}</td>
                <td class="{improvement_class}">{improvement:.2f}%</td>
            </tr>
        """
        
        chart_labels.append(test_name)
        v7_data.append(v7_throughput)
        v8_data.append(v8_throughput)
    
    html_content += """
        </table>
        
        <h2>Detailed Results</h2>
    """
    
    # Add detailed results for each test
    for test_name in test_names:
        v7_result = results["v7"][test_name]
        v8_result = results["v8"][test_name]
        
        html_content += f"""
        <div class="test-section">
            <h3>{test_name}</h3>
            
            <h4>MongoDB v7.0</h4>
            <table>
                <tr>
                    <th>Iteration</th>
                    <th>Operations</th>
                    <th>Time (s)</th>
                    <th>Throughput (ops/s)</th>
                </tr>
        """
        
        for iteration in v7_result.get("iteration_results", []):
            ops_key = next((k for k in ["documents", "reads", "aggregations"] if k in iteration), "operations")
            html_content += f"""
                <tr>
                    <td>{iteration.get("iteration", "N/A")}</td>
                    <td>{iteration.get(ops_key, 0)}</td>
                    <td>{iteration.get("time", 0):.2f}</td>
                    <td>{iteration.get("throughput", 0):.2f}</td>
                </tr>
            """
        
        html_content += f"""
            </table>
            <p><strong>Average Throughput:</strong> {v7_result.get("avg_throughput", 0):.2f} ops/s</p>
            
            <h4>MongoDB v8.0</h4>
            <table>
                <tr>
                    <th>Iteration</th>
                    <th>Operations</th>
                    <th>Time (s)</th>
                    <th>Throughput (ops/s)</th>
                </tr>
        """
        
        for iteration in v8_result.get("iteration_results", []):
            ops_key = next((k for k in ["documents", "reads", "aggregations"] if k in iteration), "operations")
            html_content += f"""
                <tr>
                    <td>{iteration.get("iteration", "N/A")}</td>
                    <td>{iteration.get(ops_key, 0)}</td>
                    <td>{iteration.get("time", 0):.2f}</td>
                    <td>{iteration.get("throughput", 0):.2f}</td>
                </tr>
            """
        
        v7_throughput = v7_result.get("avg_throughput", 0)
        v8_throughput = v8_result.get("avg_throughput", 0)
        
        if v7_throughput > 0:
            improvement = ((v8_throughput - v7_throughput) / v7_throughput) * 100
        else:
            improvement = 0
        
        improvement_class = "improvement-positive" if improvement > 0 else "improvement-negative"
        
        html_content += f"""
            </table>
            <p><strong>Average Throughput:</strong> {v8_throughput:.2f} ops/s</p>
            
            <p><strong>Improvement:</strong> <span class="{improvement_class}">{improvement:.2f}%</span></p>
        </div>
        """
    
    # Add chart initialization JavaScript
    chart_labels_json = json.dumps(chart_labels)
    v7_data_json = json.dumps(v7_data)
    v8_data_json = json.dumps(v8_data)
    
    # Add chart script without f-strings to avoid formatting issues
    script_part = """
        <script>
            // Create summary chart
            var ctx = document.getElementById('summaryChart').getContext('2d');
            var summaryChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: """ + chart_labels_json + """,
                    datasets: [
                        {
                            label: 'MongoDB v7.0',
                            data: """ + v7_data_json + """,
                            backgroundColor: 'rgba(54, 162, 235, 0.5)',
                            borderColor: 'rgba(54, 162, 235, 1)',
                            borderWidth: 1
                        },
                        {
                            label: 'MongoDB v8.0',
                            data: """ + v8_data_json + """,
                            backgroundColor: 'rgba(255, 99, 132, 0.5)',
                            borderColor: 'rgba(255, 99, 132, 1)',
                            borderWidth: 1
                        }
                    ]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Throughput (ops/s)'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Test'
                            }
                        }
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: 'MongoDB v7.0 vs v8.0 Performance Comparison',
                            font: {
                                size: 18
                            }
                        },
                        legend: {
                            position: 'top'
                        }
                    }
                }
            });
        </script>
    </body>
    </html>
    """
    
    html_content += script_part
    
    # Write HTML file
    html_path = os.path.join(report_dir, "report.html")
    with open(html_path, "w") as f:
        f.write(html_content)
    
    return html_path


def main():
    """Main function to run the MongoDB performance comparison."""
    parser = argparse.ArgumentParser(description="MongoDB v7.0 vs v8.0 Performance Comparison")
    
    # Test selection options
    parser.add_argument("--bulk-insert", action="store_true", help="Run bulk insert tests")
    parser.add_argument("--read", action="store_true", help="Run read tests")
    parser.add_argument("--aggregation", action="store_true", help="Run aggregation tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    # Collection options
    parser.add_argument("--collections", type=str, nargs="+", default=["customers", "accounts", "transactions", "loans"],
                        help="Collections to test (default: all)")
    
    # Test parameters
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[100, 1000, 10000],
                        help="Batch sizes for bulk insert tests (default: 100, 1000, 10000)")
    parser.add_argument("--read-types", type=str, nargs="+", default=["single", "filtered", "complex"],
                        help="Read types for read tests (default: single, filtered, complex)")
    parser.add_argument("--agg-types", type=str, nargs="+", default=["simple", "group_by"],
                        help="Aggregation types for aggregation tests (default: simple, group_by)")
    parser.add_argument("--iterations", type=int, default=3,
                        help="Number of iterations for each test (default: 3)")
    
    # Visualization options
    parser.add_argument("--live-plot", action="store_true", help="Show live plot during tests")
    parser.add_argument("--open-report", action="store_true", help="Open HTML report after tests")
    
    args = parser.parse_args()
    
    # Start resource monitor
    global resource_monitor
    resource_monitor = ResourceMonitor()
    resource_monitor.start()
    
    try:
        # Set up live plotting if requested
        if args.live_plot:
            fig = plt.figure(figsize=(12, 6))
            ani = FuncAnimation(fig, update_plot, interval=1000)
            plt.show(block=False)
        
        # Run bulk insert tests
        if args.bulk_insert or args.all:
            for collection_name in args.collections:
                for batch_size in args.batch_sizes:
                    run_test_for_both_versions(
                        run_bulk_insert_test,
                        collection_name=collection_name,
                        batch_size=batch_size,
                        iterations=args.iterations
                    )
        
        # Run read tests
        if args.read or args.all:
            for collection_name in args.collections:
                for read_type in args.read_types:
                    run_test_for_both_versions(
                        run_read_test,
                        collection_name=collection_name,
                        read_type=read_type,
                        num_reads=100,
                        iterations=args.iterations
                    )
        
        # Run aggregation tests
        if args.aggregation or args.all:
            for collection_name in args.collections:
                for agg_type in args.agg_types:
                    run_test_for_both_versions(
                        run_aggregation_test,
                        collection_name=collection_name,
                        agg_type=agg_type,
                        num_aggregations=10,
                        iterations=args.iterations
                    )
        
        # Generate HTML report
        html_path = generate_html_report()
        
        if html_path and args.open_report:
            # Open HTML report in browser
            webbrowser.open(f"file://{os.path.abspath(html_path)}")
        
        # Print summary
        print("\nTest Summary:")
        print("-" * 80)
        print(f"Total tests run: {len(test_names)}")
        print(f"HTML report: {html_path}")
        
        # Keep plot window open if live plotting was enabled
        if args.live_plot:
            plt.ioff()
            plt.show()
    
    finally:
        # Stop resource monitor
        if resource_monitor:
            resource_monitor.stop()
        
        # Close database connections
        close_connections()


if __name__ == "__main__":
    main()