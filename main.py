#!/usr/bin/env python3
"""
Main entry point for MongoDB performance testing.
This script provides a command-line interface to run performance tests.
"""

import os
import sys
# Add the current directory to sys.path to fix import issues
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import argparse
import logging
import json
from typing import Dict, Any, List
import time
from datetime import datetime

from config.connection import get_database, close_connections
from config.test_config import COLLECTIONS
from data_generation import (
    load_initial_dataset,
    clear_collections,
    get_collection_counts
)
from tests import (
    run_bulk_insert_tests,
    compare_bulk_insert_performance,
    run_read_tests,
    compare_read_performance,
    run_aggregation_tests,
    compare_aggregation_performance,
    run_mixed_workload_tests,
    compare_mixed_workload_performance
)
from utils import (
    PerformanceReport,
    compare_results
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("mongodb_performance_test.log")
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="MongoDB Performance Testing Framework")
    
    parser.add_argument(
        "--test-type",
        choices=["all", "bulk_insert", "read", "aggregation", "mixed"],
        default="all",
        help="Type of test to run"
    )
    
    parser.add_argument(
        "--version",
        choices=["v7", "v8", "both"],
        default="both",
        help="MongoDB version to test"
    )
    
    parser.add_argument(
        "--load-data",
        action="store_true",
        help="Load initial dataset before running tests"
    )
    
    parser.add_argument(
        "--clear-data",
        action="store_true",
        help="Clear collections before loading data"
    )
    
    parser.add_argument(
        "--customer-count",
        type=int,
        default=100000,
        help="Number of customers to generate for initial dataset"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Output directory for reports"
    )
    
    parser.add_argument(
        "--report-formats",
        type=str,
        default="json,csv,html",
        help="Comma-separated list of report formats (json, csv, html)"
    )
    
    return parser.parse_args()


def load_data(version: str, customer_count: int, clear_data: bool) -> Dict[str, int]:
    """
    Load initial dataset.
    
    Args:
        version (str): MongoDB version ('v7' or 'v8')
        customer_count (int): Number of customers to generate
        clear_data (bool): Whether to clear collections before loading data
        
    Returns:
        Dict[str, int]: Dictionary with counts of inserted documents
    """
    logger.info(f"Loading initial dataset for MongoDB {version}")
    
    # Clear collections if requested
    if clear_data:
        logger.info(f"Clearing collections for MongoDB {version}")
        clear_collections(version)
    
    # Get current collection counts
    before_counts = get_collection_counts(version)
    logger.info(f"Collection counts before loading: {before_counts}")
    
    # Load initial dataset
    counts = load_initial_dataset(version, customer_count)
    logger.info(f"Inserted document counts: {counts}")
    
    # Get updated collection counts
    after_counts = get_collection_counts(version)
    logger.info(f"Collection counts after loading: {after_counts}")
    
    return counts


def run_tests(args) -> Dict[str, Any]:
    """
    Run performance tests.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Dict[str, Any]: Test results
    """
    results = {}
    versions = []
    
    if args.version == "both":
        versions = ["v7", "v8"]
    else:
        versions = [args.version]
    
    # Load initial dataset if requested
    if args.load_data:
        for version in versions:
            load_data(version, args.customer_count, args.clear_data)
    
    # Run tests
    if args.test_type == "all" or args.test_type == "bulk_insert":
        logger.info("Running bulk insert tests")
        bulk_insert_results = compare_bulk_insert_performance()
        results["bulk_insert"] = bulk_insert_results
    
    if args.test_type == "all" or args.test_type == "read":
        logger.info("Running read tests")
        read_results = compare_read_performance()
        results["read"] = read_results
    
    if args.test_type == "all" or args.test_type == "aggregation":
        logger.info("Running aggregation tests")
        aggregation_results = compare_aggregation_performance()
        results["aggregation"] = aggregation_results
    
    if args.test_type == "all" or args.test_type == "mixed":
        logger.info("Running mixed workload tests")
        mixed_results = compare_mixed_workload_performance()
        results["mixed"] = mixed_results
    
    return results


def generate_summary_report(results: Dict[str, Any], args) -> None:
    """
    Generate summary report.
    
    Args:
        results (Dict[str, Any]): Test results
        args: Command-line arguments
    """
    logger.info("Generating summary report")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create summary report directory
    summary_dir = os.path.join(args.output_dir, f"summary_{timestamp}")
    os.makedirs(summary_dir, exist_ok=True)
    
    # Parse report formats
    report_formats = args.report_formats.split(",")
    
    # Generate summary report
    summary = {
        "timestamp": timestamp,
        "test_types": args.test_type,
        "versions": args.version,
        "results": results
    }
    
    # Save summary report as JSON
    if "json" in report_formats:
        json_path = os.path.join(summary_dir, "summary.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Summary report saved to {json_path}")
    
    # Generate HTML summary report
    if "html" in report_formats:
        html_path = os.path.join(summary_dir, "summary.html")
        
        # Create HTML content
        html_content = f"""
        <html>
        <head>
            <title>MongoDB Performance Test Summary</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #4285f4; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .improvement-positive {{ color: green; }}
                .improvement-negative {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>MongoDB Performance Test Summary</h1>
            <p><strong>Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p><strong>Test Types:</strong> {args.test_type}</p>
            <p><strong>MongoDB Versions:</strong> {args.version}</p>
        """
        
        # Add bulk insert results
        if "bulk_insert" in results:
            html_content += """
            <h2>Bulk Insert Performance</h2>
            <table>
                <tr>
                    <th>Test</th>
                    <th>MongoDB v7.0 (ops/s)</th>
                    <th>MongoDB v8.0 (ops/s)</th>
                    <th>Improvement</th>
                </tr>
            """
            
            for test_name, result in results["bulk_insert"].items():
                improvement = result.get("throughput_improvement_pct", 0)
                improvement_class = "improvement-positive" if improvement > 0 else "improvement-negative"
                
                html_content += f"""
                <tr>
                    <td>{test_name}</td>
                    <td>{result.get("v7_throughput", 0):.2f}</td>
                    <td>{result.get("v8_throughput", 0):.2f}</td>
                    <td class="{improvement_class}">{improvement:.2f}%</td>
                </tr>
                """
            
            html_content += "</table>"
        
        # Add read results
        if "read" in results:
            html_content += """
            <h2>Read Performance</h2>
            <table>
                <tr>
                    <th>Test</th>
                    <th>MongoDB v7.0 (ops/s)</th>
                    <th>MongoDB v8.0 (ops/s)</th>
                    <th>Improvement</th>
                </tr>
            """
            
            for test_name, result in results["read"].items():
                improvement = result.get("throughput_improvement_pct", 0)
                improvement_class = "improvement-positive" if improvement > 0 else "improvement-negative"
                
                html_content += f"""
                <tr>
                    <td>{test_name}</td>
                    <td>{result.get("v7_throughput", 0):.2f}</td>
                    <td>{result.get("v8_throughput", 0):.2f}</td>
                    <td class="{improvement_class}">{improvement:.2f}%</td>
                </tr>
                """
            
            html_content += "</table>"
        
        # Add aggregation results
        if "aggregation" in results:
            html_content += """
            <h2>Aggregation Performance</h2>
            <table>
                <tr>
                    <th>Test</th>
                    <th>MongoDB v7.0 (ops/s)</th>
                    <th>MongoDB v8.0 (ops/s)</th>
                    <th>Improvement</th>
                </tr>
            """
            
            for test_name, result in results["aggregation"].items():
                improvement = result.get("throughput_improvement_pct", 0)
                improvement_class = "improvement-positive" if improvement > 0 else "improvement-negative"
                
                html_content += f"""
                <tr>
                    <td>{test_name}</td>
                    <td>{result.get("v7_throughput", 0):.2f}</td>
                    <td>{result.get("v8_throughput", 0):.2f}</td>
                    <td class="{improvement_class}">{improvement:.2f}%</td>
                </tr>
                """
            
            html_content += "</table>"
        
        # Add mixed workload results
        if "mixed" in results:
            html_content += """
            <h2>Mixed Workload Performance</h2>
            <table>
                <tr>
                    <th>Concurrency</th>
                    <th>MongoDB v7.0 (ops/s)</th>
                    <th>MongoDB v8.0 (ops/s)</th>
                    <th>Improvement</th>
                </tr>
            """
            
            for concurrency_key, result in results["mixed"].items():
                concurrency = concurrency_key.split("_")[1]
                improvement = result.get("throughput_improvement_pct", 0)
                improvement_class = "improvement-positive" if improvement > 0 else "improvement-negative"
                
                html_content += f"""
                <tr>
                    <td>{concurrency}</td>
                    <td>{result.get("v7_throughput", 0):.2f}</td>
                    <td>{result.get("v8_throughput", 0):.2f}</td>
                    <td class="{improvement_class}">{improvement:.2f}%</td>
                </tr>
                """
            
            html_content += "</table>"
        
        html_content += """
        </body>
        </html>
        """
        
        with open(html_path, "w") as f:
            f.write(html_content)
        
        logger.info(f"HTML summary report saved to {html_path}")


def main():
    """Main entry point."""
    # Parse command-line arguments
    args = parse_args()
    
    try:
        # Run tests
        results = run_tests(args)
        
        # Generate summary report
        generate_summary_report(results, args)
        
        logger.info("Performance testing completed successfully")
        return 0
    
    except Exception as e:
        logger.exception(f"Error during performance testing: {e}")
        return 1
    
    finally:
        # Close connections
        close_connections()


if __name__ == "__main__":
    sys.exit(main())