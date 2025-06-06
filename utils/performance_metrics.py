"""
Performance metrics module for MongoDB performance testing.
This module provides classes to track and analyze performance metrics.
"""

import time
import threading
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import statistics
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Class for tracking performance metrics during tests."""
    
    def __init__(self, test_name: str, version: str):
        """
        Initialize a new PerformanceMetrics instance.
        
        Args:
            test_name (str): Name of the test
            version (str): MongoDB version
        """
        self.test_name = test_name
        self.version = version
        self.start_time = None
        self.end_time = None
        self.response_times = []
        self.operations = {
            "insert": 0,
            "find": 0,
            "update": 0,
            "delete": 0,
            "aggregate": 0
        }
        self.throughput_over_time = []
        self.operation_counts_over_time = []
        self.timestamps = []
    
    def start(self) -> None:
        """Start tracking metrics."""
        self.start_time = time.time()
        logger.info(f"Started metrics tracking for {self.test_name} on MongoDB {self.version}")
    
    def stop(self) -> None:
        """Stop tracking metrics."""
        self.end_time = time.time()
        logger.info(f"Stopped metrics tracking for {self.test_name} on MongoDB {self.version}")
    
    def record_response_time(self, response_time: float) -> None:
        """
        Record a response time.
        
        Args:
            response_time (float): Response time in seconds
        """
        self.response_times.append(response_time)
    
    def record_operation(self, operation_type: str, count: int = 1) -> None:
        """
        Record an operation.
        
        Args:
            operation_type (str): Type of operation
            count (int): Number of operations
        """
        if operation_type in self.operations:
            self.operations[operation_type] += count
        else:
            self.operations[operation_type] = count
        
        # Record throughput data point
        current_time = time.time()
        elapsed_time = current_time - self.start_time if self.start_time else 0
        
        if elapsed_time > 0:
            total_ops = sum(self.operations.values())
            throughput = total_ops / elapsed_time
            
            self.throughput_over_time.append(throughput)
            self.operation_counts_over_time.append(dict(self.operations))
            self.timestamps.append(current_time)
    
    def get_total_operations(self) -> int:
        """
        Get the total number of operations.
        
        Returns:
            int: Total number of operations
        """
        return sum(self.operations.values())
    
    def get_total_time(self) -> float:
        """
        Get the total time elapsed.
        
        Returns:
            float: Total time in seconds
        """
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return time.time() - self.start_time
        else:
            return 0
    
    def get_throughput(self) -> float:
        """
        Get the overall throughput.
        
        Returns:
            float: Throughput in operations per second
        """
        total_time = self.get_total_time()
        if total_time > 0:
            return self.get_total_operations() / total_time
        else:
            return 0
    
    def get_response_time_stats(self) -> Dict[str, float]:
        """
        Get response time statistics.
        
        Returns:
            Dict[str, float]: Response time statistics
        """
        if not self.response_times:
            return {
                "min": 0,
                "max": 0,
                "avg": 0,
                "median": 0,
                "p95": 0,
                "p99": 0
            }
        
        sorted_times = sorted(self.response_times)
        p95_index = int(len(sorted_times) * 0.95)
        p99_index = int(len(sorted_times) * 0.99)
        
        return {
            "min": min(self.response_times),
            "max": max(self.response_times),
            "avg": sum(self.response_times) / len(self.response_times),
            "median": statistics.median(self.response_times),
            "p95": sorted_times[p95_index],
            "p99": sorted_times[p99_index]
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all metrics.
        
        Returns:
            Dict[str, Any]: Summary of all metrics
        """
        return {
            "test_name": self.test_name,
            "version": self.version,
            "total_time": self.get_total_time(),
            "total_operations": self.get_total_operations(),
            "throughput": self.get_throughput(),
            "operations": self.operations,
            "response_time_stats": self.get_response_time_stats()
        }
    
    def to_dataframe(self) -> Dict[str, pd.DataFrame]:
        """
        Convert metrics data to pandas DataFrames.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of DataFrames
        """
        dfs = {}
        
        # Create timestamp series
        timestamps = [datetime.fromtimestamp(ts) for ts in self.timestamps]
        
        # Throughput over time
        if self.throughput_over_time:
            dfs["throughput"] = pd.DataFrame({
                "timestamp": timestamps,
                "throughput": self.throughput_over_time
            })
        
        # Response time distribution
        if self.response_times:
            dfs["response_times"] = pd.DataFrame({
                "response_time": self.response_times
            })
        
        # Operation counts over time
        if self.operation_counts_over_time:
            # Create a DataFrame with timestamps
            ops_df = pd.DataFrame({
                "timestamp": timestamps
            })
            
            # Add columns for each operation type
            for op_type in self.operations.keys():
                ops_df[op_type] = [ops.get(op_type, 0) for ops in self.operation_counts_over_time]
            
            dfs["operations"] = ops_df
        
        return dfs
    
    def save_to_csv(self, output_dir: str) -> Dict[str, str]:
        """
        Save metrics data to CSV files.
        
        Args:
            output_dir (str): Output directory
            
        Returns:
            Dict[str, str]: Dictionary with file paths
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        file_paths = {}
        dfs = self.to_dataframe()
        
        for name, df in dfs.items():
            file_name = f"{self.test_name}_{self.version}_{name}.csv"
            file_path = os.path.join(output_dir, file_name)
            df.to_csv(file_path, index=False)
            file_paths[name] = file_path
        
        # Save summary as JSON
        summary = self.get_summary()
        summary_file = os.path.join(output_dir, f"{self.test_name}_{self.version}_summary.json")
        
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        file_paths["summary"] = summary_file
        
        return file_paths
    
    def generate_charts(self, output_dir: str) -> Dict[str, str]:
        """
        Generate charts from metrics data.
        
        Args:
            output_dir (str): Output directory
            
        Returns:
            Dict[str, str]: Dictionary with file paths
        """
        # Create charts directory
        charts_dir = os.path.join(output_dir, "charts")
        os.makedirs(charts_dir, exist_ok=True)
        
        chart_paths = {}
        
        # Throughput over time
        if self.throughput_over_time and self.timestamps:
            plt.figure(figsize=(10, 6))
            plt.plot(
                [datetime.fromtimestamp(ts) for ts in self.timestamps],
                self.throughput_over_time
            )
            plt.title(f"Throughput Over Time - {self.test_name} ({self.version})")
            plt.xlabel("Time")
            plt.ylabel("Throughput (ops/s)")
            plt.grid(True)
            plt.tight_layout()
            
            chart_path = os.path.join(charts_dir, "throughput_over_time.png")
            plt.savefig(chart_path)
            plt.close()
            
            chart_paths["throughput_over_time"] = chart_path
        
        # Response time distribution
        if self.response_times:
            plt.figure(figsize=(10, 6))
            plt.hist(self.response_times, bins=30, alpha=0.7)
            plt.title(f"Response Time Distribution - {self.test_name} ({self.version})")
            plt.xlabel("Response Time (s)")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.tight_layout()
            
            chart_path = os.path.join(charts_dir, "response_time_distribution.png")
            plt.savefig(chart_path)
            plt.close()
            
            chart_paths["response_time_distribution"] = chart_path
        
        # Response time percentiles
        if self.response_times:
            sorted_times = sorted(self.response_times)
            percentiles = [50, 75, 90, 95, 99]
            percentile_values = [
                np.percentile(sorted_times, p) for p in percentiles
            ]
            
            plt.figure(figsize=(10, 6))
            plt.bar(
                [f"p{p}" for p in percentiles],
                percentile_values,
                alpha=0.7
            )
            plt.title(f"Response Time Percentiles - {self.test_name} ({self.version})")
            plt.xlabel("Percentile")
            plt.ylabel("Response Time (s)")
            plt.grid(True)
            plt.tight_layout()
            
            chart_path = os.path.join(charts_dir, "response_time_percentiles.png")
            plt.savefig(chart_path)
            plt.close()
            
            chart_paths["response_time_percentiles"] = chart_path
        
        return chart_paths