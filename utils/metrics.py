"""
Performance metrics collection module.
This module provides functions to collect and analyze performance metrics.
"""

import time
import statistics
import logging
import threading
from typing import Dict, List, Any, Tuple, Callable, Optional
from datetime import datetime
import numpy as np
import psutil
import pandas as pd
from pymongo import monitoring

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Class for collecting and analyzing performance metrics."""
    
    def __init__(self, test_name: str, version: str):
        """
        Initialize a new PerformanceMetrics instance.
        
        Args:
            test_name (str): Name of the test
            version (str): MongoDB version ('v7' or 'v8')
        """
        self.test_name = test_name
        self.version = version
        self.start_time = None
        self.end_time = None
        self.response_times = []
        self.throughput_data = []
        self.cpu_usage = []
        self.memory_usage = []
        self.disk_io = []
        self.network_io = []
        self.operation_counts = {
            "insert": 0,
            "find": 0,
            "update": 0,
            "delete": 0,
            "aggregate": 0,
            "command": 0,
            "total": 0
        }
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
    
    def start(self) -> None:
        """Start collecting metrics."""
        self.start_time = time.time()
        self._start_resource_monitoring()
    
    def stop(self) -> None:
        """Stop collecting metrics."""
        self.end_time = time.time()
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join()
    
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
        if operation_type in self.operation_counts:
            self.operation_counts[operation_type] += count
        self.operation_counts["total"] += count
    
    def _start_resource_monitoring(self, interval: float = 1.0) -> None:
        """
        Start monitoring system resources.
        
        Args:
            interval (float): Sampling interval in seconds
        """
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitor_resources,
            args=(interval,)
        )
        self._monitoring_thread.daemon = True
        self._monitoring_thread.start()
    
    def _monitor_resources(self, interval: float) -> None:
        """
        Monitor system resources.
        
        Args:
            interval (float): Sampling interval in seconds
        """
        # Get initial disk and network counters
        prev_disk_io = psutil.disk_io_counters()
        prev_net_io = psutil.net_io_counters()
        prev_time = time.time()
        
        while not self._stop_monitoring.is_set():
            # CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=None)
            memory_info = psutil.virtual_memory()
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            
            # Network I/O
            net_io = psutil.net_io_counters()
            
            # Calculate rates
            current_time = time.time()
            time_diff = current_time - prev_time
            
            # Disk I/O rates (bytes/sec)
            disk_read_rate = (disk_io.read_bytes - prev_disk_io.read_bytes) / time_diff
            disk_write_rate = (disk_io.write_bytes - prev_disk_io.write_bytes) / time_diff
            
            # Network I/O rates (bytes/sec)
            net_recv_rate = (net_io.bytes_recv - prev_net_io.bytes_recv) / time_diff
            net_sent_rate = (net_io.bytes_sent - prev_net_io.bytes_sent) / time_diff
            
            # Record metrics
            self.cpu_usage.append(cpu_percent)
            self.memory_usage.append(memory_info.percent)
            self.disk_io.append({
                "timestamp": current_time,
                "read_rate": disk_read_rate,
                "write_rate": disk_write_rate
            })
            self.network_io.append({
                "timestamp": current_time,
                "recv_rate": net_recv_rate,
                "sent_rate": net_sent_rate
            })
            
            # Update previous values
            prev_disk_io = disk_io
            prev_net_io = net_io
            prev_time = current_time
            
            # Calculate throughput
            elapsed = current_time - self.start_time
            if self.operation_counts["total"] > 0:
                throughput = self.operation_counts["total"] / elapsed
                self.throughput_data.append({
                    "timestamp": current_time,
                    "elapsed": elapsed,
                    "operations": self.operation_counts["total"],
                    "throughput": throughput
                })
            
            # Sleep for the remaining interval
            time.sleep(max(0, interval - (time.time() - current_time)))
    
    def calculate_response_time_stats(self) -> Dict[str, float]:
        """
        Calculate response time statistics.
        
        Returns:
            Dict[str, float]: Response time statistics
        """
        if not self.response_times:
            return {
                "min": 0,
                "max": 0,
                "avg": 0,
                "median": 0,
                "p90": 0,
                "p95": 0,
                "p99": 0,
                "std_dev": 0
            }
        
        response_times = np.array(self.response_times)
        
        return {
            "min": float(np.min(response_times)),
            "max": float(np.max(response_times)),
            "avg": float(np.mean(response_times)),
            "median": float(np.median(response_times)),
            "p90": float(np.percentile(response_times, 90)),
            "p95": float(np.percentile(response_times, 95)),
            "p99": float(np.percentile(response_times, 99)),
            "std_dev": float(np.std(response_times))
        }
    
    def calculate_throughput_stats(self) -> Dict[str, float]:
        """
        Calculate throughput statistics.
        
        Returns:
            Dict[str, float]: Throughput statistics
        """
        if not self.throughput_data:
            return {
                "min": 0,
                "max": 0,
                "avg": 0,
                "final": 0
            }
        
        throughputs = [data["throughput"] for data in self.throughput_data]
        
        return {
            "min": min(throughputs),
            "max": max(throughputs),
            "avg": sum(throughputs) / len(throughputs),
            "final": throughputs[-1] if throughputs else 0
        }
    
    def calculate_resource_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate resource utilization statistics.
        
        Returns:
            Dict[str, Dict[str, float]]: Resource utilization statistics
        """
        stats = {}
        
        # CPU usage
        if self.cpu_usage:
            stats["cpu"] = {
                "min": min(self.cpu_usage),
                "max": max(self.cpu_usage),
                "avg": sum(self.cpu_usage) / len(self.cpu_usage)
            }
        else:
            stats["cpu"] = {"min": 0, "max": 0, "avg": 0}
        
        # Memory usage
        if self.memory_usage:
            stats["memory"] = {
                "min": min(self.memory_usage),
                "max": max(self.memory_usage),
                "avg": sum(self.memory_usage) / len(self.memory_usage)
            }
        else:
            stats["memory"] = {"min": 0, "max": 0, "avg": 0}
        
        # Disk I/O
        if self.disk_io:
            read_rates = [data["read_rate"] for data in self.disk_io]
            write_rates = [data["write_rate"] for data in self.disk_io]
            
            stats["disk_io"] = {
                "read_min": min(read_rates),
                "read_max": max(read_rates),
                "read_avg": sum(read_rates) / len(read_rates),
                "write_min": min(write_rates),
                "write_max": max(write_rates),
                "write_avg": sum(write_rates) / len(write_rates)
            }
        else:
            stats["disk_io"] = {
                "read_min": 0, "read_max": 0, "read_avg": 0,
                "write_min": 0, "write_max": 0, "write_avg": 0
            }
        
        # Network I/O
        if self.network_io:
            recv_rates = [data["recv_rate"] for data in self.network_io]
            sent_rates = [data["sent_rate"] for data in self.network_io]
            
            stats["network_io"] = {
                "recv_min": min(recv_rates),
                "recv_max": max(recv_rates),
                "recv_avg": sum(recv_rates) / len(recv_rates),
                "sent_min": min(sent_rates),
                "sent_max": max(sent_rates),
                "sent_avg": sum(sent_rates) / len(sent_rates)
            }
        else:
            stats["network_io"] = {
                "recv_min": 0, "recv_max": 0, "recv_avg": 0,
                "sent_min": 0, "sent_max": 0, "sent_avg": 0
            }
        
        return stats
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all metrics.
        
        Returns:
            Dict[str, Any]: Summary of all metrics
        """
        duration = (self.end_time or time.time()) - (self.start_time or time.time())
        
        return {
            "test_name": self.test_name,
            "version": self.version,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
            "end_time": datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
            "duration": duration,
            "operations": self.operation_counts,
            "response_time": self.calculate_response_time_stats(),
            "throughput": self.calculate_throughput_stats(),
            "resources": self.calculate_resource_stats()
        }
    
    def to_dataframe(self) -> Dict[str, pd.DataFrame]:
        """
        Convert metrics to pandas DataFrames.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of DataFrames
        """
        dfs = {}
        
        # Response times
        if self.response_times:
            dfs["response_times"] = pd.DataFrame({
                "response_time": self.response_times
            })
        
        # Throughput
        if self.throughput_data:
            dfs["throughput"] = pd.DataFrame(self.throughput_data)
        
        # CPU and memory usage
        if self.cpu_usage and self.memory_usage:
            # Ensure same length
            min_len = min(len(self.cpu_usage), len(self.memory_usage))
            dfs["system"] = pd.DataFrame({
                "cpu_percent": self.cpu_usage[:min_len],
                "memory_percent": self.memory_usage[:min_len]
            })
        
        # Disk I/O
        if self.disk_io:
            dfs["disk_io"] = pd.DataFrame(self.disk_io)
        
        # Network I/O
        if self.network_io:
            dfs["network_io"] = pd.DataFrame(self.network_io)
        
        return dfs


class CommandListener(monitoring.CommandListener):
    """MongoDB command listener for monitoring query performance."""
    
    def __init__(self, metrics: PerformanceMetrics):
        """
        Initialize a new CommandListener instance.
        
        Args:
            metrics (PerformanceMetrics): Performance metrics instance
        """
        self.metrics = metrics
        self.start_times = {}
    
    def started(self, event):
        """Record command start time."""
        self.start_times[event.request_id] = time.time()
    
    def succeeded(self, event):
        """Record successful command execution."""
        if event.request_id in self.start_times:
            start_time = self.start_times.pop(event.request_id)
            duration = time.time() - start_time
            
            # Record response time
            self.metrics.record_response_time(duration)
            
            # Record operation type
            command_name = event.command_name.lower()
            if command_name in ["insert", "find", "update", "delete", "aggregate"]:
                self.metrics.record_operation(command_name)
            else:
                self.metrics.record_operation("command")
    
    def failed(self, event):
        """Clean up after failed command execution."""
        if event.request_id in self.start_times:
            del self.start_times[event.request_id]


def register_command_listener(metrics: PerformanceMetrics) -> None:
    """
    Register a command listener for MongoDB monitoring.
    
    Args:
        metrics (PerformanceMetrics): Performance metrics instance
    """
    listener = CommandListener(metrics)
    monitoring.register(listener)


def unregister_command_listener(listener: CommandListener) -> None:
    """
    Unregister a command listener.
    
    Args:
        listener (CommandListener): Command listener to unregister
    """
    monitoring.unregister(listener)


def measure_execution_time(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """
    Measure the execution time of a function.
    
    Args:
        func (Callable): Function to measure
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        Tuple[Any, float]: Function result and execution time
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    return result, end_time - start_time


def run_with_metrics(
    test_name: str,
    version: str,
    func: Callable,
    *args,
    **kwargs
) -> Tuple[Any, PerformanceMetrics]:
    """
    Run a function with performance metrics collection.
    
    Args:
        test_name (str): Name of the test
        version (str): MongoDB version ('v7' or 'v8')
        func (Callable): Function to run
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        Tuple[Any, PerformanceMetrics]: Function result and performance metrics
    """
    # Initialize metrics
    metrics = PerformanceMetrics(test_name, version)
    
    # Register command listener
    listener = CommandListener(metrics)
    monitoring.register(listener)
    
    # Start metrics collection
    metrics.start()
    
    try:
        # Run function
        result = func(*args, **kwargs)
        
        return result, metrics
    finally:
        # Stop metrics collection
        metrics.stop()
        
        # Unregister command listener
        monitoring.unregister(listener)