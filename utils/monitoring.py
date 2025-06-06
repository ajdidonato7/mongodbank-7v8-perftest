"""
Resource monitoring module for MongoDB performance testing.
This module provides functions to monitor system resource utilization.
"""

import time
import threading
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
import psutil
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ResourceMonitor:
    """Class for monitoring system resource utilization."""
    
    def __init__(self, interval: float = 1.0):
        """
        Initialize a new ResourceMonitor instance.
        
        Args:
            interval (float): Sampling interval in seconds
        """
        self.interval = interval
        self.cpu_usage = []
        self.memory_usage = []
        self.disk_io = []
        self.network_io = []
        self.timestamps = []
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
    
    def start(self) -> None:
        """Start monitoring resources."""
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitor_resources
        )
        self._monitoring_thread.daemon = True
        self._monitoring_thread.start()
        logger.info("Resource monitoring started")
    
    def stop(self) -> None:
        """Stop monitoring resources."""
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join()
        logger.info("Resource monitoring stopped")
    
    def _monitor_resources(self) -> None:
        """Monitor system resources."""
        # Get initial disk and network counters
        prev_disk_io = psutil.disk_io_counters()
        prev_net_io = psutil.net_io_counters()
        prev_time = time.time()
        
        while not self._stop_monitoring.is_set():
            # Record timestamp
            current_time = time.time()
            self.timestamps.append(current_time)
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            self.cpu_usage.append(cpu_percent)
            
            # Memory usage
            memory_info = psutil.virtual_memory()
            self.memory_usage.append({
                "total": memory_info.total,
                "available": memory_info.available,
                "used": memory_info.used,
                "percent": memory_info.percent
            })
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            time_diff = current_time - prev_time
            
            # Calculate rates (bytes/sec)
            if time_diff > 0:
                disk_read_rate = (disk_io.read_bytes - prev_disk_io.read_bytes) / time_diff
                disk_write_rate = (disk_io.write_bytes - prev_disk_io.write_bytes) / time_diff
            else:
                disk_read_rate = 0
                disk_write_rate = 0
            
            self.disk_io.append({
                "read_bytes": disk_io.read_bytes,
                "write_bytes": disk_io.write_bytes,
                "read_rate": disk_read_rate,
                "write_rate": disk_write_rate
            })
            
            # Network I/O
            net_io = psutil.net_io_counters()
            
            # Calculate rates (bytes/sec)
            if time_diff > 0:
                net_recv_rate = (net_io.bytes_recv - prev_net_io.bytes_recv) / time_diff
                net_sent_rate = (net_io.bytes_sent - prev_net_io.bytes_sent) / time_diff
            else:
                net_recv_rate = 0
                net_sent_rate = 0
            
            self.network_io.append({
                "bytes_recv": net_io.bytes_recv,
                "bytes_sent": net_io.bytes_sent,
                "recv_rate": net_recv_rate,
                "sent_rate": net_sent_rate
            })
            
            # Update previous values
            prev_disk_io = disk_io
            prev_net_io = net_io
            prev_time = current_time
            
            # Sleep for the remaining interval
            time.sleep(max(0, self.interval - (time.time() - current_time)))
    
    def get_cpu_stats(self) -> Dict[str, float]:
        """
        Get CPU usage statistics.
        
        Returns:
            Dict[str, float]: CPU usage statistics
        """
        if not self.cpu_usage:
            return {"min": 0, "max": 0, "avg": 0}
        
        return {
            "min": min(self.cpu_usage),
            "max": max(self.cpu_usage),
            "avg": sum(self.cpu_usage) / len(self.cpu_usage)
        }
    
    def get_memory_stats(self) -> Dict[str, float]:
        """
        Get memory usage statistics.
        
        Returns:
            Dict[str, float]: Memory usage statistics
        """
        if not self.memory_usage:
            return {"min": 0, "max": 0, "avg": 0}
        
        memory_percent = [m["percent"] for m in self.memory_usage]
        
        return {
            "min": min(memory_percent),
            "max": max(memory_percent),
            "avg": sum(memory_percent) / len(memory_percent)
        }
    
    def get_disk_io_stats(self) -> Dict[str, float]:
        """
        Get disk I/O statistics.
        
        Returns:
            Dict[str, float]: Disk I/O statistics
        """
        if not self.disk_io:
            return {
                "read_min": 0, "read_max": 0, "read_avg": 0,
                "write_min": 0, "write_max": 0, "write_avg": 0
            }
        
        read_rates = [d["read_rate"] for d in self.disk_io]
        write_rates = [d["write_rate"] for d in self.disk_io]
        
        return {
            "read_min": min(read_rates),
            "read_max": max(read_rates),
            "read_avg": sum(read_rates) / len(read_rates),
            "write_min": min(write_rates),
            "write_max": max(write_rates),
            "write_avg": sum(write_rates) / len(write_rates)
        }
    
    def get_network_io_stats(self) -> Dict[str, float]:
        """
        Get network I/O statistics.
        
        Returns:
            Dict[str, float]: Network I/O statistics
        """
        if not self.network_io:
            return {
                "recv_min": 0, "recv_max": 0, "recv_avg": 0,
                "sent_min": 0, "sent_max": 0, "sent_avg": 0
            }
        
        recv_rates = [n["recv_rate"] for n in self.network_io]
        sent_rates = [n["sent_rate"] for n in self.network_io]
        
        return {
            "recv_min": min(recv_rates),
            "recv_max": max(recv_rates),
            "recv_avg": sum(recv_rates) / len(recv_rates),
            "sent_min": min(sent_rates),
            "sent_max": max(sent_rates),
            "sent_avg": sum(sent_rates) / len(sent_rates)
        }
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get a summary of all resource statistics.
        
        Returns:
            Dict[str, Dict[str, float]]: Summary of all resource statistics
        """
        return {
            "cpu": self.get_cpu_stats(),
            "memory": self.get_memory_stats(),
            "disk_io": self.get_disk_io_stats(),
            "network_io": self.get_network_io_stats()
        }
    
    def to_dataframe(self) -> Dict[str, pd.DataFrame]:
        """
        Convert monitoring data to pandas DataFrames.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of DataFrames
        """
        dfs = {}
        
        # Create timestamp series
        timestamps = [datetime.fromtimestamp(ts) for ts in self.timestamps]
        
        # CPU usage
        if self.cpu_usage:
            dfs["cpu"] = pd.DataFrame({
                "timestamp": timestamps,
                "cpu_percent": self.cpu_usage
            })
        
        # Memory usage
        if self.memory_usage:
            memory_df = pd.DataFrame({
                "timestamp": timestamps,
                "total": [m["total"] for m in self.memory_usage],
                "available": [m["available"] for m in self.memory_usage],
                "used": [m["used"] for m in self.memory_usage],
                "percent": [m["percent"] for m in self.memory_usage]
            })
            dfs["memory"] = memory_df
        
        # Disk I/O
        if self.disk_io:
            disk_df = pd.DataFrame({
                "timestamp": timestamps,
                "read_bytes": [d["read_bytes"] for d in self.disk_io],
                "write_bytes": [d["write_bytes"] for d in self.disk_io],
                "read_rate": [d["read_rate"] for d in self.disk_io],
                "write_rate": [d["write_rate"] for d in self.disk_io]
            })
            dfs["disk_io"] = disk_df
        
        # Network I/O
        if self.network_io:
            network_df = pd.DataFrame({
                "timestamp": timestamps,
                "bytes_recv": [n["bytes_recv"] for n in self.network_io],
                "bytes_sent": [n["bytes_sent"] for n in self.network_io],
                "recv_rate": [n["recv_rate"] for n in self.network_io],
                "sent_rate": [n["sent_rate"] for n in self.network_io]
            })
            dfs["network_io"] = network_df
        
        return dfs
    
    def save_to_csv(self, output_dir: str, prefix: str = "") -> Dict[str, str]:
        """
        Save monitoring data to CSV files.
        
        Args:
            output_dir (str): Output directory
            prefix (str): Prefix for file names
            
        Returns:
            Dict[str, str]: Dictionary with file paths
        """
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        file_paths = {}
        dfs = self.to_dataframe()
        
        for name, df in dfs.items():
            file_name = f"{prefix}_{name}.csv" if prefix else f"{name}.csv"
            file_path = os.path.join(output_dir, file_name)
            df.to_csv(file_path, index=False)
            file_paths[name] = file_path
        
        return file_paths


def monitor_function(
    func: Callable,
    *args,
    interval: float = 1.0,
    **kwargs
) -> Tuple[Any, ResourceMonitor]:
    """
    Monitor system resources while executing a function.
    
    Args:
        func (Callable): Function to execute
        *args: Positional arguments for the function
        interval (float): Sampling interval in seconds
        **kwargs: Keyword arguments for the function
        
    Returns:
        Tuple[Any, ResourceMonitor]: Function result and resource monitor
    """
    # Initialize resource monitor
    monitor = ResourceMonitor(interval=interval)
    
    # Start monitoring
    monitor.start()
    
    try:
        # Execute function
        result = func(*args, **kwargs)
        
        return result, monitor
    finally:
        # Stop monitoring
        monitor.stop()


def get_mongodb_process_info() -> List[Dict[str, Any]]:
    """
    Get information about MongoDB processes.
    
    Returns:
        List[Dict[str, Any]]: List of MongoDB process information
    """
    mongo_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent']):
        try:
            # Check if this is a MongoDB process
            if proc.info['name'] and ('mongod' in proc.info['name'] or 'mongo' in proc.info['name']):
                # Get process details
                proc_info = proc.as_dict(attrs=['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent'])
                
                # Get additional information
                with proc.oneshot():
                    proc_info['create_time'] = datetime.fromtimestamp(proc.create_time()).strftime('%Y-%m-%d %H:%M:%S')
                    proc_info['status'] = proc.status()
                    proc_info['num_threads'] = proc.num_threads()
                    
                    # Get I/O counters if available
                    try:
                        io_counters = proc.io_counters()
                        proc_info['io_read_bytes'] = io_counters.read_bytes
                        proc_info['io_write_bytes'] = io_counters.write_bytes
                    except (psutil.AccessDenied, AttributeError):
                        pass
                    
                    # Get open files if available
                    try:
                        proc_info['open_files'] = len(proc.open_files())
                    except (psutil.AccessDenied, AttributeError):
                        pass
                    
                    # Get connections if available
                    try:
                        proc_info['connections'] = len(proc.connections())
                    except (psutil.AccessDenied, AttributeError):
                        pass
                
                mongo_processes.append(proc_info)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    return mongo_processes


def monitor_mongodb_processes(interval: float = 5.0, duration: float = 60.0) -> pd.DataFrame:
    """
    Monitor MongoDB processes for a specified duration.
    
    Args:
        interval (float): Sampling interval in seconds
        duration (float): Monitoring duration in seconds
        
    Returns:
        pd.DataFrame: DataFrame with MongoDB process information
    """
    start_time = time.time()
    end_time = start_time + duration
    
    data = []
    
    while time.time() < end_time:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for proc_info in get_mongodb_process_info():
            proc_data = {
                'timestamp': timestamp,
                'pid': proc_info['pid'],
                'name': proc_info['name'],
                'cpu_percent': proc_info['cpu_percent'],
                'memory_percent': proc_info['memory_percent'],
                'status': proc_info.get('status', 'unknown'),
                'num_threads': proc_info.get('num_threads', 0)
            }
            
            # Add I/O information if available
            if 'io_read_bytes' in proc_info:
                proc_data['io_read_bytes'] = proc_info['io_read_bytes']
            
            if 'io_write_bytes' in proc_info:
                proc_data['io_write_bytes'] = proc_info['io_write_bytes']
            
            # Add connection information if available
            if 'connections' in proc_info:
                proc_data['connections'] = proc_info['connections']
            
            data.append(proc_data)
        
        # Sleep for the interval
        time.sleep(interval)
    
    return pd.DataFrame(data)


class PerformanceMetrics:
    """Class for tracking performance metrics of MongoDB operations."""
    
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
            "aggregate": 0,
            "other": 0
        }
        self.resource_monitor = ResourceMonitor()
    
    def start(self) -> None:
        """Start collecting metrics."""
        self.start_time = time.time()
        self.resource_monitor.start()
        logger.info(f"Started metrics collection for {self.test_name} on MongoDB {self.version}")
    
    def stop(self) -> None:
        """Stop collecting metrics."""
        self.end_time = time.time()
        self.resource_monitor.stop()
        logger.info(f"Stopped metrics collection for {self.test_name} on MongoDB {self.version}")
    
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
            self.operations["other"] += count
    
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
        if self.start_time is None:
            return 0.0
        
        end_time = self.end_time if self.end_time is not None else time.time()
        return end_time - self.start_time
    
    def get_throughput(self) -> float:
        """
        Get the throughput (operations per second).
        
        Returns:
            float: Throughput in operations per second
        """
        total_time = self.get_total_time()
        if total_time == 0:
            return 0.0
        
        return self.get_total_operations() / total_time
    
    def get_response_time_stats(self) -> Dict[str, float]:
        """
        Get response time statistics.
        
        Returns:
            Dict[str, float]: Response time statistics
        """
        if not self.response_times:
            return {
                "min": 0.0,
                "max": 0.0,
                "avg": 0.0,
                "p50": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p99": 0.0
            }
        
        # Calculate percentiles
        sorted_times = sorted(self.response_times)
        p50_idx = int(len(sorted_times) * 0.5)
        p90_idx = int(len(sorted_times) * 0.9)
        p95_idx = int(len(sorted_times) * 0.95)
        p99_idx = int(len(sorted_times) * 0.99)
        
        return {
            "min": min(self.response_times),
            "max": max(self.response_times),
            "avg": sum(self.response_times) / len(self.response_times),
            "p50": sorted_times[p50_idx],
            "p90": sorted_times[p90_idx],
            "p95": sorted_times[p95_idx],
            "p99": sorted_times[p99_idx]
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
            "response_time": self.get_response_time_stats(),
            "resources": self.resource_monitor.get_summary()
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
        
        # Operations
        dfs["operations"] = pd.DataFrame({
            "operation_type": list(self.operations.keys()),
            "count": list(self.operations.values())
        })
        
        # Resource monitoring
        resource_dfs = self.resource_monitor.to_dataframe()
        for name, df in resource_dfs.items():
            dfs[f"resource_{name}"] = df
        
        return dfs
    
    def save_to_csv(self, output_dir: str, prefix: str = "") -> Dict[str, str]:
        """
        Save metrics to CSV files.
        
        Args:
            output_dir (str): Output directory
            prefix (str): Prefix for file names
            
        Returns:
            Dict[str, str]: Dictionary with file paths
        """
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        file_paths = {}
        dfs = self.to_dataframe()
        
        for name, df in dfs.items():
            file_name = f"{prefix}_{name}.csv" if prefix else f"{name}.csv"
            file_path = os.path.join(output_dir, file_name)
            df.to_csv(file_path, index=False)
            file_paths[name] = file_path
        
        # Save summary as JSON
        import json
        summary = self.get_summary()
        summary_file_name = f"{prefix}_summary.json" if prefix else "summary.json"
        summary_file_path = os.path.join(output_dir, summary_file_name)
        
        with open(summary_file_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        file_paths["summary"] = summary_file_path
        
        return file_paths