"""
Reporting module for MongoDB performance testing.
This module provides functions to generate reports and visualizations.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

from .metrics import PerformanceMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set Seaborn style
sns.set(style="whitegrid")


class PerformanceReport:
    """Class for generating performance reports and visualizations."""
    
    def __init__(
        self,
        test_name: str,
        v7_metrics: Optional[PerformanceMetrics] = None,
        v8_metrics: Optional[PerformanceMetrics] = None,
        output_dir: str = "reports"
    ):
        """
        Initialize a new PerformanceReport instance.
        
        Args:
            test_name (str): Name of the test
            v7_metrics (PerformanceMetrics, optional): MongoDB v7.0 metrics
            v8_metrics (PerformanceMetrics, optional): MongoDB v8.0 metrics
            output_dir (str): Output directory for reports
        """
        self.test_name = test_name
        self.v7_metrics = v7_metrics
        self.v8_metrics = v8_metrics
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def set_metrics(self, version: str, metrics: PerformanceMetrics) -> None:
        """
        Set metrics for a specific MongoDB version.
        
        Args:
            version (str): MongoDB version ('v7' or 'v8')
            metrics (PerformanceMetrics): Performance metrics
        """
        if version == 'v7':
            self.v7_metrics = metrics
        elif version == 'v8':
            self.v8_metrics = metrics
        else:
            raise ValueError("Version must be 'v7' or 'v8'")
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generate a summary report comparing MongoDB v7.0 and v8.0.
        
        Returns:
            Dict[str, Any]: Summary report
        """
        summary = {
            "test_name": self.test_name,
            "timestamp": self.timestamp,
            "mongodb_v7": self.v7_metrics.get_summary() if self.v7_metrics else None,
            "mongodb_v8": self.v8_metrics.get_summary() if self.v8_metrics else None
        }
        
        # Calculate performance improvements if both metrics are available
        if self.v7_metrics and self.v8_metrics:
            improvements = self._calculate_improvements(
                self.v7_metrics.get_summary(),
                self.v8_metrics.get_summary()
            )
            summary["improvements"] = improvements
        
        return summary
    
    def _calculate_improvements(
        self,
        v7_summary: Dict[str, Any],
        v8_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate performance improvements from v7.0 to v8.0.
        
        Args:
            v7_summary (Dict[str, Any]): MongoDB v7.0 summary
            v8_summary (Dict[str, Any]): MongoDB v8.0 summary
            
        Returns:
            Dict[str, Any]: Performance improvements
        """
        improvements = {}
        
        # Response time improvements
        v7_rt = v7_summary["response_time"]
        v8_rt = v8_summary["response_time"]
        
        rt_improvements = {}
        for metric in ["avg", "median", "p90", "p95", "p99"]:
            if v7_rt[metric] > 0:
                improvement = ((v7_rt[metric] - v8_rt[metric]) / v7_rt[metric]) * 100
                rt_improvements[metric] = improvement
        
        improvements["response_time"] = rt_improvements
        
        # Throughput improvements
        v7_tp = v7_summary["throughput"]
        v8_tp = v8_summary["throughput"]
        
        tp_improvements = {}
        for metric in ["avg", "max", "final"]:
            if v7_tp[metric] > 0:
                improvement = ((v8_tp[metric] - v7_tp[metric]) / v7_tp[metric]) * 100
                tp_improvements[metric] = improvement
        
        improvements["throughput"] = tp_improvements
        
        # Resource utilization improvements
        v7_res = v7_summary["resources"]
        v8_res = v8_summary["resources"]
        
        res_improvements = {}
        
        # CPU usage (lower is better)
        if v7_res["cpu"]["avg"] > 0:
            cpu_improvement = ((v7_res["cpu"]["avg"] - v8_res["cpu"]["avg"]) / v7_res["cpu"]["avg"]) * 100
            res_improvements["cpu_avg"] = cpu_improvement
        
        # Memory usage (lower is better)
        if v7_res["memory"]["avg"] > 0:
            memory_improvement = ((v7_res["memory"]["avg"] - v8_res["memory"]["avg"]) / v7_res["memory"]["avg"]) * 100
            res_improvements["memory_avg"] = memory_improvement
        
        # Disk I/O (lower is better for write, higher for read)
        if v7_res["disk_io"]["write_avg"] > 0:
            disk_write_improvement = ((v7_res["disk_io"]["write_avg"] - v8_res["disk_io"]["write_avg"]) / v7_res["disk_io"]["write_avg"]) * 100
            res_improvements["disk_write_avg"] = disk_write_improvement
        
        if v7_res["disk_io"]["read_avg"] > 0:
            disk_read_improvement = ((v8_res["disk_io"]["read_avg"] - v7_res["disk_io"]["read_avg"]) / v7_res["disk_io"]["read_avg"]) * 100
            res_improvements["disk_read_avg"] = disk_read_improvement
        
        improvements["resources"] = res_improvements
        
        # Overall improvement (based on response time and throughput)
        if "avg" in rt_improvements and "avg" in tp_improvements:
            overall_improvement = (rt_improvements["avg"] + tp_improvements["avg"]) / 2
            improvements["overall"] = overall_improvement
        
        return improvements
    
    def save_summary_report(self, formats: List[str] = ["json", "csv"]) -> Dict[str, str]:
        """
        Save the summary report to files.
        
        Args:
            formats (List[str]): Output formats
            
        Returns:
            Dict[str, str]: Dictionary with file paths
        """
        summary = self.generate_summary_report()
        file_paths = {}
        
        # Create test directory
        test_dir = os.path.join(self.output_dir, f"{self.test_name}_{self.timestamp}")
        os.makedirs(test_dir, exist_ok=True)
        
        # Save as JSON
        if "json" in formats:
            json_path = os.path.join(test_dir, "summary.json")
            with open(json_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            file_paths["json"] = json_path
        
        # Save as CSV
        if "csv" in formats:
            # Convert to DataFrame
            df_data = []
            
            # Add v7 data
            if self.v7_metrics:
                v7_summary = self.v7_metrics.get_summary()
                v7_row = {
                    "version": "v7",
                    "duration": v7_summary["duration"],
                    "total_operations": v7_summary["operations"]["total"],
                    "response_time_avg": v7_summary["response_time"]["avg"],
                    "response_time_p95": v7_summary["response_time"]["p95"],
                    "throughput_avg": v7_summary["throughput"]["avg"],
                    "cpu_avg": v7_summary["resources"]["cpu"]["avg"],
                    "memory_avg": v7_summary["resources"]["memory"]["avg"]
                }
                df_data.append(v7_row)
            
            # Add v8 data
            if self.v8_metrics:
                v8_summary = self.v8_metrics.get_summary()
                v8_row = {
                    "version": "v8",
                    "duration": v8_summary["duration"],
                    "total_operations": v8_summary["operations"]["total"],
                    "response_time_avg": v8_summary["response_time"]["avg"],
                    "response_time_p95": v8_summary["response_time"]["p95"],
                    "throughput_avg": v8_summary["throughput"]["avg"],
                    "cpu_avg": v8_summary["resources"]["cpu"]["avg"],
                    "memory_avg": v8_summary["resources"]["memory"]["avg"]
                }
                df_data.append(v8_row)
            
            # Add improvement data
            if "improvements" in summary:
                imp = summary["improvements"]
                imp_row = {
                    "version": "improvement_pct",
                    "response_time_avg": imp["response_time"].get("avg", 0),
                    "response_time_p95": imp["response_time"].get("p95", 0),
                    "throughput_avg": imp["throughput"].get("avg", 0),
                    "cpu_avg": imp["resources"].get("cpu_avg", 0),
                    "memory_avg": imp["resources"].get("memory_avg", 0),
                    "overall": imp.get("overall", 0)
                }
                df_data.append(imp_row)
            
            # Create DataFrame and save
            df = pd.DataFrame(df_data)
            csv_path = os.path.join(test_dir, "summary.csv")
            df.to_csv(csv_path, index=False)
            file_paths["csv"] = csv_path
        
        # Save as HTML
        if "html" in formats:
            # Create a simple HTML report
            html_content = f"""
            <html>
            <head>
                <title>MongoDB Performance Test: {self.test_name}</title>
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
                <h1>MongoDB Performance Test Report</h1>
                <p><strong>Test:</strong> {self.test_name}</p>
                <p><strong>Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                
                <h2>Summary</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>MongoDB v7.0</th>
                        <th>MongoDB v8.0</th>
                        <th>Improvement</th>
                    </tr>
            """
            
            # Add metrics to HTML
            if self.v7_metrics and self.v8_metrics:
                v7_summary = self.v7_metrics.get_summary()
                v8_summary = self.v8_metrics.get_summary()
                improvements = summary.get("improvements", {})
                
                # Duration
                html_content += f"""
                    <tr>
                        <td>Duration (s)</td>
                        <td>{v7_summary["duration"]:.2f}</td>
                        <td>{v8_summary["duration"]:.2f}</td>
                        <td></td>
                    </tr>
                """
                
                # Total operations
                html_content += f"""
                    <tr>
                        <td>Total Operations</td>
                        <td>{v7_summary["operations"]["total"]}</td>
                        <td>{v8_summary["operations"]["total"]}</td>
                        <td></td>
                    </tr>
                """
                
                # Response time
                rt_imp = improvements.get("response_time", {}).get("avg", 0)
                rt_class = "improvement-positive" if rt_imp > 0 else "improvement-negative"
                html_content += f"""
                    <tr>
                        <td>Avg Response Time (s)</td>
                        <td>{v7_summary["response_time"]["avg"]:.6f}</td>
                        <td>{v8_summary["response_time"]["avg"]:.6f}</td>
                        <td class="{rt_class}">{rt_imp:.2f}%</td>
                    </tr>
                """
                
                # Throughput
                tp_imp = improvements.get("throughput", {}).get("avg", 0)
                tp_class = "improvement-positive" if tp_imp > 0 else "improvement-negative"
                html_content += f"""
                    <tr>
                        <td>Avg Throughput (ops/s)</td>
                        <td>{v7_summary["throughput"]["avg"]:.2f}</td>
                        <td>{v8_summary["throughput"]["avg"]:.2f}</td>
                        <td class="{tp_class}">{tp_imp:.2f}%</td>
                    </tr>
                """
                
                # CPU usage
                cpu_imp = improvements.get("resources", {}).get("cpu_avg", 0)
                cpu_class = "improvement-positive" if cpu_imp > 0 else "improvement-negative"
                html_content += f"""
                    <tr>
                        <td>Avg CPU Usage (%)</td>
                        <td>{v7_summary["resources"]["cpu"]["avg"]:.2f}</td>
                        <td>{v8_summary["resources"]["cpu"]["avg"]:.2f}</td>
                        <td class="{cpu_class}">{cpu_imp:.2f}%</td>
                    </tr>
                """
                
                # Memory usage
                mem_imp = improvements.get("resources", {}).get("memory_avg", 0)
                mem_class = "improvement-positive" if mem_imp > 0 else "improvement-negative"
                html_content += f"""
                    <tr>
                        <td>Avg Memory Usage (%)</td>
                        <td>{v7_summary["resources"]["memory"]["avg"]:.2f}</td>
                        <td>{v8_summary["resources"]["memory"]["avg"]:.2f}</td>
                        <td class="{mem_class}">{mem_imp:.2f}%</td>
                    </tr>
                """
                
                # Overall improvement
                overall_imp = improvements.get("overall", 0)
                overall_class = "improvement-positive" if overall_imp > 0 else "improvement-negative"
                html_content += f"""
                    <tr>
                        <td><strong>Overall Improvement</strong></td>
                        <td></td>
                        <td></td>
                        <td class="{overall_class}"><strong>{overall_imp:.2f}%</strong></td>
                    </tr>
                """
            
            html_content += """
                </table>
            </body>
            </html>
            """
            
            html_path = os.path.join(test_dir, "summary.html")
            with open(html_path, "w") as f:
                f.write(html_content)
            file_paths["html"] = html_path
        
        return file_paths
    
    def generate_charts(self) -> Dict[str, Figure]:
        """
        Generate charts comparing MongoDB v7.0 and v8.0.
        
        Returns:
            Dict[str, Figure]: Dictionary with chart figures
        """
        if not self.v7_metrics or not self.v8_metrics:
            logger.warning("Both v7 and v8 metrics are required to generate charts")
            return {}
        
        charts = {}
        
        # Response time distribution
        fig_rt, ax_rt = plt.subplots(figsize=(10, 6))
        
        if self.v7_metrics.response_times:
            sns.histplot(self.v7_metrics.response_times, kde=True, label="MongoDB v7.0", alpha=0.6, ax=ax_rt)
        
        if self.v8_metrics.response_times:
            sns.histplot(self.v8_metrics.response_times, kde=True, label="MongoDB v8.0", alpha=0.6, ax=ax_rt)
        
        ax_rt.set_title(f"Response Time Distribution - {self.test_name}")
        ax_rt.set_xlabel("Response Time (s)")
        ax_rt.set_ylabel("Frequency")
        ax_rt.legend()
        
        charts["response_time_distribution"] = fig_rt
        
        # Response time percentiles
        fig_rtp, ax_rtp = plt.subplots(figsize=(10, 6))
        
        v7_summary = self.v7_metrics.get_summary()
        v8_summary = self.v8_metrics.get_summary()
        
        percentiles = ["min", "avg", "median", "p90", "p95", "p99", "max"]
        v7_values = [v7_summary["response_time"][p] for p in percentiles]
        v8_values = [v8_summary["response_time"][p] for p in percentiles]
        
        x = np.arange(len(percentiles))
        width = 0.35
        
        ax_rtp.bar(x - width/2, v7_values, width, label="MongoDB v7.0")
        ax_rtp.bar(x + width/2, v8_values, width, label="MongoDB v8.0")
        
        ax_rtp.set_title(f"Response Time Percentiles - {self.test_name}")
        ax_rtp.set_xlabel("Percentile")
        ax_rtp.set_ylabel("Response Time (s)")
        ax_rtp.set_xticks(x)
        ax_rtp.set_xticklabels(percentiles)
        ax_rtp.legend()
        
        charts["response_time_percentiles"] = fig_rtp
        
        # Throughput over time
        fig_tp, ax_tp = plt.subplots(figsize=(10, 6))
        
        if self.v7_metrics.throughput_data:
            v7_df = pd.DataFrame(self.v7_metrics.throughput_data)
            ax_tp.plot(v7_df["elapsed"], v7_df["throughput"], label="MongoDB v7.0")
        
        if self.v8_metrics.throughput_data:
            v8_df = pd.DataFrame(self.v8_metrics.throughput_data)
            ax_tp.plot(v8_df["elapsed"], v8_df["throughput"], label="MongoDB v8.0")
        
        ax_tp.set_title(f"Throughput Over Time - {self.test_name}")
        ax_tp.set_xlabel("Elapsed Time (s)")
        ax_tp.set_ylabel("Throughput (ops/s)")
        ax_tp.legend()
        
        charts["throughput_over_time"] = fig_tp
        
        # CPU usage over time
        fig_cpu, ax_cpu = plt.subplots(figsize=(10, 6))
        
        if self.v7_metrics.cpu_usage:
            ax_cpu.plot(range(len(self.v7_metrics.cpu_usage)), self.v7_metrics.cpu_usage, label="MongoDB v7.0")
        
        if self.v8_metrics.cpu_usage:
            ax_cpu.plot(range(len(self.v8_metrics.cpu_usage)), self.v8_metrics.cpu_usage, label="MongoDB v8.0")
        
        ax_cpu.set_title(f"CPU Usage Over Time - {self.test_name}")
        ax_cpu.set_xlabel("Sample")
        ax_cpu.set_ylabel("CPU Usage (%)")
        ax_cpu.legend()
        
        charts["cpu_usage_over_time"] = fig_cpu
        
        # Memory usage over time
        fig_mem, ax_mem = plt.subplots(figsize=(10, 6))
        
        if self.v7_metrics.memory_usage:
            ax_mem.plot(range(len(self.v7_metrics.memory_usage)), self.v7_metrics.memory_usage, label="MongoDB v7.0")
        
        if self.v8_metrics.memory_usage:
            ax_mem.plot(range(len(self.v8_metrics.memory_usage)), self.v8_metrics.memory_usage, label="MongoDB v8.0")
        
        ax_mem.set_title(f"Memory Usage Over Time - {self.test_name}")
        ax_mem.set_xlabel("Sample")
        ax_mem.set_ylabel("Memory Usage (%)")
        ax_mem.legend()
        
        charts["memory_usage_over_time"] = fig_mem
        
        # Performance improvement summary
        fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
        
        improvements = self._calculate_improvements(v7_summary, v8_summary)
        
        metrics = [
            "Response Time (avg)",
            "Response Time (p95)",
            "Throughput (avg)",
            "CPU Usage",
            "Memory Usage",
            "Overall"
        ]
        
        values = [
            improvements["response_time"].get("avg", 0),
            improvements["response_time"].get("p95", 0),
            improvements["throughput"].get("avg", 0),
            improvements["resources"].get("cpu_avg", 0),
            improvements["resources"].get("memory_avg", 0),
            improvements.get("overall", 0)
        ]
        
        colors = ["green" if v > 0 else "red" for v in values]
        
        ax_imp.barh(metrics, values, color=colors)
        ax_imp.axvline(x=0, color="black", linestyle="-", alpha=0.3)
        
        ax_imp.set_title(f"Performance Improvement (MongoDB v8.0 vs v7.0) - {self.test_name}")
        ax_imp.set_xlabel("Improvement (%)")
        
        charts["performance_improvement"] = fig_imp
        
        return charts
    
    def save_charts(self, charts: Dict[str, Figure] = None) -> Dict[str, str]:
        """
        Save charts to files.
        
        Args:
            charts (Dict[str, Figure], optional): Dictionary with chart figures
            
        Returns:
            Dict[str, str]: Dictionary with file paths
        """
        if charts is None:
            charts = self.generate_charts()
        
        if not charts:
            return {}
        
        # Create test directory
        test_dir = os.path.join(self.output_dir, f"{self.test_name}_{self.timestamp}")
        charts_dir = os.path.join(test_dir, "charts")
        os.makedirs(charts_dir, exist_ok=True)
        
        file_paths = {}
        
        for chart_name, fig in charts.items():
            chart_path = os.path.join(charts_dir, f"{chart_name}.png")
            fig.savefig(chart_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            file_paths[chart_name] = chart_path
        
        return file_paths
    
    def generate_report(self, formats: List[str] = ["json", "csv", "html"], save_charts: bool = True) -> Dict[str, str]:
        """
        Generate a complete report with summary and charts.
        
        Args:
            formats (List[str]): Output formats
            save_charts (bool): Whether to save charts
            
        Returns:
            Dict[str, str]: Dictionary with file paths
        """
        file_paths = {}
        
        # Save summary report
        summary_paths = self.save_summary_report(formats)
        file_paths.update(summary_paths)
        
        # Save charts
        if save_charts:
            charts = self.generate_charts()
            chart_paths = self.save_charts(charts)
            file_paths.update(chart_paths)
        
        return file_paths


def compare_results(
    test_name: str,
    v7_metrics: PerformanceMetrics,
    v8_metrics: PerformanceMetrics,
    output_dir: str = "reports",
    formats: List[str] = ["json", "csv", "html"],
    save_charts: bool = True
) -> Dict[str, str]:
    """
    Compare performance results between MongoDB v7.0 and v8.0.
    
    Args:
        test_name (str): Name of the test
        v7_metrics (PerformanceMetrics): MongoDB v7.0 metrics
        v8_metrics (PerformanceMetrics): MongoDB v8.0 metrics
        output_dir (str): Output directory for reports
        formats (List[str]): Output formats
        save_charts (bool): Whether to save charts
        
    Returns:
        Dict[str, str]: Dictionary with file paths
    """
    report = PerformanceReport(test_name, v7_metrics, v8_metrics, output_dir)
    return report.generate_report(formats, save_charts)