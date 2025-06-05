"""
Aggregation performance tests for MongoDB.
This module provides functions to test aggregation performance.
"""

import logging
import time
import random
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta

from pymongo import MongoClient
from bson import ObjectId

from ..config.connection import get_database, close_connections
from ..config.test_config import TEST_PARAMETERS, COLLECTIONS
from ..data_generation import get_sample_ids
from ..utils import (
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


def test_simple_aggregation(
    version: str,
    collection_name: str,
    num_aggregations: int = TEST_PARAMETERS["aggregation"]["simple_count"],
    iterations: int = TEST_PARAMETERS["aggregation"]["iterations"]
) -> Tuple[Dict[str, Any], PerformanceMetrics]:
    """
    Test simple aggregation performance.
    
    Args:
        version (str): MongoDB version ('v7' or 'v8')
        collection_name (str): Collection name
        num_aggregations (int): Number of aggregations per iteration
        iterations (int): Number of iterations
        
    Returns:
        Tuple[Dict[str, Any], PerformanceMetrics]: Test results and performance metrics
    """
    test_name = f"simple_aggregation_{collection_name}"
    logger.info(f"Running {test_name} test for MongoDB {version}")
    
    # Get database and collection
    db = get_database(version)
    collection = db[collection_name]
    
    # Define simple aggregation pipelines based on collection
    pipelines = []
    
    if collection_name == "customers":
        # Simple customer aggregations
        pipelines.append([{"$count": "total_customers"}])
        pipelines.append([{"$group": {"_id": "$address.state", "count": {"$sum": 1}}}])
        pipelines.append([{"$group": {"_id": None, "avg_credit_score": {"$avg": "$credit_score"}}}])
        pipelines.append([{"$sort": {"credit_score": -1}}, {"$limit": 10}])
    
    elif collection_name == "accounts":
        # Simple account aggregations
        pipelines.append([{"$count": "total_accounts"}])
        pipelines.append([{"$group": {"_id": "$account_type", "count": {"$sum": 1}}}])
        pipelines.append([{"$group": {"_id": "$status", "count": {"$sum": 1}}}])
        pipelines.append([{"$group": {"_id": None, "total_balance": {"$sum": "$balance"}}}])
        pipelines.append([{"$group": {"_id": None, "avg_balance": {"$avg": "$balance"}}}])
    
    elif collection_name == "transactions":
        # Simple transaction aggregations
        pipelines.append([{"$count": "total_transactions"}])
        pipelines.append([{"$group": {"_id": "$transaction_type", "count": {"$sum": 1}}}])
        pipelines.append([{"$group": {"_id": "$status", "count": {"$sum": 1}}}])
        pipelines.append([{"$group": {"_id": "$category", "count": {"$sum": 1}}}])
        pipelines.append([{"$group": {"_id": None, "total_amount": {"$sum": "$amount"}}}])
        pipelines.append([{"$group": {"_id": None, "avg_amount": {"$avg": "$amount"}}}])
    
    elif collection_name == "loans":
        # Simple loan aggregations
        pipelines.append([{"$count": "total_loans"}])
        pipelines.append([{"$group": {"_id": "$loan_type", "count": {"$sum": 1}}}])
        pipelines.append([{"$group": {"_id": "$status", "count": {"$sum": 1}}}])
        pipelines.append([{"$group": {"_id": None, "total_amount": {"$sum": "$amount"}}}])
        pipelines.append([{"$group": {"_id": None, "avg_amount": {"$avg": "$amount"}}}])
        pipelines.append([{"$group": {"_id": None, "avg_interest_rate": {"$avg": "$interest_rate"}}}])
    
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
        "num_aggregations": num_aggregations,
        "iterations": iterations,
        "total_aggregations": 0,
        "total_time": 0.0,
        "avg_throughput": 0.0
    }
    
    try:
        # Run test iterations
        for i in range(iterations):
            logger.info(f"Iteration {i+1}/{iterations}")
            
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
            
            logger.info(
                f"Completed {aggregations_completed} simple aggregations in {iteration_time:.2f}s "
                f"({aggregations_completed/iteration_time:.2f} aggs/s)"
            )
        
        # Calculate average throughput
        if results["total_time"] > 0:
            results["avg_throughput"] = results["total_aggregations"] / results["total_time"]
        
        return results, metrics
    
    finally:
        # Stop metrics collection
        metrics.stop()


def test_group_by_aggregation(
    version: str,
    collection_name: str,
    num_aggregations: int = TEST_PARAMETERS["aggregation"]["group_by_count"],
    iterations: int = TEST_PARAMETERS["aggregation"]["iterations"]
) -> Tuple[Dict[str, Any], PerformanceMetrics]:
    """
    Test group by aggregation performance.
    
    Args:
        version (str): MongoDB version ('v7' or 'v8')
        collection_name (str): Collection name
        num_aggregations (int): Number of aggregations per iteration
        iterations (int): Number of iterations
        
    Returns:
        Tuple[Dict[str, Any], PerformanceMetrics]: Test results and performance metrics
    """
    test_name = f"group_by_aggregation_{collection_name}"
    logger.info(f"Running {test_name} test for MongoDB {version}")
    
    # Get database and collection
    db = get_database(version)
    collection = db[collection_name]
    
    # Define group by aggregation pipelines based on collection
    pipelines = []
    
    if collection_name == "customers":
        # Group by customer aggregations
        pipelines.append([
            {"$group": {"_id": "$address.state", "count": {"$sum": 1}, "avg_credit_score": {"$avg": "$credit_score"}}}
        ])
        pipelines.append([
            {"$group": {"_id": {"state": "$address.state", "city": "$address.city"}, "count": {"$sum": 1}}}
        ])
        pipelines.append([
            {"$group": {"_id": {"$dateToString": {"format": "%Y-%m", "date": "$created_at"}}, "count": {"$sum": 1}}}
        ])
        pipelines.append([
            {"$bucket": {
                "groupBy": "$credit_score",
                "boundaries": [300, 400, 500, 600, 700, 800, 900],
                "default": "Other",
                "output": {"count": {"$sum": 1}}
            }}
        ])
    
    elif collection_name == "accounts":
        # Group by account aggregations
        pipelines.append([
            {"$group": {"_id": "$account_type", "count": {"$sum": 1}, "total_balance": {"$sum": "$balance"}}}
        ])
        pipelines.append([
            {"$group": {"_id": "$status", "count": {"$sum": 1}, "avg_balance": {"$avg": "$balance"}}}
        ])
        pipelines.append([
            {"$group": {"_id": {"account_type": "$account_type", "status": "$status"}, "count": {"$sum": 1}}}
        ])
        pipelines.append([
            {"$bucket": {
                "groupBy": "$balance",
                "boundaries": [0, 1000, 5000, 10000, 50000, 100000, 500000],
                "default": "Other",
                "output": {"count": {"$sum": 1}, "avg_interest_rate": {"$avg": "$interest_rate"}}
            }}
        ])
    
    elif collection_name == "transactions":
        # Group by transaction aggregations
        pipelines.append([
            {"$group": {"_id": "$transaction_type", "count": {"$sum": 1}, "total_amount": {"$sum": "$amount"}}}
        ])
        pipelines.append([
            {"$group": {"_id": "$category", "count": {"$sum": 1}, "avg_amount": {"$avg": "$amount"}}}
        ])
        pipelines.append([
            {"$group": {"_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}}, "count": {"$sum": 1}}}
        ])
        pipelines.append([
            {"$group": {"_id": {"transaction_type": "$transaction_type", "status": "$status"}, "count": {"$sum": 1}}}
        ])
        pipelines.append([
            {"$bucket": {
                "groupBy": "$amount",
                "boundaries": [0, 10, 50, 100, 500, 1000, 5000],
                "default": "Other",
                "output": {"count": {"$sum": 1}}
            }}
        ])
    
    elif collection_name == "loans":
        # Group by loan aggregations
        pipelines.append([
            {"$group": {"_id": "$loan_type", "count": {"$sum": 1}, "total_amount": {"$sum": "$amount"}}}
        ])
        pipelines.append([
            {"$group": {"_id": "$status", "count": {"$sum": 1}, "avg_amount": {"$avg": "$amount"}}}
        ])
        pipelines.append([
            {"$group": {"_id": {"loan_type": "$loan_type", "status": "$status"}, "count": {"$sum": 1}}}
        ])
        pipelines.append([
            {"$bucket": {
                "groupBy": "$amount",
                "boundaries": [0, 5000, 10000, 50000, 100000, 500000, 1000000],
                "default": "Other",
                "output": {"count": {"$sum": 1}, "avg_interest_rate": {"$avg": "$interest_rate"}}
            }}
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
        "num_aggregations": num_aggregations,
        "iterations": iterations,
        "total_aggregations": 0,
        "total_time": 0.0,
        "avg_throughput": 0.0
    }
    
    try:
        # Run test iterations
        for i in range(iterations):
            logger.info(f"Iteration {i+1}/{iterations}")
            
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
            
            logger.info(
                f"Completed {aggregations_completed} group by aggregations in {iteration_time:.2f}s "
                f"({aggregations_completed/iteration_time:.2f} aggs/s)"
            )
        
        # Calculate average throughput
        if results["total_time"] > 0:
            results["avg_throughput"] = results["total_aggregations"] / results["total_time"]
        
        return results, metrics
    
    finally:
        # Stop metrics collection
        metrics.stop()


def test_lookup_aggregation(
    version: str,
    num_aggregations: int = TEST_PARAMETERS["aggregation"]["lookup_count"],
    iterations: int = TEST_PARAMETERS["aggregation"]["iterations"]
) -> Tuple[Dict[str, Any], PerformanceMetrics]:
    """
    Test lookup aggregation performance.
    
    Args:
        version (str): MongoDB version ('v7' or 'v8')
        num_aggregations (int): Number of aggregations per iteration
        iterations (int): Number of iterations
        
    Returns:
        Tuple[Dict[str, Any], PerformanceMetrics]: Test results and performance metrics
    """
    test_name = "lookup_aggregation"
    logger.info(f"Running {test_name} test for MongoDB {version}")
    
    # Get database
    db = get_database(version)
    
    # Define lookup aggregation pipelines
    pipelines = []
    
    # Customer with accounts
    pipelines.append([
        {"$sample": {"size": 1}},
        {"$lookup": {
            "from": "accounts",
            "localField": "customer_id",
            "foreignField": "customer_id",
            "as": "accounts"
        }},
        {"$project": {
            "customer_id": 1,
            "first_name": 1,
            "last_name": 1,
            "accounts": {
                "account_id": 1,
                "account_type": 1,
                "balance": 1,
                "status": 1
            }
        }}
    ])
    
    # Customer with loans
    pipelines.append([
        {"$sample": {"size": 1}},
        {"$lookup": {
            "from": "loans",
            "localField": "customer_id",
            "foreignField": "customer_id",
            "as": "loans"
        }},
        {"$project": {
            "customer_id": 1,
            "first_name": 1,
            "last_name": 1,
            "loans": {
                "loan_id": 1,
                "loan_type": 1,
                "amount": 1,
                "status": 1
            }
        }}
    ])
    
    # Account with transactions
    pipelines.append([
        {"$sample": {"size": 1}},
        {"$lookup": {
            "from": "transactions",
            "localField": "account_id",
            "foreignField": "account_id",
            "as": "transactions"
        }},
        {"$project": {
            "account_id": 1,
            "account_type": 1,
            "balance": 1,
            "transactions": {
                "transaction_id": 1,
                "transaction_type": 1,
                "amount": 1,
                "timestamp": 1
            }
        }}
    ])
    
    # Customer with accounts and transactions
    pipelines.append([
        {"$sample": {"size": 1}},
        {"$lookup": {
            "from": "accounts",
            "localField": "customer_id",
            "foreignField": "customer_id",
            "as": "accounts"
        }},
        {"$unwind": "$accounts"},
        {"$lookup": {
            "from": "transactions",
            "localField": "accounts.account_id",
            "foreignField": "account_id",
            "as": "accounts.transactions"
        }},
        {"$group": {
            "_id": "$_id",
            "customer_id": {"$first": "$customer_id"},
            "first_name": {"$first": "$first_name"},
            "last_name": {"$first": "$last_name"},
            "accounts": {"$push": "$accounts"}
        }},
        {"$project": {
            "customer_id": 1,
            "first_name": 1,
            "last_name": 1,
            "accounts": {
                "account_id": 1,
                "account_type": 1,
                "balance": 1,
                "transactions": {
                    "transaction_id": 1,
                    "transaction_type": 1,
                    "amount": 1,
                    "timestamp": 1
                }
            }
        }}
    ])
    
    # Customer with accounts, transactions, and loans
    pipelines.append([
        {"$sample": {"size": 1}},
        {"$lookup": {
            "from": "accounts",
            "localField": "customer_id",
            "foreignField": "customer_id",
            "as": "accounts"
        }},
        {"$lookup": {
            "from": "loans",
            "localField": "customer_id",
            "foreignField": "customer_id",
            "as": "loans"
        }},
        {"$project": {
            "customer_id": 1,
            "first_name": 1,
            "last_name": 1,
            "accounts": {
                "account_id": 1,
                "account_type": 1,
                "balance": 1
            },
            "loans": {
                "loan_id": 1,
                "loan_type": 1,
                "amount": 1,
                "status": 1
            }
        }}
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
        "num_aggregations": num_aggregations,
        "iterations": iterations,
        "total_aggregations": 0,
        "total_time": 0.0,
        "avg_throughput": 0.0
    }
    
    try:
        # Run test iterations
        for i in range(iterations):
            logger.info(f"Iteration {i+1}/{iterations}")
            
            iteration_start_time = time.time()
            aggregations_completed = 0
            
            # Perform aggregations
            for pipeline in pipelines:
                # Determine collection based on pipeline
                if "$lookup" in str(pipeline) and "accounts" in str(pipeline):
                    collection = db["customers"]
                elif "$lookup" in str(pipeline) and "transactions" in str(pipeline):
                    collection = db["accounts"]
                else:
                    collection = db["customers"]
                
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
            
            logger.info(
                f"Completed {aggregations_completed} lookup aggregations in {iteration_time:.2f}s "
                f"({aggregations_completed/iteration_time:.2f} aggs/s)"
            )
        
        # Calculate average throughput
        if results["total_time"] > 0:
            results["avg_throughput"] = results["total_aggregations"] / results["total_time"]
        
        return results, metrics
    
    finally:
        # Stop metrics collection
        metrics.stop()


def test_complex_aggregation(
    version: str,
    num_aggregations: int = TEST_PARAMETERS["aggregation"]["complex_count"],
    iterations: int = TEST_PARAMETERS["aggregation"]["iterations"]
) -> Tuple[Dict[str, Any], PerformanceMetrics]:
    """
    Test complex aggregation performance.
    
    Args:
        version (str): MongoDB version ('v7' or 'v8')
        num_aggregations (int): Number of aggregations per iteration
        iterations (int): Number of iterations
        
    Returns:
        Tuple[Dict[str, Any], PerformanceMetrics]: Test results and performance metrics
    """
    test_name = "complex_aggregation"
    logger.info(f"Running {test_name} test for MongoDB {version}")
    
    # Get database
    db = get_database(version)
    
    # Define complex aggregation pipelines
    pipelines = []
    
    # Customer spending patterns by category
    pipelines.append({
        "collection": "customers",
        "pipeline": [
            {"$sample": {"size": 1}},
            {"$lookup": {
                "from": "accounts",
                "localField": "customer_id",
                "foreignField": "customer_id",
                "as": "accounts"
            }},
            {"$unwind": "$accounts"},
            {"$lookup": {
                "from": "transactions",
                "localField": "accounts.account_id",
                "foreignField": "account_id",
                "as": "transactions"
            }},
            {"$unwind": "$transactions"},
            {"$match": {"transactions.transaction_type": {"$in": ["PAYMENT", "PURCHASE"]}}},
            {"$group": {
                "_id": {
                    "customer_id": "$customer_id",
                    "category": "$transactions.category"
                },
                "total_spent": {"$sum": "$transactions.amount"},
                "count": {"$sum": 1},
                "avg_amount": {"$avg": "$transactions.amount"}
            }},
            {"$sort": {"total_spent": -1}},
            {"$group": {
                "_id": "$_id.customer_id",
                "spending_by_category": {
                    "$push": {
                        "category": "$_id.category",
                        "total_spent": "$total_spent",
                        "count": "$count",
                        "avg_amount": "$avg_amount"
                    }
                },
                "total_spent": {"$sum": "$total_spent"}
            }},
            {"$project": {
                "customer_id": "$_id",
                "total_spent": 1,
                "spending_by_category": 1,
                "_id": 0
            }}
        ]
    })
    
    # Monthly transaction summary
    pipelines.append({
        "collection": "transactions",
        "pipeline": [
            {"$match": {"timestamp": {"$gte": datetime.now() - timedelta(days=365)}}},
            {"$group": {
                "_id": {
                    "year": {"$year": "$timestamp"},
                    "month": {"$month": "$timestamp"},
                    "transaction_type": "$transaction_type"
                },
                "count": {"$sum": 1},
                "total_amount": {"$sum": "$amount"},
                "avg_amount": {"$avg": "$amount"}
            }},
            {"$sort": {"_id.year": 1, "_id.month": 1}},
            {"$group": {
                "_id": {
                    "year": "$_id.year",
                    "month": "$_id.month"
                },
                "transactions": {
                    "$push": {
                        "type": "$_id.transaction_type",
                        "count": "$count",
                        "total_amount": "$total_amount",
                        "avg_amount": "$avg_amount"
                    }
                },
                "total_count": {"$sum": "$count"},
                "total_amount": {"$sum": "$total_amount"}
            }},
            {"$sort": {"_id.year": 1, "_id.month": 1}},
            {"$project": {
                "year_month": {
                    "$concat": [
                        {"$toString": "$_id.year"},
                        "-",
                        {"$toString": "$_id.month"}
                    ]
                },
                "transactions": 1,
                "total_count": 1,
                "total_amount": 1,
                "_id": 0
            }}
        ]
    })
    
    # Account balance trends
    pipelines.append({
        "collection": "accounts",
        "pipeline": [
            {"$match": {"status": "ACTIVE"}},
            {"$lookup": {
                "from": "transactions",
                "localField": "account_id",
                "foreignField": "account_id",
                "as": "transactions"
            }},
            {"$unwind": "$transactions"},
            {"$sort": {"transactions.timestamp": 1}},
            {"$group": {
                "_id": {
                    "account_id": "$account_id",
                    "year_month": {
                        "$dateToString": {
                            "format": "%Y-%m",
                            "date": "$transactions.timestamp"
                        }
                    }
                },
                "deposits": {
                    "$sum": {
                        "$cond": [
                            {"$in": ["$transactions.transaction_type", ["DEPOSIT", "INTEREST", "REFUND"]]},
                            "$transactions.amount",
                            0
                        ]
                    }
                },
                "withdrawals": {
                    "$sum": {
                        "$cond": [
                            {"$in": ["$transactions.transaction_type", ["WITHDRAWAL", "PAYMENT", "FEE", "PURCHASE"]]},
                            "$transactions.amount",
                            0
                        ]
                    }
                },
                "transaction_count": {"$sum": 1}
            }},
            {"$sort": {"_id.account_id": 1, "_id.year_month": 1}},
            {"$group": {
                "_id": "$_id.account_id",
                "monthly_activity": {
                    "$push": {
                        "year_month": "$_id.year_month",
                        "deposits": "$deposits",
                        "withdrawals": "$withdrawals",
                        "net_change": {"$subtract": ["$deposits", "$withdrawals"]},
                        "transaction_count": "$transaction_count"
                    }
                }
            }},
            {"$lookup": {
                "from": "accounts",
                "localField": "_id",
                "foreignField": "account_id",
                "as": "account_info"
            }},
            {"$unwind": "$account_info"},
            {"$project": {
                "account_id": "$_id",
                "account_type": "$account_info.account_type",
                "current_balance": "$account_info.balance",
                "monthly_activity": 1,
                "_id": 0
            }},
            {"$limit": 10}
        ]
    })
    
    # Loan payment analysis
    pipelines.append({
        "collection": "loans",
        "pipeline": [
            {"$match": {"status": {"$in": ["ACTIVE", "PAID_OFF"]}}},
            {"$unwind": "$payments"},
            {"$group": {
                "_id": {
                    "loan_id": "$loan_id",
                    "payment_status": "$payments.status"
                },
                "count": {"$sum": 1},
                "total_amount": {"$sum": "$payments.amount"}
            }},
            {"$group": {
                "_id": "$_id.loan_id",
                "payment_summary": {
                    "$push": {
                        "status": "$_id.payment_status",
                        "count": "$count",
                        "total_amount": "$total_amount"
                    }
                },
                "total_payments": {"$sum": "$count"}
            }},
            {"$lookup": {
                "from": "loans",
                "localField": "_id",
                "foreignField": "loan_id",
                "as": "loan_info"
            }},
            {"$unwind": "$loan_info"},
            {"$project": {
                "loan_id": "$_id",
                "loan_type": "$loan_info.loan_type",
                "loan_amount": "$loan_info.amount",
                "status": "$loan_info.status",
                "payment_summary": 1,
                "total_payments": 1,
                "payment_completion": {
                    "$divide": [
                        {"$size": {
                            "$filter": {
                                "input": "$loan_info.payments",
                                "as": "payment",
                                "cond": {"$eq": ["$$payment.status", "PAID"]}
                            }
                        }},
                        {"$size": "$loan_info.payments"}
                    ]
                },
                "_id": 0
            }},
            {"$sort": {"payment_completion": -1}},
            {"$limit": 10}
        ]
    })
    
    # Customer financial overview
    pipelines.append({
        "collection": "customers",
        "pipeline": [
            {"$sample": {"size": 1}},
            {"$lookup": {
                "from": "accounts",
                "localField": "customer_id",
                "foreignField": "customer_id",
                "as": "accounts"
            }},
            {"$lookup": {
                "from": "loans",
                "localField": "customer_id",
                "foreignField": "customer_id",
                "as": "loans"
            }},
            {"$addFields": {
                "total_balance": {"$sum": "$accounts.balance"},
                "total_debt": {"$sum": "$loans.amount"},
                "net_worth": {"$subtract": [{"$sum": "$accounts.balance"}, {"$sum": "$loans.amount"}]}
            }},
            {"$unwind": {
                "path": "$accounts",
                "preserveNullAndEmptyArray": true
            }},
            {"$lookup": {
                "from": "transactions",
                "localField": "accounts.account_id",
                "foreignField": "account_id",
                "as": "accounts.transactions"
            }},
            {"$group": {
                "_id": "$_id",
                "customer_id": {"$first": "$customer_id"},
                "first_name": {"$first": "$first_name"},
                "last_name": {"$first": "$last_name"},
                "credit_score": {"$first": "$credit_score"},
                "total_balance": {"$first": "$total_balance"},
                "total_debt": {"$first": "$total_debt"},
                "net_worth": {"$first": "$net_worth"},
                "accounts": {"$push": "$accounts"}
            }},
            {"$project": {
                "customer_id": 1,
                "first_name": 1,
                "last_name": 1,
                "credit_score": 1,
                "financial_summary": {
                    "total_balance": "$total_balance",
                    "total_debt": "$total_debt",
                    "net_worth": "$net_worth",
                    "accounts": "$accounts",
                    "loans": "$loans"
                },
                "_id": 0
            }}
        ]
    })
    
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
        "num_aggregations": num_aggregations,
        "iterations": iterations,
        "total_aggregations": 0,
        "total_time": 0.0,
        "avg_throughput": 0.0
    }
    
    try:
        # Run test iterations
        for i in range(iterations):
            logger.info(f"Iteration {i+1}/{iterations}")
            
            iteration_start_time = time.time()
            aggregations_completed = 0
            
            # Perform aggregations
            for pipeline_config in pipelines:
                collection = db[pipeline_config["collection"]]
                pipeline = pipeline_config["pipeline"]
                
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
            
            logger.info(
                f"Completed {aggregations_completed} complex aggregations in {iteration_time:.2f}s "
                f"({aggregations_completed/iteration_time:.2f} aggs/s)"
            )
        
        # Calculate average throughput
        if results["total_time"] > 0:
            results["avg_throughput"] = results["total_aggregations"] / results["total_time"]
        
        return results, metrics
    
    finally:
        # Stop metrics collection
        metrics.stop()


def run_aggregation_tests(version: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, PerformanceMetrics]]:
    """
    Run all aggregation tests for a specific MongoDB version.
    
    Args:
        version (str): MongoDB version ('v7' or 'v8')
        
    Returns:
        Tuple[Dict[str, Dict[str, Any]], Dict[str, PerformanceMetrics]]:
            Dictionary with test results and dictionary with performance metrics
    """
    results = {}
    metrics_dict = {}
    
    # Test simple aggregations for each collection
    for collection_name in COLLECTIONS.values():
        test_name = f"simple_aggregation_{collection_name}"
        test_result, metrics = test_simple_aggregation(version, collection_name)
        results[test_name] = test_result
        metrics_dict[test_name] = metrics
    
    # Test group by aggregations for each collection
    for collection_name in COLLECTIONS.values():
        test_name = f"group_by_aggregation_{collection_name}"
        test_result, metrics = test_group_by_aggregation(version, collection_name)
        results[test_name] = test_result
        metrics_dict[test_name] = metrics
    
    # Test lookup aggregations
    test_name = "lookup_aggregation"
    test_result, metrics = test_lookup_aggregation(version)
    results[test_name] = test_result
    metrics_dict[test_name] = metrics
    
    # Test complex aggregations
    test_name = "complex_aggregation"
    test_result, metrics = test_complex_aggregation(version)
    results[test_name] = test_result
    metrics_dict[test_name] = metrics
    
    return results, metrics_dict


def compare_aggregation_performance() -> Dict[str, Any]:
    """
    Compare aggregation performance between MongoDB v7.0 and v8.0.
    
    Returns:
        Dict[str, Any]: Comparison results
    """
    # Run tests for MongoDB v7.0
    logger.info("Running aggregation tests for MongoDB v7.0")
    v7_results, v7_metrics = run_aggregation_tests('v7')
    
    # Run tests for MongoDB v8.0
    logger.info("Running aggregation tests for MongoDB v8.0")
    v8_results, v8_metrics = run_aggregation_tests('v8')
    
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
    comparison = compare_aggregation_performance()
    
    # Print summary
    print("\nAggregation Performance Comparison (MongoDB v8.0 vs v7.0):")
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