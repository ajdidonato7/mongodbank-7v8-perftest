"""
Test configuration parameters for MongoDB performance testing.
This module provides configuration settings for various test scenarios.
"""

import os
from typing import Dict, List, Any, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# MongoDB cluster names
MONGO_V7_CLUSTER_NAME = os.getenv("MONGO_V7_CLUSTER_NAME", "Cluster1")
MONGO_V8_CLUSTER_NAME = os.getenv("MONGO_V8_CLUSTER_NAME", "AJCluster")

# Collection names
COLLECTIONS = {
    "customers": "customers",
    "accounts": "accounts",
    "transactions": "transactions",
    "loans": "loans"
}

# Data generation parameters
DATA_GENERATION = {
    "customers": {
        "count": int(os.getenv("CUSTOMER_COUNT", "100000")),
        "batch_size": 1000
    },
    "accounts_per_customer": {
        "min": 3,
        "max": 5
    },
    "transactions_per_account": {
        "min": 10,
        "max": 20
    },
    "loans_per_customer": {
        "min": 0,
        "max": 2
    }
}

# Test parameters
TEST_PARAMETERS = {
    "bulk_insert": {
        "batch_sizes": {
            "customers": [100, 1000, 10000],
            "accounts": [100, 1000, 10000],
            "transactions": [1000, 10000, 100000],
            "loans": [100, 1000, 10000]
        },
        "iterations": 5  # Number of times to repeat each test
    },
    "read": {
        "single_reads_count": 1000,  # Number of single document reads
        "filtered_reads_count": 1000,  # Number of filtered reads
        "complex_reads_count": 500,  # Number of complex reads
        "pagination": {
            "page_sizes": [10, 50, 100, 500],
            "pages_count": 10  # Number of pages to read
        },
        "iterations": 5  # Number of times to repeat each test
    },
    "aggregation": {
        "simple_count": 100,  # Number of simple aggregations
        "group_by_count": 100,  # Number of group by aggregations
        "lookup_count": 50,  # Number of lookup aggregations
        "complex_count": 20,  # Number of complex aggregations
        "iterations": 3  # Number of times to repeat each test
    },
    "mixed_workload": {
        "duration_seconds": 300,  # Duration of mixed workload test
        "read_write_ratio": 0.8,  # 80% reads, 20% writes
        "concurrency": [1, 5, 10, 20, 50]  # Number of concurrent clients
    }
}

# Performance metrics collection parameters
METRICS = {
    "response_time": {
        "percentiles": [50, 90, 95, 99]  # Percentiles to calculate
    },
    "throughput": {
        "window_size": 5  # Window size in seconds for throughput calculation
    },
    "resource_utilization": {
        "interval": 1.0  # Sampling interval in seconds
    }
}

# Indexes to create
INDEXES = {
    "customers": [
        {"fields": [("customer_id", 1)], "unique": True},
        {"fields": [("email", 1)], "unique": True},
        {"fields": [("credit_score", -1)], "unique": False}
    ],
    "accounts": [
        {"fields": [("account_id", 1)], "unique": True},
        {"fields": [("customer_id", 1)], "unique": False},
        {"fields": [("account_type", 1), ("status", 1)], "unique": False}
    ],
    "transactions": [
        {"fields": [("transaction_id", 1)], "unique": True},
        {"fields": [("account_id", 1)], "unique": False},
        {"fields": [("timestamp", -1)], "unique": False},
        {"fields": [("account_id", 1), ("timestamp", -1)], "unique": False},
        {"fields": [("category", 1), ("timestamp", -1)], "unique": False}
    ],
    "loans": [
        {"fields": [("loan_id", 1)], "unique": True},
        {"fields": [("customer_id", 1)], "unique": False},
        {"fields": [("status", 1)], "unique": False}
    ]
}

# Reporting configuration
REPORTING = {
    "output_dir": "reports",
    "formats": ["csv", "json", "html"],
    "charts": True,
    "save_raw_data": True
}

# Account types for data generation
ACCOUNT_TYPES = [
    "CHECKING",
    "SAVINGS",
    "MONEY_MARKET",
    "CERTIFICATE_OF_DEPOSIT",
    "RETIREMENT",
    "INVESTMENT"
]

# Transaction types for data generation
TRANSACTION_TYPES = [
    "DEPOSIT",
    "WITHDRAWAL",
    "TRANSFER",
    "PAYMENT",
    "FEE",
    "INTEREST",
    "REFUND",
    "PURCHASE"
]

# Transaction categories for data generation
TRANSACTION_CATEGORIES = [
    "GROCERIES",
    "DINING",
    "ENTERTAINMENT",
    "UTILITIES",
    "TRANSPORTATION",
    "HEALTHCARE",
    "EDUCATION",
    "SHOPPING",
    "TRAVEL",
    "HOUSING",
    "INCOME",
    "INVESTMENT",
    "OTHER"
]

# Loan types for data generation
LOAN_TYPES = [
    "PERSONAL",
    "MORTGAGE",
    "AUTO",
    "STUDENT",
    "HOME_EQUITY",
    "CREDIT_CARD",
    "BUSINESS"
]

# Status values for data generation
STATUS_VALUES = {
    "account": ["ACTIVE", "INACTIVE", "CLOSED", "FROZEN", "PENDING"],
    "transaction": ["COMPLETED", "PENDING", "FAILED", "CANCELLED", "REFUNDED"],
    "loan": ["ACTIVE", "PAID_OFF", "DEFAULTED", "PENDING", "APPROVED", "REJECTED"],
    "payment": ["PAID", "PENDING", "LATE", "MISSED", "PARTIAL"]
}

# Currencies for data generation
CURRENCIES = ["USD", "EUR", "GBP", "CAD", "JPY", "AUD", "CHF"]