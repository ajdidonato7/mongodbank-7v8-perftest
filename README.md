# MongoDB Performance Testing Framework

A comprehensive framework for testing and comparing the performance of MongoDB v7.0 and v8.0 in a banking application context.

## Overview

This framework simulates a banking application workload to test the performance differences between MongoDB v7.0 and MongoDB v8.0. It includes tests for:

- Periodic bulk inserts
- Single document reads
- Complex aggregations with $group, $lookup, $unwind, $sort, and more
- Mixed workloads with concurrent operations

The framework also includes a data generation component that can create a realistic banking dataset with customers, accounts, transactions, and loans.

## Features

- **Comprehensive Data Model**: Realistic banking data model with customers, accounts, transactions, and loans collections with relationships between them.
- **Flexible Data Generation**: Generate customizable amounts of test data using Faker.
- **Multiple Test Types**: Test bulk inserts, reads, aggregations, and mixed workloads.
- **Performance Metrics**: Measure query response time, throughput, and resource utilization.
- **Detailed Reporting**: Generate comprehensive reports with visualizations comparing MongoDB v7.0 and v8.0 performance.
- **Configurable**: Easily adjust test parameters, data volume, and more.

## Requirements

- Python 3.9+
- MongoDB v7.0 and v8.0 instances/clusters
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```
git clone <repository-url>
cd mongodb-performance-test
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Configure MongoDB connections:
   - Set environment variables for MongoDB connection strings:
     - `MONGO_V7_URI`: MongoDB v7.0 connection string (default: "mongodb://localhost:27017/")
     - `MONGO_V8_URI`: MongoDB v8.0 connection string (default: "mongodb://localhost:27018/")
     - `MONGO_V7_DB`: MongoDB v7.0 database name (default: "banking_v7")
     - `MONGO_V8_DB`: MongoDB v8.0 database name (default: "banking_v8")
   - Or create a `.env` file with these variables

## Usage

### Basic Usage

Run all tests with default settings:

```
python main.py
```

### Load Test Data

Load initial test data before running tests:

```
python main.py --load-data --customer-count 10000
```

### Run Specific Tests

Run only bulk insert tests:

```
python main.py --test-type bulk_insert
```

Run only read tests:

```
python main.py --test-type read
```

Run only aggregation tests:

```
python main.py --test-type aggregation
```

Run only mixed workload tests:

```
python main.py --test-type mixed
```

### Test Specific MongoDB Version

Test only MongoDB v7.0:

```
python main.py --version v7
```

Test only MongoDB v8.0:

```
python main.py --version v8
```

### Customize Output

Specify output directory and report formats:

```
python main.py --output-dir ./my-reports --report-formats json,html
```

## Project Structure

```
mongodb-performance-test/
├── config/
│   ├── __init__.py
│   ├── connection.py        # MongoDB connection settings
│   └── test_config.py       # Test configuration parameters
├── data_generation/
│   ├── __init__.py
│   ├── faker_generator.py   # Data generation using Faker
│   └── data_loader.py       # Bulk loading data into MongoDB
├── models/
│   ├── __init__.py
│   ├── base_model.py        # Base model class
│   ├── customer.py          # Customer model
│   ├── account.py           # Account model
│   ├── transaction.py       # Transaction model
│   └── loan.py              # Loan model
├── tests/
│   ├── __init__.py
│   ├── test_bulk_insert.py  # Bulk insert performance tests
│   ├── test_read.py         # Read operation performance tests
│   ├── test_aggregation.py  # Aggregation performance tests
│   └── test_mixed.py        # Mixed workload tests
├── utils/
│   ├── __init__.py
│   ├── metrics.py           # Performance metrics collection
│   ├── reporting.py         # Results reporting and visualization
│   └── monitoring.py        # Resource utilization monitoring
├── main.py                  # Main entry point
├── requirements.txt         # Dependencies
└── README.md                # Documentation
```

## Data Model

The framework uses a comprehensive banking data model with the following collections:

### Customer Collection
- Basic customer information (name, email, phone, address)
- Credit score and other financial attributes

### Account Collection
- Different account types (checking, savings, etc.)
- Balance, currency, status, interest rate

### Transaction Collection
- Different transaction types (deposit, withdrawal, payment, etc.)
- Amount, currency, description, merchant, category

### Loan Collection
- Different loan types (personal, mortgage, auto, etc.)
- Amount, interest rate, term, status, payment history

## Performance Metrics

The framework collects the following performance metrics:

1. **Query Response Time**
   - Average response time
   - Percentiles (p50, p90, p95, p99)
   - Min/max response times

2. **Throughput**
   - Operations per second
   - Total operations completed
   - Throughput under different concurrency levels

3. **Resource Utilization**
   - CPU usage
   - Memory usage
   - Disk I/O
   - Network I/O

## Reports

The framework generates comprehensive reports comparing MongoDB v7.0 and v8.0 performance:

1. **Summary Reports**
   - Overall performance comparison
   - Percentage improvements from v7.0 to v8.0

2. **Detailed Reports**
   - Test-specific performance metrics
   - Visualizations (charts and graphs)
   - Resource utilization statistics

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.