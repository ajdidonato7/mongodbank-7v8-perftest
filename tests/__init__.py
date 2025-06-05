"""
Tests package for MongoDB performance testing.
This package provides functions to test MongoDB performance.
"""

from .test_bulk_insert import (
    test_bulk_insert_customers,
    test_bulk_insert_accounts,
    test_bulk_insert_transactions,
    test_bulk_insert_loans,
    test_parallel_bulk_insert,
    run_bulk_insert_tests,
    compare_bulk_insert_performance
)

from .test_read import (
    test_single_document_reads,
    test_filtered_reads,
    test_complex_reads,
    test_paginated_reads,
    run_read_tests,
    compare_read_performance
)

from .test_aggregation import (
    test_simple_aggregation,
    test_group_by_aggregation,
    test_lookup_aggregation,
    test_complex_aggregation,
    run_aggregation_tests,
    compare_aggregation_performance
)

from .test_mixed import (
    MixedWorkloadRunner,
    test_mixed_workload,
    run_mixed_workload_tests,
    compare_mixed_workload_performance
)

__all__ = [
    'test_bulk_insert_customers',
    'test_bulk_insert_accounts',
    'test_bulk_insert_transactions',
    'test_bulk_insert_loans',
    'test_parallel_bulk_insert',
    'run_bulk_insert_tests',
    'compare_bulk_insert_performance',
    'test_single_document_reads',
    'test_filtered_reads',
    'test_complex_reads',
    'test_paginated_reads',
    'run_read_tests',
    'compare_read_performance',
    'test_simple_aggregation',
    'test_group_by_aggregation',
    'test_lookup_aggregation',
    'test_complex_aggregation',
    'run_aggregation_tests',
    'compare_aggregation_performance',
    'MixedWorkloadRunner',
    'test_mixed_workload',
    'run_mixed_workload_tests',
    'compare_mixed_workload_performance'
]