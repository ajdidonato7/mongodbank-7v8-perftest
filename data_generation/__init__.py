"""
Data generation package for MongoDB performance testing.
This package provides functions to generate and load test data.
"""

from .faker_generator import (
    generate_customer,
    generate_account,
    generate_transaction,
    generate_loan,
    generate_dataset,
    generate_batch
)

from .data_loader import (
    create_indexes,
    bulk_insert,
    load_initial_dataset,
    parallel_bulk_insert,
    generate_and_load_batch,
    get_sample_ids,
    clear_collections,
    get_collection_counts
)

__all__ = [
    'generate_customer',
    'generate_account',
    'generate_transaction',
    'generate_loan',
    'generate_dataset',
    'generate_batch',
    'create_indexes',
    'bulk_insert',
    'load_initial_dataset',
    'parallel_bulk_insert',
    'generate_and_load_batch',
    'get_sample_ids',
    'clear_collections',
    'get_collection_counts'
]