"""
Models package for MongoDB collections.
This package provides classes for all MongoDB collection models.
"""

from .base_model import BaseModel
from .customer import Customer
from .account import Account
from .transaction import Transaction
from .loan import Loan

__all__ = [
    'BaseModel',
    'Customer',
    'Account',
    'Transaction',
    'Loan'
]