"""
Data generation module using Faker.
This module provides functions to generate realistic banking data.
"""

import random
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import uuid
from faker import Faker
import string

from models import Customer, Account, Transaction, Loan
from config.test_config import (
    DATA_GENERATION, ACCOUNT_TYPES, TRANSACTION_TYPES,
    TRANSACTION_CATEGORIES, LOAN_TYPES, STATUS_VALUES, CURRENCIES
)

# Initialize Faker
fake = Faker()


def generate_customer() -> Customer:
    """
    Generate a random customer.
    
    Returns:
        Customer: Generated customer
    """
    first_name = fake.first_name()
    last_name = fake.last_name()
    email = f"{first_name.lower()}.{last_name.lower()}@{fake.domain_name()}"
    
    # Generate address
    address = {
        "street": fake.street_address(),
        "city": fake.city(),
        "state": fake.state_abbr(),
        "zip": fake.zipcode(),
        "country": "US"
    }
    
    # Generate date of birth (18-80 years old)
    dob = fake.date_of_birth(minimum_age=18, maximum_age=80)
    
    # Generate SSN
    ssn = fake.ssn()
    
    # Generate credit score (300-850)
    credit_score = random.randint(300, 850)
    
    return Customer(
        first_name=first_name,
        last_name=last_name,
        email=email,
        phone=fake.phone_number(),
        address=address,
        date_of_birth=dob,
        ssn=ssn,
        credit_score=credit_score
    )


def generate_account(customer_id: str) -> Account:
    """
    Generate a random account for a customer.
    
    Args:
        customer_id (str): Customer ID
        
    Returns:
        Account: Generated account
    """
    # Choose account type
    account_type = random.choice(ACCOUNT_TYPES)
    
    # Generate balance based on account type
    if account_type == "CHECKING":
        balance = round(random.uniform(100, 10000), 2)
    elif account_type == "SAVINGS":
        balance = round(random.uniform(1000, 50000), 2)
    elif account_type == "MONEY_MARKET":
        balance = round(random.uniform(5000, 100000), 2)
    elif account_type == "CERTIFICATE_OF_DEPOSIT":
        balance = round(random.uniform(10000, 100000), 2)
    elif account_type == "RETIREMENT":
        balance = round(random.uniform(10000, 500000), 2)
    elif account_type == "INVESTMENT":
        balance = round(random.uniform(5000, 1000000), 2)
    else:
        balance = round(random.uniform(100, 10000), 2)
    
    # Choose currency
    currency = random.choice(CURRENCIES)
    
    # Choose status (mostly active)
    status_weights = [0.9, 0.05, 0.03, 0.01, 0.01]  # Weights for ACTIVE, INACTIVE, CLOSED, FROZEN, PENDING
    status = random.choices(
        STATUS_VALUES["account"],
        weights=status_weights,
        k=1
    )[0]
    
    # Generate interest rate based on account type
    if account_type == "SAVINGS":
        interest_rate = round(random.uniform(0.5, 2.5), 2)
    elif account_type == "MONEY_MARKET":
        interest_rate = round(random.uniform(1.0, 3.0), 2)
    elif account_type == "CERTIFICATE_OF_DEPOSIT":
        interest_rate = round(random.uniform(2.0, 4.0), 2)
    else:
        interest_rate = 0.0
    
    return Account(
        customer_id=customer_id,
        account_type=account_type,
        balance=balance,
        currency=currency,
        status=status,
        interest_rate=interest_rate
    )


def generate_transaction(account_id: str, timestamp_range: Tuple[datetime, datetime] = None) -> Transaction:
    """
    Generate a random transaction for an account.
    
    Args:
        account_id (str): Account ID
        timestamp_range (Tuple[datetime, datetime], optional): Range for transaction timestamp
        
    Returns:
        Transaction: Generated transaction
    """
    # Choose transaction type
    transaction_type = random.choice(TRANSACTION_TYPES)
    
    # Generate amount based on transaction type
    if transaction_type in ["DEPOSIT", "WITHDRAWAL"]:
        amount = round(random.uniform(10, 1000), 2)
    elif transaction_type == "TRANSFER":
        amount = round(random.uniform(50, 5000), 2)
    elif transaction_type == "PAYMENT":
        amount = round(random.uniform(20, 500), 2)
    elif transaction_type == "FEE":
        amount = round(random.uniform(1, 50), 2)
    elif transaction_type == "INTEREST":
        amount = round(random.uniform(0.1, 100), 2)
    elif transaction_type == "REFUND":
        amount = round(random.uniform(5, 200), 2)
    elif transaction_type == "PURCHASE":
        amount = round(random.uniform(5, 500), 2)
    else:
        amount = round(random.uniform(10, 1000), 2)
    
    # Choose currency
    currency = random.choice(CURRENCIES)
    
    # Generate description
    if transaction_type == "DEPOSIT":
        description = f"Deposit at {fake.company()}"
    elif transaction_type == "WITHDRAWAL":
        description = f"Withdrawal from {fake.company()} ATM"
    elif transaction_type == "TRANSFER":
        description = f"Transfer to account ending in {fake.numerify('####')}"
    elif transaction_type == "PAYMENT":
        description = f"Payment to {fake.company()}"
    elif transaction_type == "FEE":
        description = f"{fake.word().capitalize()} fee"
    elif transaction_type == "INTEREST":
        description = "Interest payment"
    elif transaction_type == "REFUND":
        description = f"Refund from {fake.company()}"
    elif transaction_type == "PURCHASE":
        description = f"Purchase at {fake.company()}"
    else:
        description = f"{transaction_type} transaction"
    
    # Choose status (mostly completed)
    status_weights = [0.95, 0.02, 0.01, 0.01, 0.01]  # Weights for COMPLETED, PENDING, FAILED, CANCELLED, REFUNDED
    status = random.choices(
        STATUS_VALUES["transaction"],
        weights=status_weights,
        k=1
    )[0]
    
    # Generate merchant for purchases and payments
    merchant = None
    if transaction_type in ["PURCHASE", "PAYMENT"]:
        merchant = fake.company()
    
    # Choose category
    category = random.choice(TRANSACTION_CATEGORIES)
    
    # Generate timestamp
    if timestamp_range is None:
        # Random timestamp within the last year
        days_ago = random.randint(0, 365)
        timestamp = datetime.utcnow() - timedelta(days=days_ago)
    else:
        start_time, end_time = timestamp_range
        timestamp = start_time + timedelta(
            seconds=random.randint(0, int((end_time - start_time).total_seconds()))
        )
    
    return Transaction(
        account_id=account_id,
        transaction_type=transaction_type,
        amount=amount,
        currency=currency,
        description=description,
        status=status,
        merchant=merchant,
        category=category,
        timestamp=timestamp
    )


def generate_loan(customer_id: str) -> Loan:
    """
    Generate a random loan for a customer.
    
    Args:
        customer_id (str): Customer ID
        
    Returns:
        Loan: Generated loan
    """
    # Choose loan type
    loan_type = random.choice(LOAN_TYPES)
    
    # Generate loan amount based on loan type
    if loan_type == "MORTGAGE":
        amount = round(random.uniform(100000, 1000000), 2)
    elif loan_type == "AUTO":
        amount = round(random.uniform(10000, 50000), 2)
    elif loan_type == "PERSONAL":
        amount = round(random.uniform(5000, 25000), 2)
    elif loan_type == "STUDENT":
        amount = round(random.uniform(10000, 100000), 2)
    elif loan_type == "HOME_EQUITY":
        amount = round(random.uniform(25000, 250000), 2)
    elif loan_type == "CREDIT_CARD":
        amount = round(random.uniform(1000, 20000), 2)
    elif loan_type == "BUSINESS":
        amount = round(random.uniform(25000, 500000), 2)
    else:
        amount = round(random.uniform(5000, 50000), 2)
    
    # Generate interest rate based on loan type
    if loan_type == "MORTGAGE":
        interest_rate = round(random.uniform(3.0, 6.0), 2)
    elif loan_type == "AUTO":
        interest_rate = round(random.uniform(4.0, 8.0), 2)
    elif loan_type == "PERSONAL":
        interest_rate = round(random.uniform(7.0, 15.0), 2)
    elif loan_type == "STUDENT":
        interest_rate = round(random.uniform(4.0, 8.0), 2)
    elif loan_type == "HOME_EQUITY":
        interest_rate = round(random.uniform(4.0, 7.0), 2)
    elif loan_type == "CREDIT_CARD":
        interest_rate = round(random.uniform(15.0, 25.0), 2)
    elif loan_type == "BUSINESS":
        interest_rate = round(random.uniform(6.0, 12.0), 2)
    else:
        interest_rate = round(random.uniform(5.0, 12.0), 2)
    
    # Generate term months based on loan type
    if loan_type == "MORTGAGE":
        term_months = random.choice([180, 240, 360])  # 15, 20, or 30 years
    elif loan_type == "AUTO":
        term_months = random.choice([36, 48, 60, 72])  # 3, 4, 5, or 6 years
    elif loan_type == "PERSONAL":
        term_months = random.choice([12, 24, 36, 48, 60])  # 1-5 years
    elif loan_type == "STUDENT":
        term_months = random.choice([120, 180, 240])  # 10, 15, or 20 years
    elif loan_type == "HOME_EQUITY":
        term_months = random.choice([60, 120, 180, 240])  # 5, 10, 15, or 20 years
    elif loan_type == "CREDIT_CARD":
        term_months = 0  # Revolving credit
    elif loan_type == "BUSINESS":
        term_months = random.choice([12, 24, 36, 48, 60, 120])  # 1-10 years
    else:
        term_months = random.choice([12, 24, 36, 48, 60])  # 1-5 years
    
    # Choose status (mostly active)
    status_weights = [0.8, 0.1, 0.05, 0.03, 0.01, 0.01]  # Weights for ACTIVE, PAID_OFF, DEFAULTED, PENDING, APPROVED, REJECTED
    status = random.choices(
        STATUS_VALUES["loan"],
        weights=status_weights,
        k=1
    )[0]
    
    # Generate start date (random date within the last 5 years)
    days_ago = random.randint(0, 365 * 5)
    start_date = datetime.utcnow() - timedelta(days=days_ago)
    
    # Generate end date based on start date and term
    if term_months > 0:
        end_date = start_date + timedelta(days=30 * term_months)
    else:
        end_date = None
    
    return Loan(
        customer_id=customer_id,
        loan_type=loan_type,
        amount=amount,
        interest_rate=interest_rate,
        term_months=term_months,
        status=status,
        start_date=start_date,
        end_date=end_date
    )


def generate_dataset(
    num_customers: int = DATA_GENERATION["customers"]["count"],
    accounts_per_customer_range: Tuple[int, int] = (
        DATA_GENERATION["accounts_per_customer"]["min"],
        DATA_GENERATION["accounts_per_customer"]["max"]
    ),
    transactions_per_account_range: Tuple[int, int] = (
        DATA_GENERATION["transactions_per_account"]["min"],
        DATA_GENERATION["transactions_per_account"]["max"]
    ),
    loans_per_customer_range: Tuple[int, int] = (
        DATA_GENERATION["loans_per_customer"]["min"],
        DATA_GENERATION["loans_per_customer"]["max"]
    )
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate a complete dataset with customers, accounts, transactions, and loans.
    
    Args:
        num_customers (int): Number of customers to generate
        accounts_per_customer_range (Tuple[int, int]): Range of accounts per customer
        transactions_per_account_range (Tuple[int, int]): Range of transactions per account
        loans_per_customer_range (Tuple[int, int]): Range of loans per customer
        
    Returns:
        Dict[str, List[Dict[str, Any]]]: Dictionary with generated data
    """
    dataset = {
        "customers": [],
        "accounts": [],
        "transactions": [],
        "loans": []
    }
    
    # Generate customers
    for _ in range(num_customers):
        customer = generate_customer()
        dataset["customers"].append(customer.to_dict())
        
        # Generate accounts for this customer
        num_accounts = random.randint(*accounts_per_customer_range)
        for _ in range(num_accounts):
            account = generate_account(customer.customer_id)
            dataset["accounts"].append(account.to_dict())
            
            # Generate transactions for this account
            num_transactions = random.randint(*transactions_per_account_range)
            for _ in range(num_transactions):
                transaction = generate_transaction(account.account_id)
                dataset["transactions"].append(transaction.to_dict())
        
        # Generate loans for this customer
        num_loans = random.randint(*loans_per_customer_range)
        for _ in range(num_loans):
            loan = generate_loan(customer.customer_id)
            dataset["loans"].append(loan.to_dict())
    
    return dataset


def generate_batch(
    collection_name: str,
    batch_size: int,
    customer_ids: Optional[List[str]] = None,
    account_ids: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Generate a batch of documents for a specific collection.
    
    Args:
        collection_name (str): Collection name
        batch_size (int): Number of documents to generate
        customer_ids (List[str], optional): List of customer IDs for generating related documents
        account_ids (List[str], optional): List of account IDs for generating transactions
        
    Returns:
        List[Dict[str, Any]]: List of generated documents
    """
    batch = []
    
    if collection_name == "customers":
        for _ in range(batch_size):
            customer = generate_customer()
            batch.append(customer.to_dict())
    
    elif collection_name == "accounts":
        if not customer_ids:
            raise ValueError("Customer IDs are required for generating accounts")
        
        for _ in range(batch_size):
            customer_id = random.choice(customer_ids)
            account = generate_account(customer_id)
            batch.append(account.to_dict())
    
    elif collection_name == "transactions":
        if not account_ids:
            raise ValueError("Account IDs are required for generating transactions")
        
        for _ in range(batch_size):
            account_id = random.choice(account_ids)
            transaction = generate_transaction(account_id)
            batch.append(transaction.to_dict())
    
    elif collection_name == "loans":
        if not customer_ids:
            raise ValueError("Customer IDs are required for generating loans")
        
        for _ in range(batch_size):
            customer_id = random.choice(customer_ids)
            loan = generate_loan(customer_id)
            batch.append(loan.to_dict())
    
    else:
        raise ValueError(f"Unknown collection name: {collection_name}")
    
    return batch