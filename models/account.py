"""
Account model for MongoDB collection.
This module provides a class for the account collection model.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import random

from .base_model import BaseModel
from config.test_config import ACCOUNT_TYPES, STATUS_VALUES, CURRENCIES


class Account(BaseModel):
    """Account model for MongoDB collection."""
    
    collection_name = "accounts"
    
    def __init__(
        self,
        account_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        account_type: Optional[str] = None,
        balance: Optional[float] = None,
        currency: Optional[str] = None,
        status: Optional[str] = None,
        interest_rate: Optional[float] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        **kwargs
    ):
        """Initialize a new Account instance."""
        # Set default account type if not provided
        if account_type is None and ACCOUNT_TYPES:
            account_type = random.choice(ACCOUNT_TYPES)
        
        # Set default status if not provided
        if status is None and STATUS_VALUES.get("account"):
            status = random.choice(STATUS_VALUES["account"])
        
        # Set default currency if not provided
        if currency is None and CURRENCIES:
            currency = random.choice(CURRENCIES)
        
        # Set default interest rate based on account type
        if interest_rate is None:
            if account_type == "SAVINGS":
                interest_rate = round(random.uniform(0.5, 2.5), 2)
            elif account_type == "MONEY_MARKET":
                interest_rate = round(random.uniform(1.0, 3.0), 2)
            elif account_type == "CERTIFICATE_OF_DEPOSIT":
                interest_rate = round(random.uniform(2.0, 4.0), 2)
            else:
                interest_rate = 0.0
        
        super().__init__(
            account_id=account_id or self.generate_id("ACCT_"),
            customer_id=customer_id,
            account_type=account_type,
            balance=balance if balance is not None else 0.0,
            currency=currency,
            status=status,
            interest_rate=interest_rate,
            created_at=created_at or self.current_timestamp(),
            updated_at=updated_at or self.current_timestamp(),
            **kwargs
        )
    
    @property
    def account_id(self) -> str:
        """Get the account ID."""
        return self._data["account_id"]
    
    @property
    def customer_id(self) -> str:
        """Get the customer ID."""
        return self._data["customer_id"]
    
    @property
    def balance(self) -> float:
        """Get the account balance."""
        return self._data["balance"]
    
    @property
    def status(self) -> str:
        """Get the account status."""
        return self._data["status"]
    
    def deposit(self, amount: float) -> None:
        """
        Deposit money into the account.
        
        Args:
            amount (float): Amount to deposit
            
        Raises:
            ValueError: If amount is negative
        """
        if amount < 0:
            raise ValueError("Deposit amount cannot be negative")
        
        if self._data["status"] not in ["ACTIVE", "PENDING"]:
            raise ValueError(f"Cannot deposit to account with status {self._data['status']}")
        
        self._data["balance"] += amount
        self._data["updated_at"] = self.current_timestamp()
    
    def withdraw(self, amount: float) -> None:
        """
        Withdraw money from the account.
        
        Args:
            amount (float): Amount to withdraw
            
        Raises:
            ValueError: If amount is negative or exceeds balance
        """
        if amount < 0:
            raise ValueError("Withdrawal amount cannot be negative")
        
        if self._data["status"] != "ACTIVE":
            raise ValueError(f"Cannot withdraw from account with status {self._data['status']}")
        
        if amount > self._data["balance"]:
            raise ValueError("Insufficient funds")
        
        self._data["balance"] -= amount
        self._data["updated_at"] = self.current_timestamp()
    
    def update_status(self, status: str) -> None:
        """
        Update the account status.
        
        Args:
            status (str): New status
            
        Raises:
            ValueError: If status is invalid
        """
        if status not in STATUS_VALUES.get("account", []):
            valid_statuses = ", ".join(STATUS_VALUES.get("account", []))
            raise ValueError(f"Invalid status. Must be one of: {valid_statuses}")
        
        self._data["status"] = status
        self._data["updated_at"] = self.current_timestamp()
    
    def apply_interest(self) -> float:
        """
        Apply interest to the account based on interest rate.
        
        Returns:
            float: Interest amount applied
        """
        if self._data["status"] != "ACTIVE":
            return 0.0
        
        interest_amount = self._data["balance"] * (self._data["interest_rate"] / 100)
        self._data["balance"] += interest_amount
        self._data["updated_at"] = self.current_timestamp()
        
        return interest_amount