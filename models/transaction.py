"""
Transaction model for MongoDB collection.
This module provides a class for the transaction collection model.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import random

from .base_model import BaseModel
from ..config.test_config import TRANSACTION_TYPES, TRANSACTION_CATEGORIES, STATUS_VALUES, CURRENCIES


class Transaction(BaseModel):
    """Transaction model for MongoDB collection."""
    
    collection_name = "transactions"
    
    def __init__(
        self,
        transaction_id: Optional[str] = None,
        account_id: Optional[str] = None,
        transaction_type: Optional[str] = None,
        amount: Optional[float] = None,
        currency: Optional[str] = None,
        description: Optional[str] = None,
        status: Optional[str] = None,
        merchant: Optional[str] = None,
        category: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        **kwargs
    ):
        """Initialize a new Transaction instance."""
        # Set default transaction type if not provided
        if transaction_type is None and TRANSACTION_TYPES:
            transaction_type = random.choice(TRANSACTION_TYPES)
        
        # Set default status if not provided
        if status is None and STATUS_VALUES.get("transaction"):
            status = random.choice(STATUS_VALUES["transaction"])
        
        # Set default currency if not provided
        if currency is None and CURRENCIES:
            currency = random.choice(CURRENCIES)
        
        # Set default category based on transaction type
        if category is None and TRANSACTION_CATEGORIES:
            category = random.choice(TRANSACTION_CATEGORIES)
        
        super().__init__(
            transaction_id=transaction_id or self.generate_id("TXN_"),
            account_id=account_id,
            transaction_type=transaction_type,
            amount=amount if amount is not None else round(random.uniform(1.0, 1000.0), 2),
            currency=currency,
            description=description or f"{transaction_type} transaction",
            status=status,
            merchant=merchant,
            category=category,
            timestamp=timestamp or self.current_timestamp(),
            **kwargs
        )
    
    @property
    def transaction_id(self) -> str:
        """Get the transaction ID."""
        return self._data["transaction_id"]
    
    @property
    def account_id(self) -> str:
        """Get the account ID."""
        return self._data["account_id"]
    
    @property
    def amount(self) -> float:
        """Get the transaction amount."""
        return self._data["amount"]
    
    @property
    def transaction_type(self) -> str:
        """Get the transaction type."""
        return self._data["transaction_type"]
    
    @property
    def status(self) -> str:
        """Get the transaction status."""
        return self._data["status"]
    
    @property
    def is_debit(self) -> bool:
        """Check if the transaction is a debit (reduces account balance)."""
        debit_types = ["WITHDRAWAL", "PAYMENT", "FEE", "PURCHASE", "TRANSFER"]
        return self._data["transaction_type"] in debit_types
    
    @property
    def is_credit(self) -> bool:
        """Check if the transaction is a credit (increases account balance)."""
        credit_types = ["DEPOSIT", "INTEREST", "REFUND"]
        return self._data["transaction_type"] in credit_types
    
    def update_status(self, status: str) -> None:
        """
        Update the transaction status.
        
        Args:
            status (str): New status
            
        Raises:
            ValueError: If status is invalid
        """
        if status not in STATUS_VALUES.get("transaction", []):
            valid_statuses = ", ".join(STATUS_VALUES.get("transaction", []))
            raise ValueError(f"Invalid status. Must be one of: {valid_statuses}")
        
        self._data["status"] = status
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the transaction to a dictionary for MongoDB storage."""
        return self._data
    
    @classmethod
    def create_transfer(
        cls,
        from_account_id: str,
        to_account_id: str,
        amount: float,
        currency: str = "USD",
        description: Optional[str] = None
    ) -> List['Transaction']:
        """
        Create a pair of transactions representing a transfer between accounts.
        
        Args:
            from_account_id (str): Source account ID
            to_account_id (str): Destination account ID
            amount (float): Transfer amount
            currency (str): Currency code
            description (str, optional): Transaction description
            
        Returns:
            List[Transaction]: List containing debit and credit transactions
        """
        timestamp = cls.current_timestamp()
        
        # Create debit transaction (from account)
        debit = cls(
            account_id=from_account_id,
            transaction_type="TRANSFER",
            amount=amount,
            currency=currency,
            description=description or f"Transfer to account {to_account_id}",
            status="COMPLETED",
            category="TRANSFER",
            timestamp=timestamp
        )
        
        # Create credit transaction (to account)
        credit = cls(
            account_id=to_account_id,
            transaction_type="TRANSFER",
            amount=amount,
            currency=currency,
            description=description or f"Transfer from account {from_account_id}",
            status="COMPLETED",
            category="TRANSFER",
            timestamp=timestamp
        )
        
        return [debit, credit]