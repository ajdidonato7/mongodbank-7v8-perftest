"""
Loan model for MongoDB collection.
This module provides a class for the loan collection model.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import random
import uuid

from .base_model import BaseModel
from ..config.test_config import LOAN_TYPES, STATUS_VALUES, CURRENCIES


class LoanPayment:
    """Class representing a loan payment."""
    
    def __init__(
        self,
        payment_id: Optional[str] = None,
        amount: float = 0.0,
        due_date: Optional[datetime] = None,
        paid_date: Optional[datetime] = None,
        status: Optional[str] = None
    ):
        """Initialize a new LoanPayment instance."""
        self.payment_id = payment_id or f"PMT_{uuid.uuid4().hex}"
        self.amount = amount
        self.due_date = due_date
        self.paid_date = paid_date
        
        # Set default status if not provided
        if status is None:
            if paid_date is not None:
                self.status = "PAID"
            elif due_date is not None and due_date < datetime.utcnow():
                self.status = "LATE"
            else:
                self.status = "PENDING"
        else:
            self.status = status
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the payment to a dictionary."""
        return {
            "payment_id": self.payment_id,
            "amount": self.amount,
            "due_date": self.due_date,
            "paid_date": self.paid_date,
            "status": self.status
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LoanPayment':
        """Create a LoanPayment instance from a dictionary."""
        return cls(
            payment_id=data.get("payment_id"),
            amount=data.get("amount", 0.0),
            due_date=data.get("due_date"),
            paid_date=data.get("paid_date"),
            status=data.get("status")
        )


class Loan(BaseModel):
    """Loan model for MongoDB collection."""
    
    collection_name = "loans"
    
    def __init__(
        self,
        loan_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        loan_type: Optional[str] = None,
        amount: Optional[float] = None,
        interest_rate: Optional[float] = None,
        term_months: Optional[int] = None,
        status: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        payments: Optional[List[Dict[str, Any]]] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        **kwargs
    ):
        """Initialize a new Loan instance."""
        # Set default loan type if not provided
        if loan_type is None and LOAN_TYPES:
            loan_type = random.choice(LOAN_TYPES)
        
        # Set default status if not provided
        if status is None and STATUS_VALUES.get("loan"):
            status = random.choice(STATUS_VALUES["loan"])
        
        # Set default interest rate based on loan type
        if interest_rate is None:
            if loan_type == "MORTGAGE":
                interest_rate = round(random.uniform(3.0, 6.0), 2)
            elif loan_type == "AUTO":
                interest_rate = round(random.uniform(4.0, 8.0), 2)
            elif loan_type == "PERSONAL":
                interest_rate = round(random.uniform(7.0, 15.0), 2)
            elif loan_type == "CREDIT_CARD":
                interest_rate = round(random.uniform(15.0, 25.0), 2)
            else:
                interest_rate = round(random.uniform(5.0, 12.0), 2)
        
        # Set default term months based on loan type
        if term_months is None:
            if loan_type == "MORTGAGE":
                term_months = random.choice([180, 240, 360])  # 15, 20, or 30 years
            elif loan_type == "AUTO":
                term_months = random.choice([36, 48, 60, 72])  # 3, 4, 5, or 6 years
            elif loan_type == "PERSONAL":
                term_months = random.choice([12, 24, 36, 48, 60])  # 1-5 years
            elif loan_type == "STUDENT":
                term_months = random.choice([120, 180, 240])  # 10, 15, or 20 years
            else:
                term_months = random.choice([12, 24, 36, 48, 60])  # 1-5 years
        
        # Set default amount based on loan type
        if amount is None:
            if loan_type == "MORTGAGE":
                amount = round(random.uniform(100000, 1000000), 2)
            elif loan_type == "AUTO":
                amount = round(random.uniform(10000, 50000), 2)
            elif loan_type == "PERSONAL":
                amount = round(random.uniform(5000, 25000), 2)
            elif loan_type == "STUDENT":
                amount = round(random.uniform(10000, 100000), 2)
            else:
                amount = round(random.uniform(5000, 50000), 2)
        
        # Set default start date if not provided
        if start_date is None:
            # Random date within the last 5 years
            days_ago = random.randint(0, 365 * 5)
            start_date = datetime.utcnow() - timedelta(days=days_ago)
        
        # Set default end date based on start date and term
        if end_date is None and start_date is not None:
            end_date = start_date + timedelta(days=30 * term_months)
        
        # Initialize payments if not provided
        if payments is None:
            payments = []
            
            # Only generate payments if loan is active or paid off
            if status in ["ACTIVE", "PAID_OFF"]:
                # Calculate monthly payment (simple calculation)
                monthly_payment = amount / term_months
                
                # Add interest (simple calculation)
                monthly_payment *= (1 + (interest_rate / 100))
                
                # Round to 2 decimal places
                monthly_payment = round(monthly_payment, 2)
                
                # Generate payment schedule
                current_date = start_date
                for i in range(term_months):
                    due_date = current_date + timedelta(days=30)
                    
                    # Determine if payment is in the past
                    is_past_due = due_date < datetime.utcnow()
                    
                    # For past payments, determine if paid
                    paid_date = None
                    if is_past_due:
                        # 90% chance of payment being made on time
                        if random.random() < 0.9:
                            # Payment made 0-5 days before due date
                            days_before = random.randint(0, 5)
                            paid_date = due_date - timedelta(days=days_before)
                    
                    payment = LoanPayment(
                        amount=monthly_payment,
                        due_date=due_date,
                        paid_date=paid_date
                    )
                    
                    payments.append(payment.to_dict())
                    current_date = due_date
        
        super().__init__(
            loan_id=loan_id or self.generate_id("LOAN_"),
            customer_id=customer_id,
            loan_type=loan_type,
            amount=amount,
            interest_rate=interest_rate,
            term_months=term_months,
            status=status,
            start_date=start_date,
            end_date=end_date,
            payments=payments,
            created_at=created_at or self.current_timestamp(),
            updated_at=updated_at or self.current_timestamp(),
            **kwargs
        )
    
    @property
    def loan_id(self) -> str:
        """Get the loan ID."""
        return self._data["loan_id"]
    
    @property
    def customer_id(self) -> str:
        """Get the customer ID."""
        return self._data["customer_id"]
    
    @property
    def amount(self) -> float:
        """Get the loan amount."""
        return self._data["amount"]
    
    @property
    def status(self) -> str:
        """Get the loan status."""
        return self._data["status"]
    
    @property
    def payments(self) -> List[LoanPayment]:
        """Get the loan payments as LoanPayment objects."""
        return [LoanPayment.from_dict(payment) for payment in self._data["payments"]]
    
    def update_status(self, status: str) -> None:
        """
        Update the loan status.
        
        Args:
            status (str): New status
            
        Raises:
            ValueError: If status is invalid
        """
        if status not in STATUS_VALUES.get("loan", []):
            valid_statuses = ", ".join(STATUS_VALUES.get("loan", []))
            raise ValueError(f"Invalid status. Must be one of: {valid_statuses}")
        
        self._data["status"] = status
        self._data["updated_at"] = self.current_timestamp()
    
    def add_payment(self, amount: float, paid_date: datetime = None) -> LoanPayment:
        """
        Add a payment to the loan.
        
        Args:
            amount (float): Payment amount
            paid_date (datetime, optional): Date payment was made
            
        Returns:
            LoanPayment: The created payment
            
        Raises:
            ValueError: If loan is not active
        """
        if self._data["status"] != "ACTIVE":
            raise ValueError(f"Cannot add payment to loan with status {self._data['status']}")
        
        # Find the next unpaid payment
        next_payment = None
        for payment_dict in self._data["payments"]:
            payment = LoanPayment.from_dict(payment_dict)
            if payment.status in ["PENDING", "LATE"]:
                next_payment = payment
                break
        
        if next_payment is None:
            raise ValueError("No pending payments found")
        
        # Update the payment
        next_payment.amount = amount
        next_payment.paid_date = paid_date or self.current_timestamp()
        next_payment.status = "PAID"
        
        # Update the payment in the data
        for i, payment_dict in enumerate(self._data["payments"]):
            if payment_dict["payment_id"] == next_payment.payment_id:
                self._data["payments"][i] = next_payment.to_dict()
                break
        
        # Check if all payments are made
        all_paid = all(
            LoanPayment.from_dict(payment).status == "PAID"
            for payment in self._data["payments"]
        )
        
        if all_paid:
            self._data["status"] = "PAID_OFF"
        
        self._data["updated_at"] = self.current_timestamp()
        
        return next_payment
    
    def calculate_remaining_balance(self) -> float:
        """
        Calculate the remaining balance on the loan.
        
        Returns:
            float: Remaining balance
        """
        if self._data["status"] == "PAID_OFF":
            return 0.0
        
        # Sum up all paid payments
        paid_amount = sum(
            LoanPayment.from_dict(payment).amount
            for payment in self._data["payments"]
            if LoanPayment.from_dict(payment).status == "PAID"
        )
        
        return self._data["amount"] - paid_amount