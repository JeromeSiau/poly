"""Custom exceptions for the Poly trading system."""


class PolyError(Exception):
    """Base exception for all Poly errors."""


class FeedError(PolyError):
    """Error connecting to or reading from a data feed."""


class ExecutionError(PolyError):
    """Error placing, cancelling, or checking an order."""


class OrderPlacementError(ExecutionError):
    """Order placement was rejected or failed."""


class OrderCancelError(ExecutionError):
    """Order cancellation failed."""


class PersistenceError(PolyError):
    """Database persistence failure."""


class ConfigError(PolyError):
    """Missing or invalid configuration."""
