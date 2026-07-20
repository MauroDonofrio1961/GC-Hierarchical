class GCHierarchicalError(Exception):
    """Base exception for the package."""


class ConfigurationError(GCHierarchicalError):
    """Raised when configuration is absent or invalid."""


class DataValidationError(GCHierarchicalError):
    """Raised when input data fail schema validation."""
