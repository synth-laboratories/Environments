# This file re-exports the actual registry functions from src.environment.registry
# to be used by the service layer, maintaining a clean separation if needed.
from src.environment.registry import register_environment, get_environment_cls, list_supported_env_types

__all__ = ["register_environment", "get_environment_cls", "list_supported_env_types"]
