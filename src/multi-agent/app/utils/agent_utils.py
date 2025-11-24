"""Utility functions for agent setup and management."""

from typing import Type, TypeVar
from pydantic import model_validator
from app.llm import LLM

T = TypeVar('T')

def setup_agent_llm(config_name: str):
    """
    Decorator factory for setting up agent LLM with specified config.

    Reduces boilerplate in agent model_validator methods.

    Args:
        config_name: The LLM config name to use (e.g., "translator_api", "reasoning_api")

    Usage:
        @setup_agent_llm("translator_api")
        class TranslatorAgent(BaseAgent):
            pass
    """
    def decorator(cls: Type[T]) -> Type[T]:
        original_init = getattr(cls, '__init__', None)

        @model_validator(mode="after")
        def setup_llm(self) -> cls:
            """Setup LLM with the specified config name."""
            print(f"🔧 {cls.__name__} model_validator called! Current LLM: {getattr(self, 'llm', 'NOT SET')}")
            print(f"🔧 FORCING new LLM with config_name='{config_name}' (Local vLLM API)")

            # Use force_new_instance to bypass singleton cache
            self.llm = LLM.force_new_instance(config_name=config_name)
            print(f"🔧 FORCED LLM: {self.llm.model} ({self.llm.api_type}) at {self.llm.base_url}")
            return self

        # Add the validator to the class
        setattr(cls, f'setup_{config_name.replace("_", "_")}_llm', setup_llm)
        return cls

    return decorator


def create_llm_setup_validator(config_name: str, agent_name: str = None):
    """
    Create a model_validator function for LLM setup.

    Args:
        config_name: The LLM config name to use
        agent_name: Optional agent name for logging (defaults to class name)

    Returns:
        A model_validator function that can be used in agent classes
    """
    def setup_llm_validator(self):
        """Setup LLM with the specified config name."""
        agent_display_name = agent_name or self.__class__.__name__
        print(f"🔧 {agent_display_name} model_validator called! Current LLM: {getattr(self, 'llm', 'NOT SET')}")
        print(f"🔧 FORCING new LLM with config_name='{config_name}' (Local vLLM API)")

        # Use force_new_instance to bypass singleton cache
        self.llm = LLM.force_new_instance(config_name=config_name)
        print(f"🔧 FORCED LLM: {self.llm.model} ({self.llm.api_type}) at {self.llm.base_url}")
        return self

    return model_validator(mode="after")(setup_llm_validator)