#!/usr/bin/env python3
"""
Modular Flow Executor

A standalone module that provides convenient interfaces for executing flows
using the existing flow architecture. This is benchmark-agnostic and reuses
all existing flow components.
"""

import asyncio
from typing import Dict, Any, Optional, Union, List
from datetime import datetime

from app.agent.base import BaseAgent
from app.flow.flow_factory import FlowFactory, FlowType
from app.flow.base import BaseFlow
from app.config import config


class FlowExecutor:
    """
    Convenience wrapper around the existing flow architecture.
    Provides simple interfaces for benchmark integration while using existing APIs.
    """

    def __init__(self,
                 flow_type: FlowType = FlowType.ITERATIVE_REFINEMENT,
                 agents: Optional[Union[BaseAgent, List[BaseAgent], Dict[str, BaseAgent]]] = None,
                 **flow_kwargs):
        """
        Initialize flow executor using existing flow factory and config.

        Args:
            flow_type: Type of flow to create (uses existing FlowType enum)
            agents: Agents to use (if None, will create default agents)
            **flow_kwargs: Additional arguments passed to flow creation (overrides config)
        """
        self.flow_type = flow_type

        # Merge config values with provided kwargs (kwargs take precedence)
        config_kwargs = self._get_config_kwargs()
        merged_kwargs = {**config_kwargs, **flow_kwargs}

        # Use existing flow factory if agents provided, otherwise defer to lazy loading
        if agents is not None:
            self.flow = FlowFactory.create_flow(
                flow_type=flow_type,
                agents=agents,
                **merged_kwargs
            )
        else:
            self.flow = None
            self.flow_kwargs = merged_kwargs

    def _get_config_kwargs(self) -> Dict[str, Any]:
        """Extract flow configuration from global config."""
        config_kwargs = {}

        try:
            if hasattr(config, '_config') and config._config and config._config.flow_config:
                flow_config = config._config.flow_config

                # Map config values to flow constructor parameters
                if hasattr(flow_config, 'max_iterations'):
                    config_kwargs['max_iterations'] = flow_config.max_iterations
                if hasattr(flow_config, 'max_steps'):
                    config_kwargs['max_steps'] = flow_config.max_steps

                print(f"🔧 Using config values: max_iterations={config_kwargs.get('max_iterations', 'not set')}")

        except Exception as e:
            print(f"⚠️ Could not load flow config, using defaults: {e}")

        return config_kwargs

    def _ensure_flow_initialized(self):
        """Lazy initialization of flow with default agents if needed."""
        if self.flow is None:
            # Import agents only when needed to avoid circular imports
            from app.agent.translator import TranslatorAgent
            from app.agent.text_only_reasoning import TextOnlyReasoningAgent

            # Create default agents using existing classes
            agents = {
                "translator": TranslatorAgent(),
                "reasoning": TextOnlyReasoningAgent()
            }

            # Use existing flow factory
            self.flow = FlowFactory.create_flow(
                flow_type=self.flow_type,
                agents=agents,
                **self.flow_kwargs
            )

    @property
    def underlying_flow(self) -> BaseFlow:
        """Get the underlying flow instance (lazy initialization)."""
        self._ensure_flow_initialized()
        return self.flow

    async def execute_async(self, input_text: str, **execute_kwargs) -> Dict[str, Any]:
        """
        Execute the flow asynchronously and return detailed result.

        Args:
            input_text: The input text for the flow (can include image_path)
            **execute_kwargs: Additional arguments passed to flow.execute()

        Returns:
            Dict containing response, execution_time, and metadata
        """
        self._ensure_flow_initialized()

        start_time = datetime.now()

        try:
            # Use existing flow execute method
            response = await self.flow.execute(input_text, **execute_kwargs)

            # Calculate execution time
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            return {
                "response": response,
                "execution_time_seconds": round(execution_time, 2),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "flow_type": self.flow_type.value,
                "success": True,
                "error": None
            }

        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            return {
                "response": f"Error: {str(e)}",
                "execution_time_seconds": round(execution_time, 2),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "flow_type": self.flow_type.value,
                "success": False,
                "error": str(e)
            }

    def execute(self, input_text: str, **execute_kwargs) -> Dict[str, Any]:
        """
        Execute the flow synchronously (wrapper for async execution).

        Args:
            input_text: The input text for the flow
            **execute_kwargs: Additional arguments passed to flow.execute()

        Returns:
            Dict containing response and execution metadata
        """
        return asyncio.run(self.execute_async(input_text, **execute_kwargs))


    @property
    def underlying_flow(self) -> BaseFlow:
        """
        Get access to the underlying flow for advanced usage.

        Returns:
            The underlying flow instance
        """
        self._ensure_flow_initialized()
        return self.flow


# Convenience functions for common use cases
def create_iterative_refinement_executor(**kwargs) -> FlowExecutor:
    """
    Create a FlowExecutor with IterativeRefinementFlow.

    Args:
        **kwargs: Additional arguments passed to flow creation

    Returns:
        FlowExecutor configured for iterative refinement
    """
    return FlowExecutor(
        flow_type=FlowType.ITERATIVE_REFINEMENT,
        **kwargs
    )


# Example usage and testing functions
async def test_flow_executor():
    """Test the FlowExecutor with sample inputs."""
    # Use the convenience function to create executor (uses config for max_iterations)
    executor = create_iterative_refinement_executor()

    # Test cases
    test_cases = [
        {
            "name": "Question with image",
            "input": "What do you see in this image?\nimage_path:/path/to/image.jpg",
        },
        {
            "name": "Multiple choice question",
            "input": "What is the capital of France?\nOptions: (A) London, (B) Paris, (C) Rome, (D) Madrid\nimage_path:/path/to/image.jpg",
        }
    ]

    print("🧪 Testing FlowExecutor with existing flow architecture...")

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {test_case['name']}")
        print(f"Input: {test_case['input']}")
        print("-" * 50)

        # Execute flow using existing APIs
        try:
            result = await executor.execute_async(test_case['input'])
            print(f"✅ Success: {result['success']}")
            print(f"⏱️ Time: {result['execution_time_seconds']}s")
            print(f"🔄 Flow Type: {result['flow_type']}")
            print(f"📄 Response: {result['response'][:200]}...")
        except Exception as e:
            print(f"❌ Error: {e}")

        print("=" * 70)


def main():
    """Main function for testing the FlowExecutor."""
    asyncio.run(test_flow_executor())


if __name__ == "__main__":
    main()