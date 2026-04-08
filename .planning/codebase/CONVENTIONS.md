# Coding Conventions

**Analysis Date:** 2026-04-07

## Naming Patterns

**Files:**
- Lowercase with underscores: `config.py`, `llm.py`, `flow_executor.py`
- Tool/component grouping: `agent/base.py`, `agent/react.py`, `flow/flow_executor.py`
- Utility files: `utils/benchmark_infer.py`, `utils/vllm_setup.py`

**Functions:**
- snake_case for all functions: `parse_tool_calls_multiple_formats()`, `get_project_root()`, `define_log_level()`
- Private functions prefixed with underscore: `_import_model_providers()`, `_get_config_path()`, `_log_step_start()`
- Async functions use `async def`: `async def step()`, `async def run()`, `async def think()`
- Helper parsing functions: `parse_xml_tool_calls_standard()`, `parse_loose_json_tool_calls()`, `extract_first_complete_json()`

**Variables:**
- snake_case for all variables: `max_steps`, `current_step`, `project_root`, `config_path`
- Constants in UPPER_SNAKE_CASE: `PROJECT_ROOT`, `WORKSPACE_ROOT`, `TOOL_CALL_REQUIRED`, `ROLE_VALUES`
- Class attributes use snake_case: `system_prompt`, `next_step_prompt`, `final_step_prompt`
- Private attributes prefixed with underscore: `_instance`, `_lock`, `_initialized`, `_config`, `_print_level`
- Global singletons: `config = Config()`, `logger = define_log_level()`

**Types:**
- Type hints use PascalCase: `BaseModel`, `BaseAgent`, `ReActAgent`, `Message`, `Memory`
- Enum types use PascalCase: `Role`, `AgentState`, `ToolChoice`
- Generic type annotations: `Dict[str, LLMSettings]`, `List[Message]`, `Optional[str]`, `Union[BaseAgent, List[BaseAgent], Dict[str, BaseAgent]]`

## Code Style

**Formatting:**
- Python 3.12+ syntax and features used throughout
- Line length appears to be 100+ characters based on observed code
- Consistent spacing: two blank lines between top-level definitions, one blank line between methods
- Imports organized by: standard library, third-party, local application imports

**Linting:**
- No .eslintrc or linting config files detected - relies on convention
- Pydantic v2.10.6+ used with Field descriptors for all model attributes
- Type hints present on function signatures and class attributes
- Docstrings in module docstrings and class docstrings

## Import Organization

**Order:**
1. Standard library imports: `import sys`, `import json`, `from pathlib import Path`, `from typing import Dict, List, Optional`
2. Third-party imports: `from pydantic import BaseModel, Field`, `from loguru import logger`, `from openai import AsyncOpenAI`
3. Local application imports: `from app.config import config`, `from app.logger import logger`, `from app.schema import Message`

**Path Aliases:**
- No path aliases detected (no `@` or `~` aliases)
- Relative imports used within packages: `from app.agent.base import BaseAgent`
- Absolute imports from module root when crossing packages: `sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))`

**Module Structure:**
- Explicit module loading: `_import_model_providers()` function demonstrates graceful handling of optional imports
- Config singleton pattern: `Config` class with thread-safe `__new__` and `__init__` for single instance

## Error Handling

**Patterns:**
- Generic exception catching with logging: `except Exception as e:` followed by `logger.error()` or `logger.exception()`
- Custom exception classes in `app/exceptions.py`: `ToolError`, `OpenManusError`, `TokenLimitExceeded`
- Try-except-finally blocks for resource cleanup: seen in `state_context()` context manager
- Context managers for state transitions: `async with self.state_context(AgentState.RUNNING):`
- Specific exception handling for API errors: `APIError`, `AuthenticationError`, `RateLimitError` from openai
- Fallback mechanisms: `parse_tool_calls_multiple_formats()` tries multiple parsing strategies before returning empty list
- Logging errors with context: `logger.error(f"Failed to force termination in {self.name}: {e}")`
- Re-raising exceptions after logging: `raise e` or `raise ValueError(f"...")`
- State transitions on error: `self.state = AgentState.ERROR` in exception handlers
- Graceful degradation for optional dependencies: wrapped imports return fallback `available: False` when packages unavailable

## Logging

**Framework:** loguru v0.7.3

**Patterns:**
- Global logger instance: `logger = define_log_level()` exported from `app/logger.py`
- Logger configuration: `define_log_level(print_level="INFO", logfile_level="DEBUG", name=None)`
- Dual output: stderr for console + rotating log files in `logs/` directory
- Log file naming: timestamped with optional prefix, e.g., `{name}_{YYYYMMDDHHMMSS}.log`
- Logging levels used: `logger.info()`, `logger.debug()`, `logger.warning()`, `logger.error()`, `logger.exception()`
- Contextual logging: `logger.info(f"Executing step {self.current_step}/{self.max_steps}")`
- Exception logging: `logger.exception(f"An error occurred: {e}")` to capture full traceback
- Warning for incomplete data: `logger.warning("Found incomplete tool call (possibly truncated)")`

## Comments

**When to Comment:**
- Docstrings on all classes and public methods
- Module-level docstrings explaining purpose
- Complex logic explained with inline comments, e.g., `# Priority 1: Try XML format with proper closing tags`
- Algorithm explanations: comments before complex parsing logic
- Edge cases noted: `# Handle cases where empty arguments are actually valid (rare but possible)`
- State transitions documented: `# Transition to ERROR on failure`

**JSDoc/TSDoc:**
- Python docstrings use triple-quoted strings: `"""`
- Class docstrings include purpose statement: `"""Abstract base class for managing agent state and execution."""`
- Method docstrings include Args, Returns, Raises sections:
  ```python
  def update_memory(self, role: ROLE_TYPE, content: str, base64_image: Optional[str] = None, **kwargs) -> None:
      """Add a message to the agent's memory.

      Args:
          role: The role of the message sender (user, system, assistant, tool).
          content: The message content.
          base64_image: Optional base64 encoded image.
          **kwargs: Additional arguments (e.g., tool_call_id for tool messages).

      Raises:
          ValueError: If the role is unsupported.
      """
  ```
- Pydantic Field descriptions used for model attributes: `Field(..., description="Unique name of the agent")`

## Function Design

**Size:**
- Functions range from ~5-50 lines for utilities
- Larger functions (100+ lines) decomposed into helper methods, e.g., `step()` broken into `_log_step_start()`, `_log_thinking_phase()`, `_log_action_phase()`, `_log_step_complete()`
- Recursive tool call parsing demonstrates clear separation of concerns

**Parameters:**
- Use Pydantic Field annotations for class attributes with descriptions
- Type hints on all function parameters
- Optional parameters have `Optional[type] = None` or `Optional[type] = Field(None, description=...)`
- Configuration classes use Field with `default_factory` for mutable defaults: `Field(default_factory=list)`
- Keyword-only arguments documented: `**flow_kwargs`, `**kwargs`

**Return Values:**
- Explicit return types on all function signatures
- Async functions return `str`, `Optional[str]`, `Dict[str, int]`, etc.
- Constructor methods return instance of class: `-> "BaseAgent"`, `-> "Message"`
- Void methods explicitly return `-> None`
- Class methods use `@classmethod` and return instances: `-> "Message"`

## Module Design

**Exports:**
- Explicit module exports: `logger = define_log_level()` at module level
- Singleton exports: `config = Config()` at end of config.py
- Class exports through inheritance: `class ReActAgent(BaseAgent, ABC):`
- Exception exports from exceptions module: `class ToolError(Exception):`

**Barrel Files:**
- `app/__init__.py` exists but minimal content
- Tool collection: `from app.tool import CreateChatCompletion, Terminate, ToolCollection`
- Agent imports: `from app.agent.base import BaseAgent`
- Schema exports: `from app.schema import ROLE_TYPE, AgentState, Memory, Message`

## Pydantic Usage

**Model Definition Pattern:**
- All data models inherit from `BaseModel`
- Field descriptors with descriptions on all attributes
- Type hints required: `model: str = Field(..., description="Model name")`
- Optional fields use `Optional[type]`: `api_key: str = Field(default="", description="API key")`
- Mutable defaults use `default_factory`: `args: List[str] = Field(default_factory=list, description="...")`
- Nested models: `proxy: Optional[ProxySettings] = Field(None, description="Proxy settings")`
- Model validation with `@model_validator(mode="after")` for post-initialization logic
- Flexible configs: `class Config: arbitrary_types_allowed = True; extra = "allow"`

## Async Patterns

**Async/Await:**
- Async context managers: `@asynccontextmanager` for `state_context()`
- Async methods in agent classes: `async def step()`, `async def run()`, `async def think()`
- Async iteration: `while (...):` with `await self.step()`
- Task management: `asyncio.gather()`, `await asyncio.create_task()`
- Cleanup: `await SANDBOX_CLIENT.cleanup()` at end of run loop

---

*Convention analysis: 2026-04-07*
