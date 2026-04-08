# Testing Patterns

**Analysis Date:** 2026-04-07

## Test Framework

**Runner:**
- pytest v8.3.5
- pytest-asyncio v0.25.3 for async test support
- Config: No pytest.ini or pyproject.toml with pytest config found - uses defaults

**Assertion Library:**
- Standard Python assertions (assert keyword)
- Pydantic model validation for data model testing

**Run Commands:**
```bash
pytest                          # Run all tests
pytest --asyncio-mode=auto     # Run async tests
pytest -v                       # Verbose output
pytest -s                       # Show print statements
pytest --cov                    # Coverage report (if plugin installed)
```

## Test File Organization

**Location:**
- No test files detected in codebase - testing appears not yet implemented
- Test files would be organized separately from source: `tests/` directory at project root
- Unit test pattern: `tests/app/test_agent_base.py`, `tests/app/test_config.py`
- Integration tests: `tests/integration/test_flow_executor.py`

**Naming:**
- Would follow pytest convention: `test_*.py` or `*_test.py`
- Test functions: `test_` prefix, e.g., `test_parse_tool_calls_multiple_formats()`
- Async tests: `async def test_agent_step():`

**Structure:**
```
tests/
├── __init__.py
├── conftest.py                 # Shared fixtures
├── app/
│   ├── test_config.py
│   ├── test_llm.py
│   ├── agent/
│   │   ├── test_base.py
│   │   ├── test_react.py
│   │   └── test_toolcall.py
│   └── flow/
│       └── test_flow_executor.py
└── integration/
    └── test_e2e_flows.py
```

## Test Structure

**Suite Organization:**
```python
# Based on existing code patterns observed in docstrings and error handling

import pytest
from app.agent.base import BaseAgent
from app.schema import AgentState, Message


class TestBaseAgent:
    """Tests for BaseAgent initialization and state management"""

    @pytest.fixture
    def agent(self):
        """Create a test agent instance"""
        # Would create minimal agent for testing
        pass

    def test_initialization(self, agent):
        """Test agent initializes with correct defaults"""
        assert agent.state == AgentState.IDLE
        assert agent.current_step == 0
        assert agent.max_steps == 10

    @pytest.mark.asyncio
    async def test_state_context_transitions(self, agent):
        """Test state transitions in context manager"""
        async with agent.state_context(AgentState.RUNNING):
            assert agent.state == AgentState.RUNNING
        assert agent.state == AgentState.IDLE

    def test_update_memory_with_valid_role(self, agent):
        """Test adding message to memory with valid roles"""
        agent.update_memory("user", "Hello")
        assert len(agent.memory.messages) == 1
        assert agent.memory.messages[0].role == "user"
```

**Patterns:**
- Setup/fixture pattern: `@pytest.fixture` for test data
- Async test pattern: `@pytest.mark.asyncio` for async test functions
- Assertion pattern: `assert` statements for conditions
- Class-based organization: `class Test*` for grouping related tests
- Parameterization: Would use `@pytest.mark.parametrize` for multiple test cases

## Mocking

**Framework:** unittest.mock (from Python standard library)

**Patterns:**
```python
# Based on tool call parsing patterns observed in app/agent/toolcall.py

from unittest.mock import Mock, patch, AsyncMock

# Mock LLM responses
@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "test response"}}]
    }
    return llm

# Mock tool calls
def test_parse_tool_calls_with_mock_response():
    mock_response = '<tool_call>{"name": "test", "arguments": "{}"}</tool_call>'
    tool_calls = parse_tool_calls_multiple_formats(mock_response)
    assert len(tool_calls) > 0

# Patch external dependencies
@patch('app.llm.OpenAI')
def test_llm_initialization(mock_openai):
    mock_openai.return_value.chat.completions.create.return_value = Mock()
    # Test code
    pass
```

**What to Mock:**
- External LLM APIs: OpenAI, Azure OpenAI, together.ai calls
- File I/O: config file loading, log file creation
- Browser instances and Playwright interactions
- Network requests: search API calls, HTTP requests
- Docker/Sandbox operations: container creation, execution

**What NOT to Mock:**
- Pydantic model validation - test with real models
- Core agent logic: BaseAgent, ReActAgent methods
- Message creation and memory management
- State transitions and context managers
- Tool call parsing logic - test with various formats

## Fixtures and Factories

**Test Data:**
```python
# Based on schema patterns observed in app/schema.py

import pytest
from app.schema import Message, Memory, AgentState

@pytest.fixture
def test_message():
    """Factory for creating test messages"""
    return Message.user_message("Test user input")

@pytest.fixture
def test_memory():
    """Factory for creating test memory with sample messages"""
    memory = Memory()
    memory.add_message(Message.system_message("Test system"))
    memory.add_message(Message.user_message("Test user"))
    memory.add_message(Message.assistant_message("Test assistant response"))
    return memory

@pytest.fixture
def mock_llm_config():
    """Factory for creating mock LLM configuration"""
    from app.config import LLMSettings
    return LLMSettings(
        model="gpt-4",
        base_url="http://localhost",
        api_key="test-key",
        temperature=0.7,
        max_tokens=2048
    )

@pytest.fixture
def mock_agent(mock_llm_config):
    """Factory for creating test agent"""
    from app.agent.react import ReActAgent
    agent = ReActAgent(
        name="test_agent",
        system_prompt="You are a test agent",
        max_steps=3
    )
    return agent

# Parameterized fixtures for testing multiple scenarios
@pytest.fixture(params=[
    '<tool_call>{"name": "test", "arguments": "{}"}</tool_call>',
    '[{"name": "test", "arguments": "{}"}]',
    '{"name": "test", "arguments": "{}"}'
])
def tool_call_formats(request):
    """Test data for different tool call formats"""
    return request.param
```

**Location:**
- `tests/conftest.py`: Shared fixtures for entire test suite
- `tests/app/conftest.py`: App-specific fixtures
- Fixtures defined at module level in conftest files
- Inline fixtures in test modules for test-specific data

## Coverage

**Requirements:**
- No coverage enforcement detected in codebase
- Would recommend: minimum 70% coverage for critical paths
- Highest priority: BaseAgent, ReActAgent, flow execution

**View Coverage:**
```bash
pytest --cov=src/multi-agent/app --cov-report=html
open htmlcov/index.html

pytest --cov=src/multi-agent/app --cov-report=term-missing
```

## Test Types

**Unit Tests:**
- Scope: Individual functions and methods
- Approach: Test with mocked dependencies
- Example targets: `parse_tool_calls_multiple_formats()`, `Message.user_message()`, `Memory.add_message()`
- Location: `tests/app/test_*.py`
- Pattern: Fast execution, no I/O, focused assertions

```python
def test_parse_tool_calls_multiple_formats():
    """Test tool call parsing with various input formats"""
    # Arrange
    content = '<tool_call>{"name": "search", "arguments": "{}"}</tool_call>'

    # Act
    tool_calls = parse_tool_calls_multiple_formats(content)

    # Assert
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "search"
```

**Integration Tests:**
- Scope: Multiple components working together
- Approach: Test with real dependencies (except external APIs)
- Example targets: Agent stepping with tool execution, flow execution with agents
- Location: `tests/integration/test_*.py`
- Pattern: Slower execution, may involve I/O, test component interaction

```python
@pytest.mark.asyncio
async def test_agent_step_with_memory():
    """Test agent step processes memory correctly"""
    # Create real agent and memory
    agent = TestAgent(...)
    agent.update_memory("user", "test input")

    # Execute step
    result = await agent.step()

    # Verify memory updated
    assert len(agent.memory.messages) >= 2
```

**E2E Tests:**
- Scope: Complete workflows from input to output
- Approach: Full execution with real or mocked external APIs
- Framework: Not yet implemented - would use pytest with pytest-asyncio
- Example: Test complete flow execution from initial request to final output

## Common Patterns

**Async Testing:**
```python
import pytest

# Mark test as async
@pytest.mark.asyncio
async def test_async_agent_run():
    """Test async agent execution"""
    agent = create_test_agent()
    result = await agent.run("test request")
    assert result is not None

# Using async fixtures
@pytest.fixture
async def async_agent():
    agent = create_test_agent()
    yield agent
    # Cleanup
    await agent.cleanup()

# Testing with asyncio.gather
@pytest.mark.asyncio
async def test_concurrent_agent_steps():
    agent1 = create_test_agent()
    agent2 = create_test_agent()
    results = await asyncio.gather(
        agent1.step(),
        agent2.step()
    )
    assert len(results) == 2
```

**Error Testing:**
```python
def test_invalid_message_role():
    """Test error handling with invalid role"""
    agent = create_test_agent()

    # Should raise ValueError
    with pytest.raises(ValueError, match="Unsupported message role"):
        agent.update_memory("invalid_role", "content")

def test_agent_state_error_transition():
    """Test error state transitions"""
    agent = create_test_agent()
    agent.state = AgentState.RUNNING

    # Simulate error in context manager
    try:
        async with agent.state_context(AgentState.ERROR):
            raise Exception("Test error")
    except Exception:
        pass

    assert agent.state == AgentState.RUNNING  # Reverted to previous state

@pytest.mark.asyncio
async def test_force_termination_on_max_steps():
    """Test force termination when max steps exceeded"""
    agent = TestAgent(max_steps=1)

    with pytest.raises(RuntimeError):
        result = await agent.run("test")
        # Verify agent was force terminated
        assert agent.state == AgentState.FINISHED
```

**Parametrized Tests:**
```python
@pytest.mark.parametrize("role,expected_success", [
    ("user", True),
    ("system", True),
    ("assistant", True),
    ("tool", True),
    ("invalid", False)
])
def test_update_memory_with_various_roles(role, expected_success):
    """Test memory update with various message roles"""
    agent = create_test_agent()

    if expected_success:
        agent.update_memory(role, "test content")
        assert len(agent.memory.messages) > 0
    else:
        with pytest.raises(ValueError):
            agent.update_memory(role, "test content")

@pytest.mark.parametrize("tool_call_format", [
    '<tool_call>{"name": "test", "arguments": "{}"}</tool_call>',
    '[{"name": "test", "arguments": "{}"}]',
    '{"name": "test", "arguments": "{}"}'
])
def test_parse_multiple_formats(tool_call_format):
    """Test parsing of multiple tool call formats"""
    tool_calls = parse_tool_calls_multiple_formats(tool_call_format)
    assert len(tool_calls) > 0
```

## Current Testing Status

**Implementation Status:**
- No test files found in repository
- Testing infrastructure not yet implemented
- Pytest installed in requirements.txt (v8.3.5, pytest-asyncio v0.25.3)
- Ready to implement comprehensive test suite

**Recommended Implementation Order:**
1. Unit tests for schema models (Message, Memory, AgentState)
2. Unit tests for parsing functions (tool call parsing in `app/agent/toolcall.py`)
3. Unit tests for configuration loading (app/config.py)
4. Integration tests for BaseAgent and ReActAgent
5. Integration tests for flows (flow_executor.py)
6. E2E tests for complete workflows

---

*Testing analysis: 2026-04-07*
