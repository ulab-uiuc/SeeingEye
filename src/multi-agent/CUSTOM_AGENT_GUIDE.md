# Custom Agent Development Guide

This guide explains how to create custom agents in the multi-agent framework by analyzing the existing `SWEAgent` and `Manus` implementations.

## Agent Architecture Overview

The multi-agent framework uses a hierarchical inheritance structure:

```
BaseAgent (Abstract)
└── ReActAgent 
    └── ToolCallAgent
        ├── SWEAgent
        ├── Manus
        └── [Your Custom Agent]
```

## Core Components

### 1. Agent Class Structure

Every custom agent inherits from `ToolCallAgent` and must define these key attributes:

```python
from app.agent.toolcall import ToolCallAgent
from app.tool import ToolCollection

class MyCustomAgent(ToolCallAgent):
    # Required identification
    name: str = "my_agent"
    description: str = "Description of what your agent does"
    
    # Prompts (defined in separate prompt file)
    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT
    
    # Tool configuration
    available_tools: ToolCollection = ToolCollection(
        # Your tools here
    )
    special_tool_names: List[str] = ["terminate"]
    
    # Execution limits
    max_steps: int = 20
```

### 2. Prompt System

Agents use a two-prompt system:

#### System Prompt (`system_prompt`)
- Defines the agent's role, capabilities, and constraints
- Sets the overall behavior and personality
- Usually stored in `app/prompt/[agent_name].py`

#### Next Step Prompt (`next_step_prompt`) 
- Guides the agent's reasoning for each step
- Can be dynamic and context-aware
- Used between actions to maintain focus

## Creating a Custom Agent

### Step 1: Define Your Agent Class

Create `app/agent/my_agent.py`:

```python
from typing import List
from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.prompt.my_agent import SYSTEM_PROMPT, NEXT_STEP_PROMPT
from app.tool import Bash, StrReplaceEditor, Terminate, ToolCollection

class MyAgent(ToolCallAgent):
    """Custom agent for [your specific use case]"""
    
    name: str = "my_agent"
    description: str = "A specialized agent that [describe capabilities]"
    
    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT
    
    available_tools: ToolCollection = ToolCollection(
        Bash(),              # Command execution
        StrReplaceEditor(),  # File editing
        Terminate()          # Task completion
    )
    
    special_tool_names: List[str] = Field(default_factory=lambda: ["terminate"])
    max_steps: int = 25
```

### Step 2: Create Prompt File

Create `app/prompt/my_agent.py`:

```python
SYSTEM_PROMPT = """
You are [Agent Name], a specialized AI assistant designed to [specific purpose].

Your capabilities include:
- [Capability 1]
- [Capability 2] 
- [Capability 3]

Guidelines:
- [Important rule 1]
- [Important rule 2]
- Always use tools appropriately
- Terminate when task is complete
"""

NEXT_STEP_PROMPT = """
Based on the current situation, analyze what needs to be done next:
1. Assess the current state
2. Determine the most appropriate action
3. Use the right tool for the task
4. If the task is complete, use the terminate tool
"""
```

## Agent Types and Examples

### 1. Programming Agent (SWE-style)

**Characteristics:**
- File system navigation and editing
- Code execution and debugging
- Structured approach to software tasks

**Key Tools:**
- `Bash()` - Command execution
- `StrReplaceEditor()` - File editing with precise control
- `Terminate()` - Task completion

**Prompt Pattern:**
```python
SYSTEM_PROMPT = """You are an autonomous programmer working in a command line interface...
- File editor shows {WINDOW} lines at a time
- Proper indentation is required
- One tool call per response
- Include thought process before each action
"""
```

### 2. General-Purpose Agent (Manus-style)

**Characteristics:**
- Multi-modal capabilities
- Web browsing support
- MCP (Model Context Protocol) integration
- Human interaction capability

**Key Tools:**
- `PythonExecute()` - Code execution
- `BrowserUseTool()` - Web interaction  
- `StrReplaceEditor()` - File operations
- `AskHuman()` - Human consultation
- `MCPClientTool()` - External service integration

**Advanced Features:**
```python
class MyGeneralAgent(ToolCallAgent):
    # MCP integration
    mcp_clients: MCPClients = Field(default_factory=MCPClients)
    
    # Browser context management
    browser_context_helper: Optional[BrowserContextHelper] = None
    
    # Dynamic initialization
    @classmethod
    async def create(cls, **kwargs) -> "MyGeneralAgent":
        instance = cls(**kwargs)
        await instance.initialize_mcp_servers()
        return instance
```

## Advanced Patterns

### 1. Dynamic Tool Management

```python
async def connect_external_service(self, service_config):
    """Dynamically add tools from external services"""
    new_tools = await self.load_external_tools(service_config)
    self.available_tools.add_tools(*new_tools)
```

### 2. Context-Aware Prompting

```python
async def think(self) -> bool:
    """Override to provide dynamic context"""
    # Analyze recent actions
    recent_messages = self.memory.messages[-3:]
    
    # Adjust prompt based on context
    if self.detect_browser_usage(recent_messages):
        self.next_step_prompt = self.get_browser_focused_prompt()
    
    return await super().think()
```

### 3. State Management

```python
class StatefulAgent(ToolCallAgent):
    # Custom state tracking
    task_progress: Dict[str, Any] = Field(default_factory=dict)
    current_focus: str = "initialization"
    
    async def update_state(self, new_state_info):
        """Update agent's internal state"""
        self.task_progress.update(new_state_info)
        self.current_focus = self.determine_focus()
```

## Tool Integration

### Available Built-in Tools

```python
from app.tool import (
    Bash,                    # Command execution
    StrReplaceEditor,        # File editing
    PythonExecute,           # Python code execution
    BrowserUseTool,          # Web browsing
    AskHuman,               # Human interaction
    Terminate,              # Task completion
    MCPClientTool,          # External MCP services
    # ... and more
)
```

### Creating Custom Tools

```python
from app.tool.base import BaseTool

class MyCustomTool(BaseTool):
    name: str = "my_tool"
    description: str = "Does something specific"
    
    async def execute(self, **kwargs) -> str:
        # Your tool logic here
        return "Tool execution result"
```

## Configuration and Deployment

### 1. Agent Registration

Add your agent to the appropriate factory or registry:

```python
# In agent factory or main application
from app.agent.my_agent import MyAgent

available_agents = {
    "swe": SWEAgent,
    "manus": Manus, 
    "my_agent": MyAgent,  # Register your agent
}
```

### 2. Configuration

Agents can be configured through:
- Environment variables
- Configuration files (`config/config.toml`)
- Runtime parameters

```python
# Example configuration
class MyAgent(ToolCallAgent):
    @model_validator(mode="after")
    def configure_from_env(self) -> "MyAgent":
        # Load configuration from environment/config files
        self.max_steps = config.my_agent.max_steps
        return self
```

## Best Practices

### 1. Prompt Design
- **Be specific**: Clear role definition and constraints
- **Include examples**: Show expected behavior patterns
- **Set boundaries**: Define what the agent should/shouldn't do
- **Use consistent formatting**: Help the LLM understand structure

### 2. Tool Selection
- **Minimal viable set**: Start with essential tools only
- **Focused capability**: Tools should align with agent's purpose
- **Error handling**: Consider failure modes and recovery

### 3. State Management
- **Track progress**: Maintain awareness of task completion
- **Memory efficiency**: Clean up obsolete state information
- **Reproducible**: State should be clear and debuggable

### 4. Error Handling
```python
async def think(self) -> bool:
    try:
        return await super().think()
    except TokenLimitExceeded:
        # Handle token limits gracefully
        await self.summarize_and_continue()
    except Exception as e:
        logger.error(f"Agent {self.name} error: {e}")
        return False
```

## Testing Your Agent

```python
# Basic agent test
async def test_my_agent():
    agent = MyAgent()
    
    # Set up test scenario
    agent.memory.add_user_message("Test task description")
    
    # Execute steps
    for step in range(5):
        should_continue = await agent.think()
        if not should_continue:
            break
    
    # Verify results
    assert agent.state == AgentState.COMPLETED
```

## Example: Complete Custom Agent

Here's a complete example of a custom documentation agent:

```python
# app/agent/doc_agent.py
from typing import List
from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.prompt.doc_agent import SYSTEM_PROMPT, NEXT_STEP_PROMPT
from app.tool import StrReplaceEditor, Terminate, ToolCollection, PythonExecute

class DocumentationAgent(ToolCallAgent):
    """Agent specialized in creating and maintaining documentation"""
    
    name: str = "doc_agent"
    description: str = "Generates and maintains technical documentation"
    
    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT
    
    available_tools: ToolCollection = ToolCollection(
        StrReplaceEditor(),  # For writing documentation files
        PythonExecute(),     # For analyzing code structure
        Terminate()
    )
    
    special_tool_names: List[str] = Field(default_factory=lambda: ["terminate"])
    max_steps: int = 15
    
    # Custom configuration
    doc_format: str = "markdown"
    include_examples: bool = True
```

```python
# app/prompt/doc_agent.py
SYSTEM_PROMPT = """
You are a Documentation Agent, specialized in creating clear, comprehensive technical documentation.

Your responsibilities:
- Analyze code structure and functionality
- Generate well-formatted documentation
- Maintain consistency in documentation style
- Include practical examples when helpful

Guidelines:
- Use clear, concise language
- Follow markdown formatting standards
- Include code examples where appropriate
- Structure information logically
- Always verify documentation accuracy
"""

NEXT_STEP_PROMPT = """
Analyze the current documentation task:
1. What documentation needs to be created or updated?
2. What information do you need to gather?
3. What's the most logical next step?
4. Use appropriate tools to accomplish the task
"""
```

This guide provides the foundation for creating custom agents tailored to your specific needs while following the established patterns in the multi-agent framework.