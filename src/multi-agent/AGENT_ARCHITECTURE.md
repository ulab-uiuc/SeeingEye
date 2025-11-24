# Multi-Agent Framework Architecture

This document explains how the agent execution system works, focusing on the `run()`, `step()`, `think()`, and `act()` methods.

## Overview

The framework uses a hierarchical architecture with three main classes:

1. **BaseAgent** - Abstract base class managing state, memory, and execution loop
2. **ReActAgent** - Implements ReAct pattern (Reasoning + Acting) 
3. **ToolCallAgent** - Concrete implementation with LLM tool calling

## Architecture Hierarchy

```
BaseAgent (base.py)
└── ReActAgent (react.py) 
    └── ToolCallAgent (toolcall.py)
        ├── VQAAgent (vqa.py)
        ├── TranslatorAgent (translator.py)
        └── Other specific agents...
```

## Execution Flow

### 1. Entry Point (`main.py`)

```python
# Create agent
agent = await Manus.create()

# Execute with user prompt  
await agent.run(prompt)
```

### 2. Main Execution Loop (`BaseAgent.run()`)

**File: `app/agent/base.py` lines 116-154**

```python
async def run(self, request: Optional[str] = None) -> str:
    # 1. Validate agent is in IDLE state
    if self.state != AgentState.IDLE:
        raise RuntimeError(f"Cannot run agent from state: {self.state}")

    # 2. Add user request to memory if provided
    if request:
        self.update_memory("user", request)

    # 3. Execute step-by-step loop
    results: List[str] = []
    async with self.state_context(AgentState.RUNNING):
        while (
            self.current_step < self.max_steps and 
            self.state != AgentState.FINISHED
        ):
            self.current_step += 1
            logger.info(f"Executing step {self.current_step}/{self.max_steps}")
            
            # 4. Execute one reasoning cycle
            step_result = await self.step()
            
            # 5. Check for stuck state and handle
            if self.is_stuck():
                self.handle_stuck_state()
            
            results.append(f"Step {self.current_step}: {step_result}")

    return "\n".join(results) if results else "No steps executed"
```

**Key Points:**
- **State Management**: Ensures agent starts in IDLE state, transitions to RUNNING
- **Memory Management**: Adds user request to agent's memory
- **Loop Control**: Continues until max_steps reached OR agent sets state to FINISHED
- **Step Execution**: Calls `step()` method for each iteration
- **Stuck Detection**: Detects repeated outputs and adds guidance prompts
- **Result Formatting**: Returns formatted string with all step results

### 3. ReAct Pattern (`ReActAgent.step()`)

**File: `app/agent/react.py` lines 33-38**

```python
async def step(self) -> str:
    """Execute a single step: think and act."""
    should_act = await self.think()
    if not should_act:
        return "Thinking complete - no action needed"
    return await self.act()
```

**Key Points:**
- **Think First**: Calls `think()` to process current state and decide actions
- **Conditional Acting**: Only calls `act()` if thinking determines action is needed
- **Return Control**: Returns result string to be logged by BaseAgent

### 4. Decision Making (`ToolCallAgent.think()`)

**File: `app/agent/toolcall.py` lines 39-129**

```python
async def think(self) -> bool:
    # 1. Add next step prompt to memory
    if self.next_step_prompt:
        user_msg = Message.user_message(self.next_step_prompt)
        self.messages += [user_msg]

    # 2. Query LLM with tools available
    response = await self.llm.ask_tool(
        messages=self.messages,
        system_msgs=[Message.system_message(self.system_prompt)],
        tools=self.available_tools.to_params(),
        tool_choice=self.tool_choices,
    )

    # 3. Extract tool calls from response
    self.tool_calls = response.tool_calls if response.tool_calls else []
    content = response.content if response.content else ""

    # 4. Add assistant message to memory
    assistant_msg = (
        Message.from_tool_calls(content=content, tool_calls=self.tool_calls)
        if self.tool_calls
        else Message.assistant_message(content)
    )
    self.memory.add_message(assistant_msg)

    # 5. Return whether to proceed with action
    return bool(self.tool_calls)  # Act if tools were selected
```

**Key Points:**
- **LLM Consultation**: Sends current conversation + available tools to LLM
- **Tool Selection**: LLM decides which tools (if any) to use based on context
- **Memory Update**: Adds LLM's response (including tool calls) to memory
- **Decision Return**: Returns True if tools selected, False if no action needed

### 5. Action Execution (`ToolCallAgent.act()`)

**File: `app/agent/toolcall.py` lines 131-164**

```python
async def act(self) -> str:
    # 1. Handle case with no tool calls
    if not self.tool_calls:
        return self.messages[-1].content or "No content or commands to execute"

    # 2. Execute each selected tool
    results = []
    for command in self.tool_calls:
        # Execute tool with arguments
        result = await self.execute_tool(command)
        
        # Add tool result to memory
        tool_msg = Message.tool_message(
            content=result,
            tool_call_id=command.id,
            name=command.function.name,
        )
        self.memory.add_message(tool_msg)
        results.append(result)

    return "\n\n".join(results)
```

**Key Points:**
- **Tool Execution**: Calls each selected tool with parsed arguments
- **Result Handling**: Processes tool outputs and adds to memory
- **Memory Updates**: Each tool result becomes a tool message in conversation
- **Result Aggregation**: Combines all tool results into single response

### 6. Tool Execution (`ToolCallAgent.execute_tool()`)

**File: `app/agent/toolcall.py` lines 166-208**

```python
async def execute_tool(self, command: ToolCall) -> str:
    # 1. Validate command format
    name = command.function.name
    if name not in self.available_tools.tool_map:
        return f"Error: Unknown tool '{name}'"

    # 2. Parse arguments
    args = json.loads(command.function.arguments or "{}")

    # 3. Execute tool
    result = await self.available_tools.execute(name=name, tool_input=args)

    # 4. Handle special tools (like terminate)
    await self._handle_special_tool(name=name, result=result)

    # 5. Format result for observation
    observation = (
        f"Observed output of cmd `{name}` executed:\n{str(result)}"
        if result
        else f"Cmd `{name}` completed with no output"
    )
    return observation
```

**Key Points:**
- **Tool Validation**: Ensures tool exists in available tools
- **Argument Parsing**: Converts JSON arguments to Python objects
- **Tool Execution**: Calls actual tool implementation
- **Special Handling**: Tools like `terminate` can set agent state to FINISHED
- **Result Formatting**: Wraps tool output in observation format

## Special Tool Handling

### Termination Tools

```python
async def _handle_special_tool(self, name: str, result: Any, **kwargs):
    if self._is_special_tool(name):
        if self._should_finish_execution(name=name, result=result, **kwargs):
            logger.info(f"🏁 Special tool '{name}' has completed the task!")
            self.state = AgentState.FINISHED  # This stops the main loop
```

**Special tools** (like `terminate_and_answer`) can set `self.state = AgentState.FINISHED`, which breaks the main execution loop in `BaseAgent.run()`.

## Memory Management

The agent maintains a conversation history in `self.memory.messages`:

```python
# User input
Message.user_message(content, base64_image=image_data)

# System instructions  
Message.system_message(system_prompt)

# Assistant responses
Message.assistant_message(content)
Message.from_tool_calls(content, tool_calls)  # With tool calls

# Tool results
Message.tool_message(content, tool_call_id, name)
```

**Images** are handled through the `base64_image` field in messages, allowing multimodal interactions.

## Example Execution Flow

For a VQA agent answering a visual question:

1. **`main.py`**: `await agent.run("What is 2+2?")`
2. **`BaseAgent.run()`**: Adds question to memory, starts loop
3. **Step 1**:
   - **`ReActAgent.step()`**: Calls think() then act()
   - **`ToolCallAgent.think()`**: LLM sees image+question, decides to use `python_execute`
   - **`ToolCallAgent.act()`**: Executes `python_execute("2+2")`, gets result "4"
4. **Step 2**:
   - **`ToolCallAgent.think()`**: LLM sees calculation result, decides to use `terminate_and_answer`  
   - **`ToolCallAgent.act()`**: Executes `terminate_and_answer("4", "high", "Simple arithmetic")`, sets state to FINISHED
5. **`BaseAgent.run()`**: Loop exits due to FINISHED state, returns formatted results

## Key Features

- **Modular Design**: Each class has specific responsibilities
- **State Management**: Automatic state transitions and error handling
- **Memory Persistence**: Full conversation history including images
- **Tool Integration**: Seamless LLM-to-tool execution pipeline
- **Error Handling**: Robust error handling at each layer
- **Logging**: Comprehensive logging for debugging and monitoring
- **Stuck Detection**: Automatic detection and handling of repeated behaviors

## Agent Configuration

Each agent can be configured with:
- **`max_steps`**: Maximum execution steps before termination
- **`available_tools`**: Tools the agent can use
- **`special_tool_names`**: Tools that can terminate execution
- **`system_prompt`**: Instructions for the agent's role
- **`next_step_prompt`**: Guidance for each thinking step