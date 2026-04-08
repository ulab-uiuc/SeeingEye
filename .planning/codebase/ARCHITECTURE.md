# Architecture

**Analysis Date:** 2026-04-07

## Pattern Overview

**Overall:** Multi-Agent Orchestration with Iterative Refinement

**Key Characteristics:**
- Modular agent-based architecture supporting multiple specialized agents
- Asynchronous execution model for concurrent processing
- Tool-based action execution with flexible parsing strategies
- Configuration-driven LLM backend selection (OpenAI, DashScope, vLLM, Together AI, Bedrock)
- Structured Intermediate Representation (SIR) for visual analysis refinement
- Flow-based orchestration enabling agent collaboration and iteration

## Layers

**Configuration Layer:**
- Purpose: Central configuration management for LLM models, tools, sandbox, browser, and MCP settings
- Location: `src/multi-agent/app/config.py`, `src/multi-agent/config/config.toml`
- Contains: Pydantic models (AppConfig, LLMSettings, FlowSettings, MCPSettings), TOML config parsing, environment variable handling
- Depends on: Pydantic, tomllib/tomli
- Used by: All application layers for accessing configuration

**Entry Point Layer:**
- Purpose: Application bootstrap and command-line interface
- Location: `src/multi-agent/main.py`
- Contains: Argument parsing, agent instantiation, input handling (prompt/image), async execution loop
- Depends on: TranslatorAgent, logger, configuration
- Used by: End users running the CLI

**Agent Core Layer:**
- Purpose: Abstract agent execution framework with state management and memory handling
- Location: `src/multi-agent/app/agent/base.py`
- Contains: BaseAgent class with state machines (IDLE, RUNNING, FINISHED, ERROR), memory management, step-based execution loop, lifecycle management
- Depends on: LLM, Memory, Message schema, logger
- Used by: All specialized agent implementations

**Agent Specialization Layer:**
- Purpose: Domain-specific agent implementations extending BaseAgent
- Location: `src/multi-agent/app/agent/`
- Contains:
  - `translator.py`: TranslatorAgent - Vision-Language model for image captioning with SIR management
  - `toolcall.py`: ToolCallAgent - Supports tool invocation with multiple parsing strategies (XML/JSON)
  - `react.py`: ReActAgent - ReAct pattern with reasoning and acting loops
  - `text_only_reasoning.py`: TextOnlyReasoningAgent - Pure text reasoning without vision
  - `mcp.py`: MCPAgent - Model Context Protocol support
  - `browser.py`: BrowserAgent - Web browsing capabilities
- Depends on: BaseAgent, LLM, tools, prompts, schema
- Used by: Flow orchestration and direct instantiation

**LLM Integration Layer:**
- Purpose: Unified LLM interface abstracting multiple provider backends
- Location: `src/multi-agent/app/llm.py`
- Contains: LLM class with provider factory, request/response handling, error handling with retry logic, token counting, streaming support
- Depends on: Configuration, model providers (OpenAI, DashScope, vLLM, Together, Bedrock), tenacity (retry decorators)
- Used by: All agents for LLM calls

**Tool Execution Layer:**
- Purpose: Extensible tool system for agent actions
- Location: `src/multi-agent/app/tool/`
- Contains:
  - `base.py`: BaseTool abstract class defining tool interface
  - `tool_collection.py`: ToolCollection managing tool registry and execution
  - Vision tools: `smart_grid_caption.py`, `ocr.py`, `crop_and_caption.py`, `read_table.py`
  - Action tools: `bash.py`, `file_operators.py`, `str_replace_editor.py`
  - Termination tools: `terminate_and_output_caption.py`, `terminate_and_answer.py`
  - Search: `search/` (web search integration)
  - Utility: `think.py`, `create_chat_completion.py`, `planning.py`
- Depends on: Schema, logger, configuration
- Used by: ToolCallAgent for action execution

**Flow Orchestration Layer:**
- Purpose: Multi-agent workflow orchestration and iteration management
- Location: `src/multi-agent/app/flow/`
- Contains:
  - `base.py`: BaseFlow - Abstract base for execution flows supporting multiple agents
  - `iterative_refinement.py`: IterativeRefinementFlow - Primary pattern supporting translator/reasoning agent collaboration with SIR refinement across iterations
  - `flow_executor.py`: FlowExecutor - Execution runtime for flow instances
  - `flow_factory.py`: FlowFactory - Factory for instantiating flows
  - `planning.py`: PlanningFlow - Task planning flow variant
- Depends on: BaseAgent, BaseFlow, LLM, configuration
- Used by: Main entry point for execution

**Schema & Data Layer:**
- Purpose: Core data models and type definitions
- Location: `src/multi-agent/app/schema.py`
- Contains: Role (SYSTEM, USER, ASSISTANT, TOOL), AgentState (IDLE, RUNNING, FINISHED, ERROR), Message, ToolCall, Function, Memory
- Depends on: Pydantic, enum
- Used by: All layers for type safety and data validation

**Prompt & Memory Layer:**
- Purpose: Prompt engineering and agent memory management
- Location: `src/multi-agent/app/prompt/`, `schema.py` (Memory class)
- Contains: System prompts, step prompts, final step prompts organized by agent type; Message-based conversation history
- Depends on: Schema
- Used by: Agents for instruction and context

**Utilities & Support:**
- Purpose: Cross-cutting concerns and helper functions
- Location: `src/multi-agent/app/utils/`, `src/multi-agent/app/logger.py`, `src/multi-agent/app/exceptions.py`
- Contains: Token counting, LLM setup validators, logging, sandbox client, MCP protocol support
- Depends on: Configuration, schema, external libraries
- Used by: All layers

## Data Flow

**Image Analysis with Iterative Refinement (IterativeRefinementFlow):**

1. **Input Phase**: User provides prompt and image path (or base64)
2. **Initialization**: Main creates TranslatorAgent and ReasoningAgent instances
3. **Iteration 1 (Translation)**:
   - Translator receives image + user question
   - Calls smart_grid_caption tool to analyze visual grid/layout
   - Builds Structured Intermediate Representation (SIR) capturing visual details
   - Returns detailed visual description
4. **Iteration 1 (Reasoning)**:
   - Reasoning agent receives SIR + original question
   - Analyzes context and determines if answer sufficient or feedback needed
   - Returns either: final answer OR feedback for refinement
5. **Iteration 2+ (Conditional Refinement)**:
   - If feedback provided, translator receives previous SIR + feedback
   - Refines SIR with additional visual analysis
   - Process repeats until max iterations or final answer
6. **Output Phase**: Final answer from reasoning agent returned to user

**State Management:**
- Agent state tracked via AgentState enum (IDLE → RUNNING → FINISHED/ERROR)
- Memory maintained as ordered Message list in each agent
- SIR evolves across translator iterations as previous_sir → current_sir
- Configuration controls max_iterations and flow_type

## Key Abstractions

**BaseAgent:**
- Purpose: Defines agent contract and lifecycle
- Examples: `TranslatorAgent`, `ReActAgent`, `ToolCallAgent`, `TextOnlyReasoningAgent`
- Pattern: Template method pattern with abstract `step()` method; lifecycle: initialize → run loop → cleanup

**BaseFlow:**
- Purpose: Orchestrates multi-agent execution with shared context
- Examples: `IterativeRefinementFlow`, `PlanningFlow`
- Pattern: Strategy pattern for different flow types; agents managed by key-based dictionary

**BaseTool:**
- Purpose: Extensible action interface for agents
- Examples: `SmartGridCaption`, `OCR`, `ReadTable`, `Bash`, `StrReplaceEditor`
- Pattern: Factory pattern via ToolCollection; lazy imports for performance

**LLM Provider Abstraction:**
- Purpose: Unified interface across heterogeneous LLM backends
- Implementations: OpenAI, Azure OpenAI, DashScope, vLLM, Together AI, AWS Bedrock, Ollama
- Pattern: Factory pattern with configuration-driven selection

## Entry Points

**Primary Entry Point:**
- Location: `src/multi-agent/main.py::main()`
- Triggers: `python -m main --prompt "..." --image "..."`
- Responsibilities: Parse args, create TranslatorAgent, instantiate IterativeRefinementFlow, handle async execution, manage cleanup

**Agent Creation Pattern:**
- Location: Agent subclass constructors (e.g., `TranslatorAgent.__init__()`)
- Responsibilities: Initialize LLM with correct config_name, set up prompts, configure available tools, set max_steps

**Tool Execution Trigger:**
- Location: `ToolCallAgent.step()` → tool call parsing → tool execution
- Responsibilities: Parse LLM response for tool invocations, validate tool existence, execute with arguments, collect results

## Error Handling

**Strategy:** Layered error handling with context-aware state transitions

**Patterns:**
- **Agent Level**: `state_context()` async context manager transitions to ERROR state on exception, then reverts to previous state
- **LLM Level**: Retry decorator (`@retry`) with exponential backoff and rate limit handling; raises TokenLimitExceeded when quota exceeded
- **Tool Level**: ToolError exception for tool-specific failures; captured and converted to tool message for LLM feedback
- **Flow Level**: Exceptions bubble up with logging; cleanup called in finally blocks

**Custom Exceptions:**
- `ToolError`: Tool execution failures
- `OpenManusError`: Base exception for agent framework errors
- `TokenLimitExceeded`: LLM token quota exceeded

## Cross-Cutting Concerns

**Logging:**
- Framework: Custom logger in `src/multi-agent/app/logger.py`
- Pattern: Module-level logger instance; structured logging with context prefix (emoji indicators like 📝 for SIR updates, ⚡ for execution)
- Implementation: Python logging module with custom formatting

**Validation:**
- Pydantic models throughout for schema validation (BaseAgent, LLMSettings, Message, etc.)
- Custom validators using `@model_validator` for complex initialization logic
- Configuration validation during AppConfig construction

**Authentication:**
- API keys managed via environment variables (OPENAI_API_KEY, DASHSCOPE_API_KEY, TOGETHER_API_KEY, etc.)
- Configuration system reads from env with fallback to config file
- Sensitive values never logged

**Asynchronous Execution:**
- Agents use async/await throughout (agent.run(), agent.step(), etc.)
- LLM calls made via AsyncOpenAI/AsyncAzureOpenAI clients
- Flow execution orchestrated with asyncio event loop in main.py
- Proper cleanup ensured via async context managers and finally blocks

---

*Architecture analysis: 2026-04-07*
