# Codebase Structure

**Analysis Date:** 2026-04-07

## Directory Layout

```
SeeingEye/
├── src/
│   ├── multi-agent/                    # Main application package
│   │   ├── main.py                     # CLI entry point
│   │   ├── setup.py                    # Package setup
│   │   ├── app/                        # Core application logic
│   │   │   ├── __init__.py
│   │   │   ├── agent/                  # Agent implementations
│   │   │   │   ├── base.py             # BaseAgent abstract class
│   │   │   │   ├── translator.py       # TranslatorAgent (vision-language)
│   │   │   │   ├── toolcall.py         # ToolCallAgent (tool invocation)
│   │   │   │   ├── react.py            # ReActAgent (reasoning pattern)
│   │   │   │   ├── text_only_reasoning.py  # TextOnlyReasoningAgent
│   │   │   │   ├── mcp.py              # MCPAgent
│   │   │   │   ├── browser.py          # BrowserAgent
│   │   │   │   ├── manus.py            # ManusAgent
│   │   │   │   ├── vqa.py              # VQAAgent
│   │   │   │   └── deprecated/         # Deprecated agent implementations
│   │   │   ├── flow/                   # Flow orchestration
│   │   │   │   ├── base.py             # BaseFlow abstract class
│   │   │   │   ├── iterative_refinement.py  # IterativeRefinementFlow (primary)
│   │   │   │   ├── planning.py         # PlanningFlow
│   │   │   │   ├── flow_executor.py    # Flow execution runtime
│   │   │   │   └── flow_factory.py     # Factory for flow instantiation
│   │   │   ├── tool/                   # Tool implementations
│   │   │   │   ├── base.py             # BaseTool abstract class
│   │   │   │   ├── tool_collection.py  # ToolCollection manager
│   │   │   │   ├── smart_grid_caption.py  # Vision grid analysis
│   │   │   │   ├── ocr.py              # Optical character recognition
│   │   │   │   ├── read_table.py       # Table reading from images
│   │   │   │   ├── crop_and_caption.py # Image cropping with captioning
│   │   │   │   ├── terminate_and_output_caption.py  # Termination with caption
│   │   │   │   ├── terminate_and_answer.py         # Termination with answer
│   │   │   │   ├── think.py            # Reasoning/thinking tool
│   │   │   │   ├── bash.py             # Shell command execution
│   │   │   │   ├── file_operators.py   # File operations
│   │   │   │   ├── str_replace_editor.py  # Text replacement
│   │   │   │   ├── create_chat_completion.py  # LLM chat calls
│   │   │   │   ├── planning.py         # Planning tool
│   │   │   │   ├── crawl4ai.py         # Web crawling
│   │   │   │   ├── locate.py           # UI element location
│   │   │   │   ├── browser_use_tool.py # Browser automation
│   │   │   │   ├── split_patch.py      # Image splitting
│   │   │   │   ├── ask_human.py        # Human interaction
│   │   │   │   ├── python_execute.py   # Python code execution
│   │   │   │   ├── mcp_tool.py         # MCP protocol tool
│   │   │   │   ├── search/             # Search tool
│   │   │   │   └── chart_visualization/  # Chart generation
│   │   │   │       ├── src/
│   │   │   │       └── test/
│   │   │   ├── prompt/                 # Prompt templates
│   │   │   │   ├── __init__.py
│   │   │   │   ├── translator.py       # Translator agent prompts
│   │   │   │   ├── toolcall.py         # Tool call prompts
│   │   │   │   └── ...other prompts
│   │   │   ├── mcp/                    # Model Context Protocol support
│   │   │   │   └── server.py
│   │   │   ├── sandbox/                # Execution sandbox
│   │   │   │   ├── client.py
│   │   │   │   ├── core/               # Sandbox core
│   │   │   │   │   └── exceptions.py
│   │   │   │   └── ...
│   │   │   ├── utils/                  # Utility modules
│   │   │   │   ├── agent_utils.py      # Agent helper functions
│   │   │   │   ├── log_save.py         # Logging utilities
│   │   │   │   ├── vllm_setup.py       # vLLM configuration
│   │   │   │   └── ...
│   │   │   ├── config.py               # Configuration management (Pydantic models)
│   │   │   ├── llm.py                  # LLM interface & provider abstraction
│   │   │   ├── vllm_provider.py        # vLLM provider implementation
│   │   │   ├── bedrock.py              # AWS Bedrock integration
│   │   │   ├── schema.py               # Data models (Message, AgentState, etc.)
│   │   │   ├── logger.py               # Logging configuration
│   │   │   ├── exceptions.py           # Custom exceptions
│   │   │   └── token_counter.py        # Token counting utilities
│   │   ├── config/                     # Configuration files
│   │   │   ├── config.toml             # Main application config (multi-model setup)
│   │   │   └── mcp.json                # MCP server configurations
│   │   ├── protocol/                   # Protocol implementations
│   │   │   └── a2a/                    # Agent-to-agent protocol
│   │   │       ├── app/
│   │   │       └── __init__.py
│   │   └── workspace/                  # Runtime workspace directory
│   ├── plan/                           # Task planning directory
│   └── config.py                       # Root-level config (version check)
├── assets/                             # Asset files
├── logs/                               # Runtime logs
├── utils/                              # Utility scripts
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
└── CLAUDE.md                           # Claude Code instructions
```

## Directory Purposes

**src/multi-agent/:**
- Purpose: Complete multi-agent application package
- Entry point: `main.py` for CLI execution
- Core modules: agent, flow, tool subsystems

**src/multi-agent/app/agent/:**
- Purpose: Agent implementations extending BaseAgent
- Contains: Specialized agents for different reasoning patterns and domains
- Architecture: Inheritance hierarchy (BaseAgent → ReActAgent → ToolCallAgent → TranslatorAgent)

**src/multi-agent/app/flow/:**
- Purpose: Multi-agent workflow orchestration
- Contains: Flow implementations and execution infrastructure
- Key: IterativeRefinementFlow for vision + reasoning collaboration

**src/multi-agent/app/tool/:**
- Purpose: Extensible tool system for agent actions
- Contains: Vision tools, action tools, termination tools, search, utilities
- Pattern: Lazy loading via `__getattr__` in `__init__.py` to reduce startup time

**src/multi-agent/app/prompt/:**
- Purpose: Prompt engineering for each agent type
- Contains: System prompts, next-step prompts, final-step prompts as Python strings/constants
- Usage: Loaded into agent system_prompt field during initialization

**src/multi-agent/app/utils/:**
- Purpose: Shared utility functions and helpers
- Contains: LLM setup validators, logging utilities, vLLM configuration, sandbox client access

**src/multi-agent/config/:**
- Purpose: Application configuration
- Files:
  - `config.toml`: TOML configuration for all LLM models, browser, search, sandbox, MCP, flow settings
  - `mcp.json`: MCP server endpoint configurations

## Key File Locations

**Entry Points:**
- `src/multi-agent/main.py`: Application CLI entry point - accepts --prompt and --image arguments
- `src/multi-agent/app/__init__.py`: Application package initialization

**Configuration:**
- `src/multi-agent/app/config.py`: Pydantic models defining configuration schema (AppConfig, LLMSettings, etc.)
- `src/multi-agent/config/config.toml`: Runtime configuration with 15+ LLM model profiles and tool settings
- `src/multi-agent/config/mcp.json`: MCP server configurations (if present)

**Core Logic:**
- `src/multi-agent/app/agent/base.py`: BaseAgent with state machine and memory management
- `src/multi-agent/app/flow/iterative_refinement.py`: Primary workflow orchestration
- `src/multi-agent/app/llm.py`: LLM abstraction supporting 7+ providers
- `src/multi-agent/app/tool/tool_collection.py`: Tool registry and execution

**Testing:**
- No dedicated test directory; test files co-located as `.../test/` subdirectories (e.g., `tool/chart_visualization/test/`)

## Naming Conventions

**Files:**
- `snake_case.py` for all Python modules
- Agent implementations: `[name].py` (e.g., `translator.py`, `react.py`)
- Tool implementations: `[action_name].py` (e.g., `smart_grid_caption.py`, `terminate_and_output_caption.py`)
- Prompt modules: `[agent_name].py` containing prompt constants
- Config files: `config.toml`, `mcp.json`

**Directories:**
- Lowercase with underscores: `agent`, `flow`, `tool`, `prompt`, `config`, `utils`, `sandbox`, `mcp`
- Agent-related: grouped under `agent/`
- Workflow related: grouped under `flow/`
- Tools grouped under `tool/` with optional subdirectories for complex tools (e.g., `chart_visualization/`, `search/`)

**Python Naming:**
- Classes: PascalCase (BaseAgent, TranslatorAgent, IterativeRefinementFlow)
- Functions/methods: snake_case
- Constants: UPPERCASE (ROLE_VALUES, TOOL_CHOICE_VALUES, SYSTEM_PROMPT)
- Enums: PascalCase class name (Role, AgentState, ToolChoice)

## Where to Add New Code

**New Agent Implementation:**
- Primary code: `src/multi-agent/app/agent/[agent_name].py`
- Extends: BaseAgent or specialized subclass (ReActAgent, ToolCallAgent)
- Prompts: `src/multi-agent/app/prompt/[agent_name].py` containing SYSTEM_PROMPT, NEXT_STEP_PROMPT, FINAL_STEP_PROMPT
- Registration: Optional - agents instantiated directly, not from registry
- Example pattern:
  ```python
  from app.agent.base import BaseAgent
  from app.llm import LLM
  from app.prompt.my_agent import SYSTEM_PROMPT, NEXT_STEP_PROMPT

  class MyAgent(BaseAgent):
      name: str = "my_agent"
      system_prompt: str = SYSTEM_PROMPT
      # ... implementation
  ```

**New Tool Implementation:**
- Primary code: `src/multi-agent/app/tool/[tool_name].py`
- Extends: BaseTool from `src/multi-agent/app/tool/base.py`
- Registration: Add to `ToolCollection([Tool1(), Tool2(), ...])` in agent configuration
- Lazy import: Add to `_LAZY_IMPORTS` dict in `src/multi-agent/app/tool/__init__.py`
- Example pattern:
  ```python
  from app.tool.base import BaseTool

  class MyTool(BaseTool):
      name: str = "my_tool"
      description: str = "..."

      async def execute(self, **kwargs) -> str:
          # Implementation
          pass
  ```

**New Flow Implementation:**
- Primary code: `src/multi-agent/app/flow/[flow_name].py`
- Extends: BaseFlow from `src/multi-agent/app/flow/base.py`
- Requires: Multiple agents passed as dict (e.g., {"translator": agent1, "reasoning": agent2})
- Factory: Register in `src/multi-agent/app/flow/flow_factory.py` if needed for dynamic instantiation
- Example pattern:
  ```python
  from app.flow.base import BaseFlow

  class MyFlow(BaseFlow):
      async def execute(self, input_text: str) -> str:
          # Orchestrate agents
          pass
  ```

**New Prompt Template:**
- Location: `src/multi-agent/app/prompt/[agent_name].py`
- Format: String constants named SYSTEM_PROMPT, NEXT_STEP_PROMPT, FINAL_STEP_PROMPT, FIRST_STEP_PROMPT
- Usage: Imported directly in agent class definition
- Pattern:
  ```python
  SYSTEM_PROMPT = """You are a specialized agent that...
  Your role is to...
  """

  NEXT_STEP_PROMPT = "What should you do next?"

  FINAL_STEP_PROMPT = "Provide your final answer..."
  ```

**Utilities:**
- Shared helpers: `src/multi-agent/app/utils/[function_name].py`
- Cross-cutting validators: `src/multi-agent/app/utils/agent_utils.py`

## Special Directories

**src/multi-agent/workspace/:**
- Purpose: Runtime workspace for agent operations
- Generated: Yes - created at runtime for file operations and intermediate results
- Committed: No - workspace directory is gitignored

**src/multi-agent/app/agent/deprecated/:**
- Purpose: Archive of deprecated agent implementations
- Generated: No - manually maintained historical code
- Committed: Yes - for reference and potential recovery

**assets/:**
- Purpose: Static assets (images, documentation, examples)
- Generated: No - manually managed
- Committed: Yes - version controlled

**logs/:**
- Purpose: Runtime execution logs
- Generated: Yes - created at runtime by logger and flow execution
- Committed: No - logs are gitignored

**src/plan/:**
- Purpose: Task planning and TODO tracking
- Generated: Yes - GSD orchestrator creates task files here
- Committed: No - temporary planning files

---

*Structure analysis: 2026-04-07*
