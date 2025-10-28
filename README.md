# SeeingEye: Agentic Information Flow Unlocks Multimodal Reasoning in Text-Only LLMs

This repository contains the official implementation for the paper **"SeeingEye: Agentic Information Flow Unlocks Multimodal Reasoning in Text-Only LLMs"**. The project demonstrates how text-only language models can achieve multimodal reasoning capabilities through sophisticated agentic information flow, using a multi-agent framework with specialized agents for vision and reasoning tasks.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Multi-Agent System](#multi-agent-system)
- [Agent Architecture](#agent-architecture)
- [Creating Custom Agents](#creating-custom-agents)
- [Supported Benchmarks](#supported-benchmarks)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

## 🎯 Overview

**SeeingEye** presents a novel approach to multimodal reasoning that enables text-only LLMs to process and reason about visual information through agentic information flow. The framework implements:

- **Multi-Agent Architecture**: Flexible agent-based system with specialized agents (Translator for vision, Reasoner for text-only reasoning)
- **Agentic Information Flow**: Sophisticated communication patterns between vision and reasoning agents
- **Tool-Augmented Reasoning**: ReAct pattern with tool calling for step-by-step reasoning
- **Benchmark Evaluation**: Extensive evaluation suite for popular VQA benchmarks (MMMU, MMMU-Pro, GQA, OCRBench, MIA)
- **Modular Design**: Easy to extend with custom agents and tools

### Key Innovation

Unlike traditional multimodal models that process vision and text jointly, SeeingEye uses:
1. **Translator Agent** (Vision-Language Model): Interprets visual content
2. **Reasoner Agent** (Text-Only LLM): Performs complex reasoning using visual descriptions
3. **Agentic Flow**: Structured information exchange enabling text-only models to "see"

## ✨ Features

### Core Components

- **Multi-Agent Framework** ([src/multi-agent/](src/multi-agent/))
  - Modular agent architecture with base classes and specialized implementations
  - ReAct pattern (Reasoning + Acting) for systematic problem-solving
  - Support for tool-calling agents with various capabilities
  - Planning and execution flow management
  - MCP (Model Context Protocol) integration for distributed agents

- **Vision-Language Models Support**
  - Native support for Qwen2.5-VL and Qwen3 models
  - OpenAI API compatibility
  - vLLM serving for efficient inference
  - Flexible model provider abstraction

- **Tool Ecosystem**
  - Python code execution with sandboxing
  - File operations and editing (StrReplaceEditor)
  - Web search and crawling (Crawl4AI)
  - Browser automation
  - Data visualization and charting
  - OCR capabilities
  - Planning and task decomposition

### Supported Agents

1. **VQA Agent** ([app/agent/vqa.py](src/multi-agent/app/agent/vqa.py))
   - Visual Question Answering with direct answer selection
   - Multiple choice question handling
   - Python execution for calculations
   - Optimized for benchmark evaluation

2. **Manus Agent** ([app/agent/manus.py](src/multi-agent/app/agent/manus.py))
   - General-purpose versatile agent
   - MCP server integration for external tools
   - Browser automation support
   - Extensive tool collection
   - Async initialization pattern

3. **Planning Flow** ([app/flow/planning.py](src/multi-agent/app/flow/planning.py))
   - Multi-step task planning and execution
   - Dynamic agent selection based on task type
   - Progress tracking and status management (NOT_STARTED, IN_PROGRESS, COMPLETED, BLOCKED)
   - Step-by-step execution with configurable executors

## 🚀 Installation

### Prerequisites

- Python 3.8+ (Python 3.12 recommended)
- CUDA-compatible GPU (for local model inference)
- 16GB+ RAM recommended
- vLLM for model serving

### Setup

1. Create a conda environment:
```bash
conda create -n seeingeye python=3.12
conda activate seeingeye
```

2. Clone the repository:
```bash
git clone https://github.com/CharlieDreemur/MPU-RL.git
cd MPU-RL
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package:
```bash
pip install -e .
```

5. (Optional) Install VERL for reinforcement learning:
```bash
# VERL is included as a git submodule
pip install -e ./verl
```

## 🤖 Multi-Agent System

The multi-agent system is built on a flexible architecture that supports the SeeingEye agentic information flow pattern.

### Agent Execution Flow

The framework uses a hierarchical execution model:

```
BaseAgent.run()           # Main execution loop (max_steps iterations)
└── ReActAgent.step()     # Single reasoning cycle
    ├── think()           # Decision making (LLM queries with tools)
    └── act()             # Tool execution
        └── execute_tool()  # Individual tool calls
```

**Key Execution Steps:**

1. **`run(request)`**: Main entry point that manages the agent lifecycle
   - Validates agent state (must be IDLE)
   - Adds user request to memory
   - Executes step-by-step loop until completion or max_steps
   - Handles stuck state detection

2. **`step()`**: Single reasoning cycle implementing the ReAct pattern
   - Calls `think()` to decide next action
   - Calls `act()` to execute selected tools
   - Returns formatted step result

3. **`think()`**: Decision making with LLM
   - Sends conversation history + available tools to LLM
   - LLM selects appropriate tools based on context
   - Updates memory with assistant response
   - Returns True if tools were selected, False otherwise

4. **`act()`**: Tool execution
   - Executes each selected tool with arguments
   - Adds tool results to memory
   - Handles special tools (e.g., terminate)
   - Aggregates and returns all tool outputs

### Agent Architecture Example

```python
from app.agent.vqa import VQAAgent
from app.agent.manus import Manus

# Create a VQA agent (synchronous initialization)
vqa_agent = VQAAgent()

# Create a general-purpose agent with MCP support (async initialization)
manus_agent = await Manus.create()

# Run the agent with a user request
result = await vqa_agent.run("What color is the car in this image?")
```

### Flow Management

```python
from app.flow.planning import PlanningFlow

# Create a planning flow with multiple agents
flow = PlanningFlow(
    agents={
        "planner": planner_agent,
        "executor": manus_agent,
        "vqa": vqa_agent
    },
    executors=["executor", "vqa"]  # Agents available for task execution
)

# Execute a complex task with automatic planning
result = await flow.execute("Analyze this chart and create a summary report")
```

### Key Components

- **BaseAgent** ([app/agent/base.py](src/multi-agent/app/agent/base.py)): Abstract base class managing state, memory, and execution loop
- **ReActAgent** ([app/agent/react.py](src/multi-agent/app/agent/react.py)): Implements ReAct pattern (Reasoning + Acting)
- **ToolCallAgent** ([app/agent/toolcall.py](src/multi-agent/app/agent/toolcall.py)): Concrete agent with LLM tool calling
- **Flow System** ([app/flow/](src/multi-agent/app/flow/)): Orchestrates multi-agent workflows
- **MCP Integration** ([app/mcp/](src/multi-agent/app/mcp/)): Model Context Protocol for distributed systems

For detailed architecture information, see [AGENT_ARCHITECTURE.md](src/multi-agent/AGENT_ARCHITECTURE.md).

## 🛠️ Creating Custom Agents

The framework is designed for easy extensibility. Create custom agents by inheriting from `ToolCallAgent`:

### Quick Start

```python
from typing import List
from pydantic import Field
from app.agent.toolcall import ToolCallAgent
from app.prompt.my_agent import SYSTEM_PROMPT, NEXT_STEP_PROMPT
from app.tool import Bash, StrReplaceEditor, Terminate, ToolCollection

class MyCustomAgent(ToolCallAgent):
    """Custom agent for specific tasks"""

    name: str = "my_agent"
    description: str = "A specialized agent that does X, Y, Z"

    # Define agent behavior through prompts
    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    # Configure available tools
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            Bash(),
            StrReplaceEditor(),
            Terminate()
        )
    )

    # Tools that can terminate execution
    special_tool_names: List[str] = Field(default_factory=lambda: ["terminate"])

    # Execution limits
    max_steps: int = 25
```

### Agent Types

1. **Programming Agents** (SWE-style)
   - File system operations
   - Code execution and debugging
   - Tools: Bash, StrReplaceEditor, Terminate

2. **General-Purpose Agents** (Manus-style)
   - Multi-modal capabilities
   - Web browsing
   - MCP integration
   - Tools: PythonExecute, BrowserUseTool, StrReplaceEditor, MCPClientTool

3. **Specialized Agents** (VQA-style)
   - Domain-specific tasks
   - Optimized tool sets
   - Custom prompting strategies

For a complete guide, see [CUSTOM_AGENT_GUIDE.md](src/multi-agent/CUSTOM_AGENT_GUIDE.md).

## 📊 Supported Benchmarks

The framework includes evaluation pipelines for several multimodal benchmarks, demonstrating SeeingEye's effectiveness:

### 1. MMMU & MMMU-Pro
- **Location**: [benchmark_evaluation/mmmu/](src/multi-agent/benchmark_evaluation/mmmu/), [benchmark_evaluation/mmmu_pro/](src/multi-agent/benchmark_evaluation/mmmu_pro/)
- **Description**: Massive Multi-discipline Multimodal Understanding benchmark
- **Features**:
  - Flow-based evaluation with agentic reasoning
  - Multiple model backends (Qwen2-VL, GPT-4V)
  - Support for multi-turn reasoning

### 2. GQA (Visual Reasoning)
- **Location**: [benchmark_evaluation/GQA/](src/multi-agent/benchmark_evaluation/GQA/)
- **Description**: Scene graph-based visual question answering
- **Features**:
  - Official evaluation pipeline integration
  - Compositional reasoning support

### 3. OCRBench
- **Location**: [benchmark_evaluation/ocrbench/](src/multi-agent/benchmark_evaluation/ocrbench/)
- **Description**: Text recognition and understanding benchmark
- **Features**:
  - Multiple OCR metrics (IoU, TEDS, VQA)
  - Flow-based evaluation
  - Support for various OCR tasks

### 4. MIA (Multimodal Interaction)
- **Location**: [benchmark_evaluation/MIA/](src/multi-agent/benchmark_evaluation/MIA/)
- **Description**: Multi-turn multimodal interaction evaluation
- **Features**:
  - Flow-based inference
  - Conversational context handling

### Benchmark Integration

The framework provides a modular `FlowExecutor` for easy integration with other benchmarks. See [example_benchmark_integration.md](src/multi-agent/example_benchmark_integration.md) for examples of integrating with:
- Generic question-answering benchmarks
- VQA-style benchmarks
- Math reasoning benchmarks
- Custom evaluation frameworks

## 💻 Usage

### Running Multi-Agent System

```bash
cd src/multi-agent
python main.py --prompt "Your task here" --image "path/to/image.jpg"
```

### Benchmark Evaluation

#### MMMU Evaluation
```bash
cd src/multi-agent/benchmark_evaluation/mmmu
python run_mmmu_flow.py --model_path Qwen/Qwen2.5-VL-7B-Instruct
```

#### MMMU-Pro Evaluation
```bash
cd src/multi-agent/benchmark_evaluation/mmmu_pro
python run_mmmu_pro_flow.py
```

#### GQA Evaluation
```bash
cd src/multi-agent/benchmark_evaluation/GQA
python run_gqa_official_pipeline.py
```

#### OCRBench Evaluation
```bash
cd src/multi-agent/benchmark_evaluation/ocrbench
python infer_ocrbench_flow.py
```

#### MIA Evaluation
```bash
cd src/multi-agent/benchmark_evaluation/MIA
python run_mia_inference_flow.py
```

### Serving Models with vLLM

For the SeeingEye architecture, you typically need two models:
1. **Vision-Language Model** (Translator Agent) - e.g., Qwen2.5-VL-3B
2. **Text-Only Model** (Reasoner Agent) - e.g., Qwen3-8B

See [docs/vllm_serving_guide.md](docs/vllm_serving_guide.md) for detailed instructions.

**Quick Start:**

```bash
# Terminal 1: Serve vision-language model (Translator Agent)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.45 \
    --disable-custom-all-reduce \
    --enforce-eager \
    --dtype float16 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes

# Terminal 2: Serve text-only model (Reasoner Agent)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B \
    --port 8001 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.45 \
    --disable-custom-all-reduce \
    --enforce-eager \
    --dtype float16 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

**Important Notes:**
- Use `python -m vllm.entrypoints.openai.api_server` for multi-modal models
- `--max-model-len` is omitted to use model's native context length
- `--enable-auto-tool-choice` and `--tool-call-parser hermes` enable tool calling

## ⚙️ Configuration

Configuration is managed through TOML files and environment variables:

1. **Model Configuration**: Configure model providers in `src/multi-agent/config/config.toml`
2. **MCP Servers**: Set up Model Context Protocol servers for distributed agents
3. **Tool Settings**: Customize available tools and their parameters
4. **Agent Settings**: Configure max_steps, prompts, and tool collections

### Environment Variables

```bash
export OPENAI_API_KEY="your-api-key"
export WORKSPACE_ROOT="/path/to/workspace"
```

### Configuration Files

- `config/config.toml`: Main configuration
- `app/prompt/*.py`: Agent prompts
- Agent class definitions: Tool configurations

## 📁 Project Structure

```
MPU-RL/
├── src/
│   ├── model.py                    # Model providers and factory
│   ├── message_types.py            # Message data structures
│   ├── config.py                   # Configuration management
│   └── multi-agent/                # Multi-agent system (SeeingEye)
│       ├── app/
│       │   ├── agent/              # Agent implementations
│       │   │   ├── base.py         # Base agent class (state, memory, run loop)
│       │   │   ├── react.py        # ReAct pattern (think + act)
│       │   │   ├── toolcall.py     # Tool-calling agent (LLM integration)
│       │   │   ├── vqa.py          # VQA agent
│       │   │   ├── manus.py        # General-purpose agent
│       │   │   └── browser.py      # Browser automation
│       │   ├── flow/               # Flow management
│       │   │   ├── base.py         # Base flow
│       │   │   ├── planning.py     # Planning flow with step tracking
│       │   │   ├── flow_factory.py # Flow factory
│       │   │   └── flow_executor.py # Modular flow executor
│       │   ├── tool/               # Tool implementations
│       │   │   ├── base.py         # Base tool class
│       │   │   ├── bash.py         # Shell execution
│       │   │   ├── python_execute.py # Python code execution
│       │   │   ├── str_replace_editor.py # File editing
│       │   │   ├── file_operators.py # File operations
│       │   │   ├── web_search.py   # Web search (multiple engines)
│       │   │   ├── planning.py     # Planning tool
│       │   │   ├── terminate.py    # Task termination
│       │   │   ├── ocr.py          # OCR capabilities
│       │   │   ├── mcp_tool.py     # MCP integration
│       │   │   └── chart_visualization/ # Data visualization
│       │   ├── prompt/             # Prompt templates
│       │   │   ├── vqa.py          # VQA prompts
│       │   │   ├── manus.py        # Manus prompts
│       │   │   └── planning.py     # Planning prompts
│       │   ├── sandbox/            # Sandboxed execution
│       │   │   ├── client.py       # Sandbox client
│       │   │   └── core/           # Sandbox core
│       │   ├── mcp/                # MCP integration
│       │   │   └── server.py       # MCP server
│       │   ├── llm.py              # LLM interface
│       │   ├── schema.py           # Data schemas
│       │   └── logger.py           # Logging utilities
│       ├── benchmark_evaluation/   # Benchmark evaluation
│       │   ├── mmmu/               # MMMU benchmark
│       │   │   ├── run_mmmu_flow.py
│       │   │   └── dataset_utils.py
│       │   ├── mmmu_pro/           # MMMU-Pro benchmark
│       │   │   ├── run_mmmu_pro_flow.py
│       │   │   └── evaluate.py
│       │   ├── GQA/                # GQA benchmark
│       │   │   └── run_gqa_official_pipeline.py
│       │   ├── MIA/                # MIA benchmark
│       │   │   └── run_mia_inference_flow.py
│       │   ├── ocrbench/           # OCRBench
│       │   │   ├── infer_ocrbench_flow.py
│       │   │   └── OCRBench_v2_eval/
│       │   └── utils/              # Evaluation utilities
│       ├── protocol/               # Agent-to-agent protocol
│       │   └── a2a/                # A2A protocol implementation
│       ├── main.py                 # Main entry point
│       ├── AGENT_ARCHITECTURE.md   # Architecture documentation
│       ├── CUSTOM_AGENT_GUIDE.md   # Custom agent guide
│       └── example_benchmark_integration.md # Benchmark integration examples
├── docs/
│   └── vllm_serving_guide.md      # vLLM serving guide
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup
└── README.md                       # This file
```

## 🔧 Tools and Utilities

The framework provides a rich set of tools for agentic reasoning:

### Execution Tools
- **Python Execute**: Sandboxed Python code execution for calculations and data processing
- **Bash**: Shell command execution for system operations

### File Operations
- **StrReplaceEditor**: Precise file editing with string replacement
- **File Operators**: Read, write, and manage files

### Web Tools
- **Web Search**: Multi-engine search (Google, Bing, DuckDuckGo, Baidu)
- **Crawl4AI**: Advanced web crawling and content extraction
- **Browser Use Tool**: Browser automation for complex web interactions

### Specialized Tools
- **OCR**: Optical character recognition for text extraction
- **Chart Visualization**: Data visualization and chart creation
- **Planning**: Task planning and decomposition
- **Terminate**: Task completion and answer submission

### MCP Tools
- **MCP Client**: Connect to external MCP servers for additional capabilities

## 🎨 Advanced Features

### MCP (Model Context Protocol)

Run distributed agents with MCP servers for extended capabilities:

```bash
# Start MCP server
python src/multi-agent/run_mcp_server.py

# Connect agents via MCP
python src/multi-agent/run_mcp.py
```

### Agent-to-Agent Protocol

Enable agent communication using A2A protocol for collaborative reasoning:

```python
from protocol.a2a.app.agent import create_agent

agent = await create_agent("agent_name")
```

### Custom Tool Development

Create custom tools by inheriting from `BaseTool`:

```python
from app.tool.base import BaseTool

class MyCustomTool(BaseTool):
    name: str = "my_tool"
    description: str = "Does something specific"

    async def execute(self, **kwargs) -> str:
        # Your tool logic here
        return "Tool execution result"
```

## 📝 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{seeingeye2024,
  title={SeeingEye: Agentic Information Flow Unlocks Multimodal Reasoning in Text-Only LLMs},
  author={[Authors]},
  journal={[Journal/Conference]},
  year={2024}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- VERL framework for reinforcement learning integration
- Qwen team for the vision-language models
- vLLM for efficient model serving
- The open-source community for various tools and libraries
- All benchmark dataset creators and maintainers

## 📮 Contact

For questions and discussions, please open an issue on GitHub.

## 🔗 Related Resources

- [Agent Architecture Documentation](src/multi-agent/AGENT_ARCHITECTURE.md)
- [Custom Agent Development Guide](src/multi-agent/CUSTOM_AGENT_GUIDE.md)
- [Benchmark Integration Examples](src/multi-agent/example_benchmark_integration.md)
- [vLLM Serving Guide](docs/vllm_serving_guide.md)

---

**Note**: This is a research project under active development. APIs and features may change.
