# Technology Stack

**Analysis Date:** 2026-04-07

## Languages

**Primary:**
- Python 3.12 - Core language for multi-agent system, vision-language models, and LLM orchestration

**Secondary:**
- YAML (PyYAML 6.0.2) - Configuration files and data serialization
- TOML (tomli 2.0.0+) - Application configuration (`config.toml`)

## Runtime

**Environment:**
- Python 3.12 (required per `src/multi-agent/setup.py`)
- CUDA-compatible GPU (recommended 16GB+ RAM for local model inference)

**Package Manager:**
- pip - Python package management
- Lockfile: Not detected (requirements.txt used for pinned versions)

## Frameworks

**Core:**
- FastAPI 0.115.11 - Web server framework for MCP server and API endpoints
- uvicorn 0.34.0 - ASGI server for FastAPI applications
- Pydantic 2.10.6 - Data validation and settings management
- pydantic-core 2.27.2 - Core validation engine

**LLM Orchestration:**
- openai 1.66.3 - OpenAI client for API integration (base provider used by multiple LLM adapters)
- tenacity 9.0.0 - Retry logic and rate limit handling for LLM API calls
- tiktoken 0.9.0 - Token counting for OpenAI models

**Model Serving:**
- vLLM (lazy loaded, optional) - Local model serving engine for vision-language models
- mcp 1.5.0 - Model Context Protocol for distributed agent capabilities

**Testing:**
- pytest 8.3.5 - Testing framework
- pytest-asyncio 0.25.3 - Async test support

**Build/Dev:**
- setuptools 75.8.0 - Package distribution and setup
- docker 7.1.0 - Docker SDK for containerized operations

## Key Dependencies

**Critical:**
- openai 1.66.3 - Primary LLM provider interface; all providers adapt to OpenAI API format
- Pydantic 2.10.6 - Configuration system, message types, and agent schemas throughout codebase
- browser-use 0.1.40 - Web automation and browser control for agent tasks
- Playwright 1.51.0 - Browser automation library underlying browser-use

**Vision-Language Models:**
- transformers - Vision model loading (Qwen models)
- torch - Deep learning framework for local vision models
- qwen_vl_utils - Qwen model vision processing utilities
- PIL (Pillow) 11.1.0 - Image manipulation and preprocessing

**Infrastructure:**
- FastAPI 0.115.11 - HTTP server framework
- uvicorn 0.34.0 - ASGI application server
- aiofiles 24.1.0 - Async file I/O operations
- httpx 0.27.0+ - HTTP client for async requests
- boto3 1.37.18 - AWS services (Bedrock integration)

**Utilities:**
- loguru 0.7.3 - Advanced logging system
- numpy - Numerical computing (datasets, image processing)
- datasets 3.4.1 - Dataset loading and processing (HuggingFace)
- html2text 2024.2.26 - HTML to text conversion
- beautifulsoup4 4.13.3 - HTML/XML parsing
- requests 2.32.3 - HTTP requests (sync operations)
- colorama 0.4.6 - Terminal color support
- unidiff 0.7.5 - Unified diff parsing
- huggingface-hub 0.29.2 - HuggingFace model hub interface

**Specialized:**
- gymnasium 1.1.1 - RL environment framework (evaluation/benchmarking)
- browsergym 0.13.3 - Browser-based RL environment for web agents
- img2table - Image to table extraction
- crawl4ai - Web crawling and content extraction (optional, lazy loaded)

## Configuration

**Environment:**
- Configuration via `src/multi-agent/config/config.toml` (TOML format)
- Environment variable overrides supported for API keys:
  - `OPENAI_API_KEY` - OpenAI API key
  - `DASHSCOPE_API_KEY` - Alibaba DashScope API key
  - `TOGETHER_API_KEY` - Together AI API key
  - `WORKSPACE_ROOT` - Custom workspace directory

**Build:**
- `setup.py` at `src/multi-agent/setup.py` for package distribution
- `requirements.txt` at project root for dependency pinning
- Package name: `openmanus`
- Version: 0.1.0

## Platform Requirements

**Development:**
- Python 3.12+
- CUDA-compatible GPU (optional but recommended for local inference)
- 16GB+ RAM (recommended for vLLM model serving)
- pip for package installation

**Production:**
- Python 3.12+ runtime
- GPU with CUDA support (if using local vLLM)
- Docker support optional (docker SDK integrated)
- AWS credentials for Bedrock integration (if using AWS models)

## vLLM Configuration

Local vLLM servers support:
- Tensor parallelism (configurable tensor_parallel_size)
- GPU memory utilization settings (default 0.8, up to 0.95)
- GLIBC compatibility options:
  - disable_custom_all_reduce
  - enforce_eager mode
  - configurable max_model_len

## Supported LLM Provider APIs

1. **OpenAI** - GPT-4o, GPT-4o-mini models via `https://api.openai.com/v1`
2. **Azure OpenAI** - Azure-hosted OpenAI models (api_type: "azure")
3. **DashScope** (Alibaba) - Qwen models via `https://dashscope.aliyuncs.com/compatible-mode/v1`
4. **Together AI** - Open-source models via `https://api.together.xyz/v1`
5. **AWS Bedrock** - Amazon-hosted models (requires boto3, api_type: "aws")
6. **Local vLLM** - Self-hosted model server (api_type: "vllm")
7. **Ollama** - Local Ollama server (api_type: "ollama", commented examples in config)

---

*Stack analysis: 2026-04-07*
