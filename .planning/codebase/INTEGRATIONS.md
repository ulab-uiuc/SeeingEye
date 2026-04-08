# External Integrations

**Analysis Date:** 2026-04-07

## APIs & External Services

**Language Models (LLM Providers):**
- OpenAI - Primary LLM provider for reasoning and instruction following
  - SDK/Client: `openai` (1.66.3)
  - Models: gpt-4o, gpt-4o-mini
  - Auth: Environment variable `OPENAI_API_KEY`
  - Integration: `src/model.py` → OpenAIProvider class
  - Usage: Default reasoning agent model

- Azure OpenAI - Enterprise OpenAI deployment
  - SDK/Client: `openai` (AsyncAzureOpenAI client)
  - Auth: Environment variable `AZURE_API_KEY`, endpoint URL
  - Configuration: `api_type: "azure"` in config.toml
  - Integration: `src/model.py` → create_azure_openai() factory method

- Alibaba DashScope (Qwen Models) - Vision-language and text models
  - SDK/Client: Custom HTTP implementation via `httpx` and `requests`
  - Models: qwen2.5-vl-3b, qwen2.5-vl-7b, qwen2.5-vl-32b, qwen3-8b
  - Auth: Environment variable `DASHSCOPE_API_KEY`
  - Endpoint: `https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions`
  - Configuration: `api_type: "dashscope"` in config.toml
  - Integration: `src/model.py` → DashScopeProvider class
  - Usage: Translator agent (vision-language) and reasoning agent (text-only)

- Together AI - Cloud-hosted open-source models
  - SDK/Client: `openai` (compatible API, reused client)
  - Models: Qwen/Qwen3-VL-8B-Instruct, Qwen/Qwen2.5-7B-Instruct-Turbo
  - Auth: Environment variable `TOGETHER_API_KEY`
  - Endpoint: `https://api.together.xyz/v1`
  - Configuration: `api_type: "together"` in config.toml
  - Integration: `src/multi-agent/app/llm.py` → LLM class

- AWS Bedrock - Amazon-hosted foundation models
  - SDK/Client: `boto3` (1.37.18)
  - Models: Anthropic Claude models (claude-3-7-sonnet-20250219)
  - Auth: AWS credentials (environment or AWS config)
  - Configuration: `api_type: "aws"` in config.toml
  - Integration: `src/multi-agent/app/bedrock.py` → BedrockClient class
  - Usage: Alternative reasoning model option

- vLLM (Local) - Self-hosted model server
  - SDK/Client: `vllm` (lazy loaded)
  - Models: Qwen2.5-VL-3B, Qwen2.5-VL-7B, Qwen3-8B (local inference)
  - Configuration: `api_type: "vllm"` in config.toml
  - Integration: `src/model.py` → VLLMProvider class
  - Endpoints: Configurable (default: http://localhost:8000/v1, http://localhost:8001/v1)

**Web & Search:**
- Google Search - Web search engine
  - SDK/Client: Custom via BeautifulSoup + requests
  - Integration: `src/multi-agent/app/tool/search/google.py` → GoogleSearchEngine
  - Used by: WebSearch tool, Browser agent for web navigation

- DuckDuckGo Search - Privacy-focused web search
  - SDK/Client: Custom via requests
  - Integration: `src/multi-agent/app/tool/search/duckduckgo.py` → DuckDuckGoSearchEngine
  - Fallback search engine

- Bing Search - Microsoft web search
  - SDK/Client: Custom via requests
  - Integration: `src/multi-agent/app/tool/search/bing.py` → BingSearchEngine
  - Fallback search engine

- Baidu Search - Chinese search engine
  - SDK/Client: Custom via requests
  - Integration: `src/multi-agent/app/tool/search/baidu.py` → BaiduSearchEngine
  - Fallback search engine for Chinese queries

## Data Storage

**Databases:**
- Not detected - Project does not currently integrate with persistent databases
- Workspace filesystem used for temporary storage (`WORKSPACE_ROOT` directory)

**File Storage:**
- Local filesystem only
  - Workspace directory: Configurable via `WORKSPACE_ROOT` environment variable
  - Default location: `src/multi-agent/config/../../../workspace`
  - Used for: Temporary files, agent outputs, intermediate results

**Caching:**
- vLLM KV cache - For efficient model inference (GPU memory managed by vLLM)
- No external cache service (Redis, Memcached) detected

## Authentication & Identity

**Auth Providers:**
- Custom API key management
  - OpenAI: `OPENAI_API_KEY` environment variable
  - DashScope: `DASHSCOPE_API_KEY` environment variable
  - Together AI: `TOGETHER_API_KEY` environment variable
  - AWS Bedrock: Standard AWS credential chain (IAM roles, ~/.aws/credentials)
  - Azure OpenAI: `AZURE_API_KEY` environment variable

**Implementation:**
- Environment variable injection in `src/multi-agent/app/config.py` via Pydantic settings
- Config file support via `config.toml` with fallback to environment variables
- No OAuth or SSO integration detected

## Monitoring & Observability

**Error Tracking:**
- Not detected - No external error tracking service integrated

**Logs:**
- Local file logging via `loguru` (0.7.3)
  - Logger configured in `src/multi-agent/app/logger.py`
  - Logged to console and log files in `logs/` directory
  - Fallback: Standard Python logging when loguru unavailable

**Token Usage Tracking:**
- TokenCounter class in `src/multi-agent/app/token_counter.py`
- Tracks input/output tokens per request
- Used for cost estimation and quota management

## CI/CD & Deployment

**Hosting:**
- Not detected - Project is research/development focused
- Self-hosted deployment via Python scripts
- Optional Docker support via `docker` SDK (7.1.0)

**CI Pipeline:**
- Not detected - No GitHub Actions, Jenkins, or other CI service configured

## Environment Configuration

**Required env vars:**
- `OPENAI_API_KEY` - If using OpenAI models (gpt-4o, gpt-4o-mini)
- `DASHSCOPE_API_KEY` - If using Alibaba DashScope Qwen models
- `TOGETHER_API_KEY` - If using Together AI models
- `WORKSPACE_ROOT` - Custom workspace location (optional, defaults to `./workspace`)

**Secrets location:**
- Environment variables (preferred for security)
- Config file values (`config.toml` - contains empty placeholders, populated via env vars)
- AWS credentials via standard AWS credential chain for Bedrock

## Web Automation

**Browser Automation:**
- browser-use 0.1.40 - Higher-level browser control framework
  - Underlying: Playwright 1.51.0 for browser automation
  - Integration: `src/multi-agent/app/tool/browser_use_tool.py` → BrowserUseTool
  - Capabilities: Navigation, form filling, element clicking, content extraction
  - Configuration: `BrowserSettings` in config.toml (headless mode, security, proxy)

- Playwright 1.51.0 - Chromium/Firefox/WebKit browser control
  - Used by: browser-use for actual browser operations
  - Supports: Desktop browsers, mobile emulation, network interception

**Web Crawling:**
- crawl4ai (optional, lazy loaded) - Advanced web crawling and content extraction
  - Integration: `src/multi-agent/app/tool/crawl4ai.py` → Crawl4AI tool
  - Features: JavaScript rendering, content extraction, markdown conversion

## Vision & Image Processing

**OCR (Optical Character Recognition):**
- Azure Computer Vision API - Text extraction from images
  - Integration: `src/multi-agent/app/tool/ocr.py` → OCR tool
  - Auth: Azure subscription key and endpoint (hardcoded placeholders, must be configured)
  - Supports: Multiple languages, orientation detection
  - Endpoint pattern: `https://{resource-name}.cognitiveservices.azure.com/vision/v2.1/ocr`

**Image Processing:**
- PIL/Pillow 11.1.0 - Image loading, resizing, format conversion
  - Used by: Vision models, image preprocessing
  - Integration: Qwen VL utils, browser screenshot handling

**Vision Models (Local):**
- Qwen2.5-VL-3B/7B/32B - Vision-language models via vLLM or DashScope
  - Used by: Translator agent for visual understanding
  - Inference: Local vLLM or cloud API (DashScope)

- Qwen3-VL-8B - Newer vision-language model
  - Available via: Together AI API, local vLLM

## Model Context Protocol (MCP)

**MCP Server:**
- FastMCP server implementation
  - Location: `src/multi-agent/app/mcp/server.py`
  - Purpose: Expose tools to external Claude/LLM instances via MCP protocol
  - Tools exposed: Bash, StrReplaceEditor, Terminate
  - Configuration: `mcp.json` for server definitions

**MCP Integration:**
- mcp 1.5.0 - Model Context Protocol library
- Enables: Remote tool access, distributed agent orchestration
- Configuration: Optional `mcp.json` file with server definitions

## Specialized Tools

**Data Processing:**
- img2table - Extract tables from images (OCR-based)
  - Used by: Vision processing pipeline

- html2text 2024.2.26 - Convert HTML to markdown
  - Used by: Web content extraction, content normalization

- beautifulsoup4 4.13.3 - HTML/XML parsing
  - Used by: Web search result parsing, content extraction

- unidiff 0.7.5 - Unified diff parsing
  - Used by: Code change analysis, file editing validation

**Dataset Handling:**
- datasets 3.4.1 (HuggingFace) - Benchmark dataset loading
  - Used by: MMMU, MMMU-Pro, GQA evaluation benchmarks
  - Integration: Direct HuggingFace hub access via huggingface-hub

- gymnasium 1.1.1 - RL environment framework
  - Used by: Benchmark evaluation infrastructure
  - Integration: browsergym for web-based RL environments

---

*Integration audit: 2026-04-07*
