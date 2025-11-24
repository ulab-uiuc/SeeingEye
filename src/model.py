"""Model Providers and Factory

This module contains ALL model provider implementations consolidated from:
- src/model.py (original)
- src/multi-agent/app/llm.py
- src/multi-agent/app/vllm_provider.py

Provides unified interfaces for different model providers with adapter patterns.
"""

import os
import sys
import base64
import json
import asyncio
from io import BytesIO
from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict, Union

# Standard model imports
from openai import (
    OpenAI, AsyncOpenAI, AsyncAzureOpenAI, APIError, AuthenticationError,
    RateLimitError, OpenAIError
)
from openai.types.chat import ChatCompletion, ChatCompletionMessage

# Defer heavy imports until needed
torch = None
process_vision_info = None
AutoProcessor = None
Qwen2_5_VLForConditionalGeneration = None
PIL_Image = None

def _import_qwen_dependencies():
    """Lazy import Qwen dependencies when needed."""
    global torch, process_vision_info, AutoProcessor, Qwen2_5_VLForConditionalGeneration, PIL_Image
    if torch is None:
        import torch as torch_module
        from qwen_vl_utils import process_vision_info as pvi
        from transformers import AutoProcessor as AP, Qwen2_5_VLForConditionalGeneration as Q2VL
        from PIL import Image
        torch = torch_module
        process_vision_info = pvi
        AutoProcessor = AP
        Qwen2_5_VLForConditionalGeneration = Q2VL
        PIL_Image = Image

# Conditional vLLM imports - only import if actually needed
def _check_vllm_needed():
    """Check if any configuration actually requires vLLM"""
    try:
        # Read the config file directly as text and look for vllm usage
        config_path = os.path.join(os.path.dirname(__file__), 'multi-agent/config/config.toml')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_content = f.read()

            # Simple text search for vllm usage - more reliable than parsing TOML
            # Look for lines containing 'api_type = "vllm"' or similar patterns
            lines = config_content.split('\n')
            for line in lines:
                line = line.strip()
                if 'api_type' in line and 'vllm' in line:
                    return True

        # No vLLM usage found in config
        return False
    except Exception:
        # If we can't check config, be conservative and don't load vLLM
        # This is safer than assuming it's needed
        return False

# Defer vLLM imports until actually needed (lazy loading)
vLLM_Engine = None
SamplingParams = None
MultiModalDataDict = None
VLLM_AVAILABLE = None  # Will be determined when first needed

def _import_vllm():
    """Lazy import vLLM when actually needed."""
    global vLLM_Engine, SamplingParams, MultiModalDataDict, VLLM_AVAILABLE
    if VLLM_AVAILABLE is None:
        try:
            from vllm import LLM as vLLM_Engine, SamplingParams
            from vllm.multimodal import MultiModalDataDict
            VLLM_AVAILABLE = True
        except ImportError:
            vLLM_Engine = None
            SamplingParams = None
            MultiModalDataDict = None
            VLLM_AVAILABLE = False
    return VLLM_AVAILABLE

# Import from modular files
from config import ModelConfig
from message_types import (
    MessageRole, ContentType, MessageContent, Message, Conversation
)

# Add support for multi-agent app imports if available
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'multi-agent/app'))
    from logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ============================================================================
# ABSTRACT BASE CLASSES
# ============================================================================

class BaseModelProvider(ABC):
    """Base model provider for MPU-RL conversation format"""
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()

    @abstractmethod
    def chat_completion(self, conversation: Conversation, **kwargs) -> str:
        """Generate completion from Conversation object"""
        pass

    @abstractmethod
    def chat_completion_raw(self, conversation: Conversation, **kwargs) -> Any:
        """Generate raw completion from Conversation object"""
        pass


class ModelProvider(ABC):
    """Model provider for multi-agent message list format"""

    @abstractmethod
    async def create_completion(
        self,
        messages: List[dict],
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Union[ChatCompletion, str]:
        """Create completion from message list"""
        pass

    @abstractmethod
    async def create_tool_completion(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        tool_choice: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ChatCompletionMessage:
        """Create tool-enabled completion"""
        pass


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def conversation_to_messages(conversation: Conversation) -> List[dict]:
    """Convert Conversation to OpenAI message format"""
    messages = []

    # Add system message if present
    if conversation.system_prompt:
        messages.append({
            "role": "system",
            "content": conversation.system_prompt
        })

    # Convert conversation messages
    for msg in conversation.messages:
        if isinstance(msg.content, str):
            # Simple text message
            messages.append({
                "role": msg.role.value,
                "content": msg.content
            })
        elif isinstance(msg.content, list):
            # Multimodal message
            content = []
            for content_item in msg.content:
                if content_item.type == ContentType.TEXT:
                    content.append({"type": "text", "text": content_item.text})
                elif content_item.type == ContentType.IMAGE_URL:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": content_item.image_url}
                    })
                elif content_item.type == ContentType.IMAGE_BASE64:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{content_item.image_base64}"}
                    })
            messages.append({
                "role": msg.role.value,
                "content": content
            })

    return messages


def messages_to_conversation(messages: List[dict]) -> Conversation:
    """Convert OpenAI message format to Conversation"""
    conversation = Conversation()

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role == "system":
            conversation.system_prompt = content if isinstance(content, str) else str(content)
        elif role == "user":
            if isinstance(content, str):
                conversation.add_user_message(content)
            elif isinstance(content, list):
                # Handle multimodal content
                mpu_content = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            mpu_content.append(
                                MessageContent(type=ContentType.TEXT, text=item["text"])
                            )
                        elif item.get("type") == "image_url":
                            image_url = item["image_url"]["url"]
                            if image_url.startswith("data:image"):
                                # Base64 image
                                base64_data = image_url.split(",")[1]
                                mpu_content.append(
                                    MessageContent(type=ContentType.IMAGE_BASE64, image_base64=base64_data)
                                )
                            else:
                                # URL image
                                mpu_content.append(
                                    MessageContent(type=ContentType.IMAGE_URL, image_url=image_url)
                                )
                conversation.add_user_message(mpu_content)
        elif role == "assistant":
            conversation.add_assistant_message(content if isinstance(content, str) else str(content))

    return conversation


# ============================================================================
# OPENAI PROVIDERS
# ============================================================================

class OpenAIProvider(BaseModelProvider, ModelProvider):
    """Unified OpenAI provider supporting both interface styles"""

    def __init__(self,
                 client_or_key: Union[OpenAI, AsyncOpenAI, AsyncAzureOpenAI, str, None] = None,
                 model: str = "gpt-4o-mini",
                 config: ModelConfig = None,
                 api_type: str = "openai",
                 base_url: Optional[str] = None,
                 api_version: Optional[str] = None):
        super().__init__(config)

        self.model = model
        self.api_type = api_type

        # Handle different client types
        if isinstance(client_or_key, (OpenAI, AsyncOpenAI, AsyncAzureOpenAI)):
            self.client = client_or_key
            self.sync_client = client_or_key if isinstance(client_or_key, OpenAI) else None
        else:
            # Create clients from key/config
            api_key = client_or_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key is required")

            if api_type == "azure":
                self.client = AsyncAzureOpenAI(
                    base_url=base_url,
                    api_key=api_key,
                    api_version=api_version,
                )
                self.sync_client = None  # Could create sync version if needed
            else:
                self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
                self.sync_client = OpenAI(api_key=api_key, base_url=base_url)

        # Reasoning models that have different parameter names
        self.reasoning_models = ["o1", "o3-mini"]

        # Supported OpenAI models (expanded list)
        self.model_variants = {
            "gpt-4": "gpt-4",
            "gpt-4-turbo": "gpt-4-turbo",
            "gpt-4o": "gpt-4o",
            "gpt-4o-mini": "gpt-4o-mini",
            "gpt-5": "gpt-5",
            "gpt-3.5-turbo": "gpt-3.5-turbo",
        }

        logger.info(f"Initialized OpenAI provider with model: {model}")

    # BaseModelProvider interface (sync, uses Conversation)
    def chat_completion(self, conversation: Conversation, **kwargs) -> str:
        response = self.chat_completion_raw(conversation, **kwargs)
        return response.choices[0].message.content

    def chat_completion_raw(self, conversation: Conversation, **kwargs) -> Any:
        if not self.sync_client:
            raise ValueError("Sync client not available. Use async methods or recreate provider with sync client.")

        # Convert conversation to messages
        messages = conversation_to_messages(conversation)

        # Merge config with kwargs, with kwargs taking precedence
        params = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "frequency_penalty": kwargs.get("frequency_penalty", self.config.frequency_penalty),
            "presence_penalty": kwargs.get("presence_penalty", self.config.presence_penalty),
            "stream": kwargs.get("stream", self.config.stream),
        }

        # Add vLLM-specific parameters if provided
        if "top_k" in kwargs:
            params["top_k"] = kwargs["top_k"]
        if "repetition_penalty" in kwargs:
            params["repetition_penalty"] = kwargs["repetition_penalty"]

        # Handle reasoning models
        if self.model in self.reasoning_models:
            max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
            if max_tokens:
                params["max_completion_tokens"] = max_tokens
                del params["max_tokens"]

        # Only add temperature if it's not None (let OpenAI use default)
        temperature = kwargs.get("temperature", self.config.temperature)
        if temperature is not None and self.model not in self.reasoning_models:
            params["temperature"] = temperature

        return self.sync_client.chat.completions.create(**params)

    # ModelProvider interface (async, uses message list)
    async def create_completion(
        self,
        messages: List[dict],
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Union[ChatCompletion, str]:
        params = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
        }

        # Extract vLLM-specific parameters for extra_body
        vllm_params = {}
        standard_params = {}

        for key, value in kwargs.items():
            if key in ['top_k', 'repetition_penalty', 'min_p', 'length_penalty',
                      'early_stopping', 'ignore_eos', 'min_tokens', 'stop_token_ids',
                      'skip_special_tokens', 'spaces_between_special_tokens', 'use_beam_search']:
                vllm_params[key] = value
            else:
                standard_params[key] = value

        # Add standard parameters directly
        params.update(standard_params)

        # Add vLLM-specific parameters to extra_body if any
        if vllm_params:
            params["extra_body"] = vllm_params

        if self.model in self.reasoning_models:
            if max_tokens:
                params["max_completion_tokens"] = max_tokens
        else:
            if max_tokens:
                params["max_tokens"] = max_tokens
            if temperature is not None:
                params["temperature"] = temperature

        return await self.client.chat.completions.create(**params)

    async def create_tool_completion(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        tool_choice: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ChatCompletionMessage:
        params = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "stream": False,  # Always non-streaming for tools
        }

        # Extract vLLM-specific parameters for extra_body
        vllm_params = {}
        standard_params = {}

        for key, value in kwargs.items():
            if key in ['top_k', 'repetition_penalty', 'min_p', 'length_penalty',
                      'early_stopping', 'ignore_eos', 'min_tokens', 'stop_token_ids',
                      'skip_special_tokens', 'spaces_between_special_tokens', 'use_beam_search']:
                vllm_params[key] = value
            else:
                standard_params[key] = value

        # Add standard parameters directly
        params.update(standard_params)

        # Add vLLM-specific parameters to extra_body if any
        if vllm_params:
            params["extra_body"] = vllm_params

        if self.model in self.reasoning_models:
            if max_tokens:
                params["max_completion_tokens"] = max_tokens
        else:
            if max_tokens:
                params["max_tokens"] = max_tokens
            if temperature is not None:
                params["temperature"] = temperature

        response: ChatCompletion = await self.client.chat.completions.create(**params)
        return response.choices[0].message if response.choices else None


# ============================================================================
# QWEN PROVIDERS
# ============================================================================

class Qwen25VLProvider(BaseModelProvider, ModelProvider):
    """Unified Qwen 2.5-VL provider supporting both interface styles"""

    def __init__(self,
                 model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 device_map: str = "auto",
                 torch_dtype: str = "auto",
                 flash_attention: bool = False,
                 config: ModelConfig = None):
        super().__init__(config)

        self.model_name = model_name
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.flash_attention = flash_attention

        self.model = None
        self.processor = None

        # Supported model variants
        self.model_variants = {
            "3b": "Qwen/Qwen2.5-VL-3B-Instruct",
            "7b": "Qwen/Qwen2.5-VL-7B-Instruct",
        }

        # Set model name based on variant if provided
        if model_name.lower() in ["3b", "7b"]:
            self.model_name = self.model_variants[model_name.lower()]

    def load_model(self):
        """Load the model and processor"""
        if self.model is not None:
            return

        # Load model with appropriate configuration
        if self.flash_attention:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                attn_implementation='flash_attention_2',
                device_map=self.device_map
            )
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                device_map=self.device_map,
                torch_dtype=self.torch_dtype
            )

        # Lazy import dependencies when needed
        _import_qwen_dependencies()
        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def _transform_conversation_to_messages(self, conversation: Conversation) -> List[Dict[str, Any]]:
        """Transform conversation to Qwen VL message format"""
        messages = []

        for msg in conversation.messages:
            content = []

            if isinstance(msg.content, str):
                # Simple text message
                content.append({"type": "text", "text": msg.content})
            elif isinstance(msg.content, list):
                # Multi-modal message
                for content_item in msg.content:
                    if content_item.type == ContentType.TEXT:
                        content.append({"type": "text", "text": content_item.text})
                    elif content_item.type == ContentType.IMAGE_URL:
                        content.append({"type": "image", "image": content_item.image_url})
                    elif content_item.type == ContentType.IMAGE_BASE64:
                        # Convert base64 to data URL format
                        data_url = f"data:image/jpeg;base64,{content_item.image_base64}"
                        content.append({"type": "image", "image": data_url})

            messages.append({
                "role": msg.role.value,
                "content": content
            })

        return messages

    def chat_completion(self, conversation: Conversation, **kwargs) -> str:
        """Generate chat completion"""
        if self.model is None:
            self.load_model()

        # Transform conversation to Qwen VL format
        messages = self._transform_conversation_to_messages(conversation)

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process vision information (images/videos)
        _import_qwen_dependencies()  # Ensure dependencies are loaded
        image_inputs, video_inputs = process_vision_info(messages)

        # Prepare inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors='pt'
        )
        inputs = inputs.to(self.model.device)

        # Generation parameters
        generation_params = {
            "max_new_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            **inputs
        }

        # Handle temperature - only add if not None (let model use default)
        temperature = kwargs.get("temperature", self.config.temperature)
        if temperature is not None:
            generation_params["temperature"] = temperature
            generation_params["do_sample"] = temperature > 0
        else:
            # If no temperature specified, use default sampling behavior
            generation_params["do_sample"] = True

        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(**generation_params)

        # Extract only the new tokens (response)
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]

        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        return response.strip()

    def chat_completion_raw(self, conversation: Conversation, **kwargs) -> Any:
        """Return raw generation result (for compatibility with base class)"""
        response = self.chat_completion(conversation, **kwargs)

        # Return a mock response object similar to OpenAI format
        class MockResponse:
            def __init__(self, content):
                self.choices = [MockChoice(content)]

        class MockChoice:
            def __init__(self, content):
                self.message = MockMessage(content)

        class MockMessage:
            def __init__(self, content):
                self.content = content

        return MockResponse(response)

    # ModelProvider interface (async, uses message list)
    async def create_completion(
        self,
        messages: List[dict],
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        try:
            # Convert messages to conversation format
            conversation = messages_to_conversation(messages)

            # Use sync chat_completion method
            response = self.chat_completion(
                conversation,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            return response

        except Exception as e:
            logger.error(f"Error in Qwen VLM request: {e}")
            raise

    async def create_tool_completion(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        tool_choice: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ChatCompletionMessage:
        logger.warning("Tool calling with Qwen VLM has limited support. Falling back to regular completion.")

        # For now, just do regular completion and wrap in ChatCompletionMessage format
        response_text = await self.create_completion(messages, False, temperature, max_tokens, **kwargs)

        # Create a mock ChatCompletionMessage
        class MockMessage:
            def __init__(self, content):
                self.content = content
                self.tool_calls = None

        return MockMessage(response_text)


# ============================================================================
# VLLM PROVIDER
# ============================================================================

class VLLMProvider(ModelProvider):
    """vLLM provider for efficient local inference"""

    def __init__(self, model_name: str, tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.95, **kwargs):
        # Lazy import vLLM when actually creating the provider
        if not _import_vllm():
            raise ImportError("vLLM is required for VLLMProvider. Install with: pip install vllm")

        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization

        # Initialize vLLM engine
        self.engine = vLLM_Engine(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,  # Needed for Qwen models
            **kwargs
        )

        logger.info(f"Initialized vLLM engine with model: {model_name}")

    def _create_sampling_params(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: float = 0.9,
        **kwargs
    ) -> SamplingParams:
        """Create sampling parameters for generation"""
        return SamplingParams(
            temperature=temperature or 0.7,
            max_tokens=max_tokens or 1024,
            top_p=top_p,
            **kwargs
        )

    def _format_messages_to_prompt(self, messages: List[dict]) -> str:
        """Convert message format to prompt string"""
        # Check if this is a Qwen3 model which uses newer chat template format
        if "Qwen3" in self.model_name:
            # For Qwen3 models, use the tokenizer's apply_chat_template if available
            try:
                tokenizer = self.engine.get_tokenizer()
                if hasattr(tokenizer, 'apply_chat_template'):
                    # Use the model's native chat template
                    prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    return prompt
            except Exception as e:
                logger.warning(f"Failed to use native chat template for Qwen3: {e}, falling back to manual format")

        # Fallback for Qwen2.5 and other models using im_start/im_end format
        prompt_parts = []

        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "system":
                prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                if isinstance(content, str):
                    prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
                elif isinstance(content, list):
                    # Handle multimodal content - extract text parts
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(item["text"])
                    combined_text = " ".join(text_parts)
                    prompt_parts.append(f"<|im_start|>user\n{combined_text}<|im_end|>")
            elif role == "assistant":
                prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")

        # Add assistant start for generation
        prompt_parts.append("<|im_start|>assistant\n")

        return "\n".join(prompt_parts)

    def _extract_multimodal_data(self, messages: List[dict]) -> Dict:
        """Extract images from messages for multimodal processing"""
        multimodal_data = {}
        images = []

        for message in messages:
            content = message.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        image_url = item["image_url"]["url"]
                        if image_url.startswith("data:image"):
                            # Handle base64 images
                            try:
                                # Extract base64 data
                                base64_data = image_url.split(",")[1]
                                image_bytes = base64.b64decode(base64_data)
                                image = Image.open(BytesIO(image_bytes))
                                images.append(image)
                            except Exception as e:
                                logger.error(f"Failed to process base64 image: {e}")
                        else:
                            # Handle URL images (would need to fetch)
                            logger.warning(f"URL images not yet supported: {image_url}")

        if images:
            multimodal_data["image"] = images

        return multimodal_data

    async def create_completion(
        self,
        messages: List[dict],
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Create a chat completion using vLLM"""
        try:
            # Convert messages to prompt
            prompt = self._format_messages_to_prompt(messages)

            # Extract multimodal data
            multimodal_data = self._extract_multimodal_data(messages)

            # Create sampling parameters
            sampling_params = self._create_sampling_params(temperature, max_tokens, **kwargs)

            # Generate response
            if multimodal_data:
                # Multimodal generation
                outputs = self.engine.generate(
                    prompt,
                    sampling_params=sampling_params,
                    multi_modal_data=multimodal_data
                )
            else:
                # Text-only generation
                outputs = self.engine.generate(prompt, sampling_params=sampling_params)

            # Extract response text
            if outputs and len(outputs) > 0:
                generated_text = outputs[0].outputs[0].text
                return generated_text.strip()
            else:
                raise ValueError("No output generated from vLLM")

        except Exception as e:
            logger.error(f"Error in vLLM completion: {e}")
            raise

    async def create_tool_completion(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        tool_choice: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """Create a tool-enabled completion (limited support)"""
        logger.warning("Tool calling with vLLM has experimental support.")

        # For basic tool support, we can append tool definitions to the system message
        if tools:
            tool_descriptions = []
            for tool in tools:
                if "function" in tool:
                    func_info = tool["function"]
                    tool_descriptions.append(f"Function: {func_info.get('name', 'unknown')}")
                    if "description" in func_info:
                        tool_descriptions.append(f"Description: {func_info['description']}")

            # Add tools info to system message
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] += f"\n\nAvailable tools:\n" + "\n".join(tool_descriptions)
            else:
                tool_message = {
                    "role": "system",
                    "content": f"Available tools:\n" + "\n".join(tool_descriptions)
                }
                messages.insert(0, tool_message)

        # Use regular completion
        response_text = await self.create_completion(
            messages, False, temperature, max_tokens, **kwargs
        )

        # Create mock response
        class MockMessage:
            def __init__(self, content):
                self.content = content
                self.tool_calls = None

        return MockMessage(response_text)


class VLLMModelProvider(ModelProvider):
    """vLLM provider wrapper implementing ModelProvider interface"""

    def __init__(self, model_name: str, **kwargs):
        # Lazy import vLLM when actually creating the provider
        if not _import_vllm():
            raise ImportError("vLLM support requires vllm package. Install with: pip install vllm")

        self.vllm_provider = VLLMProvider(model_name, **kwargs)

    async def create_completion(
        self,
        messages: List[dict],
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Create completion using vLLM - always returns string"""
        # vLLM doesn't support streaming in the same way, so we ignore stream parameter
        return await self.vllm_provider.create_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

    async def create_tool_completion(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        tool_choice: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ChatCompletionMessage:
        """Create tool completion using vLLM"""
        return await self.vllm_provider.create_tool_completion(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )


# ============================================================================
# DASHSCOPE PROVIDER
# ============================================================================

class DashScopeProvider(ModelProvider):
    """DashScope API provider implementation"""

    def __init__(self, model: str, api_base: str, api_key: str, **kwargs):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.timeout = kwargs.get('timeout', 60)
        self.retry = kwargs.get('retry', 5)
        self.wait = kwargs.get('wait', 5)

    def _convert_messages_to_dashscope_format(self, messages: List[dict]) -> List[dict]:
        """Convert OpenAI format messages to DashScope format"""
        try:
            import requests
            from PIL import Image
            from common_utils import encode_image_to_base64
        except ImportError:
            logger.warning("Missing dependencies for image processing in DashScope")

        formatted_messages = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, str):
                # Simple text message
                formatted_messages.append({
                    "role": role,
                    "content": [{"type": "text", "text": content}]
                })
            elif isinstance(content, list):
                # Multimodal content
                dashscope_content = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            dashscope_content.append({"type": "text", "text": item["text"]})
                        elif item.get("type") == "image_url":
                            image_url = item["image_url"]["url"]
                            if image_url.startswith("data:image"):
                                # Already base64 encoded
                                dashscope_content.append({
                                    "type": "image_url",
                                    "image_url": {"url": image_url}
                                })
                            else:
                                # Need to load and encode image
                                try:
                                    from common_utils import encode_image_to_base64
                                    image = Image.open(image_url)
                                    image_data = encode_image_to_base64(image)
                                    dashscope_content.append({
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                                    })
                                except Exception as e:
                                    logger.warning(f"Failed to load image {image_url}: {e}")
                                    continue

                formatted_messages.append({
                    "role": role,
                    "content": dashscope_content
                })

        return formatted_messages

    async def create_completion(
        self,
        messages: List[dict],
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Create completion using DashScope API"""
        import requests
        import time
        import asyncio

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }

        # Convert messages to DashScope format
        formatted_messages = self._convert_messages_to_dashscope_format(messages)

        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "max_completion_tokens": max_tokens or 4096,
            "temperature": temperature or 0.01,
            "stream": False
        }

        # Add additional parameters if provided
        if kwargs.get('top_p') is not None:
            payload["top_p"] = kwargs['top_p']
        if kwargs.get('top_k') is not None:
            payload["top_k"] = kwargs['top_k']

        for i in range(self.retry):
            try:
                # Use asyncio to make async request
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: requests.post(
                        self.api_base,
                        headers=headers,
                        json=payload,
                        timeout=self.timeout
                    )
                )

                if response.status_code == 200:
                    resp_json = response.json()

                    # Check for errors in response
                    if 'error' in resp_json:
                        logger.error(f"DashScope API error: {resp_json['error']}")
                        if i < self.retry - 1:
                            await asyncio.sleep(self.wait)
                            continue
                        else:
                            raise Exception(f"DashScope API error: {resp_json['error']}")

                    # Check finish reason
                    choices = resp_json.get('choices', [])
                    if not choices:
                        raise Exception("No choices returned from DashScope API")

                    finish_reason = choices[0].get('finish_reason')
                    if finish_reason not in ['stop', 'function_call']:
                        logger.warning(f"DashScope finished with reason: {finish_reason}")
                        if i < self.retry - 1:
                            await asyncio.sleep(self.wait)
                            continue

                    return choices[0]['message']['content'].strip()
                else:
                    logger.error(f"DashScope API error: HTTP {response.status_code}")
                    try:
                        error_content = response.json()
                        logger.error(f"Error details: {error_content}")
                    except:
                        logger.error(f"Raw error content: {response.content}")

                if i < self.retry - 1:
                    await asyncio.sleep(self.wait)

            except Exception as e:
                logger.error(f"DashScope request error (attempt {i+1}): {e}")
                if i < self.retry - 1:
                    await asyncio.sleep(self.wait)
                else:
                    raise

        raise Exception("Failed to get response from DashScope API after all retries")

    async def create_tool_completion(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        tool_choice: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ChatCompletionMessage:
        """Create tool completion using DashScope API"""
        logger.warning("Tool calling with DashScope has limited support. Falling back to regular completion.")

        # For now, just do regular completion and wrap in ChatCompletionMessage format
        response_text = await self.create_completion(messages, False, temperature, max_tokens, **kwargs)

        # Create a mock ChatCompletionMessage
        class MockMessage:
            def __init__(self, content):
                self.content = content
                self.tool_calls = None

        return MockMessage(response_text)


# ============================================================================
# ADAPTER FOR QWEN WITH MULTI-AGENT COMPATIBILITY
# ============================================================================

class QwenProvider(ModelProvider):
    """Qwen VLM provider implementation for multi-agent compatibility"""

    def __init__(self, qwen_provider, token_counter):
        self.qwen_provider = qwen_provider
        self.token_counter = token_counter

    async def create_completion(
        self,
        messages: List[dict],
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        try:
            # Convert messages to MPU conversation format
            conversation = messages_to_conversation(messages)

            # Use the Qwen provider from model.py
            response = self.qwen_provider.chat_completion(
                conversation,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            return response

        except Exception as e:
            logger.error(f"Error in Qwen VLM request: {e}")
            raise

    async def create_tool_completion(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        tool_choice: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ChatCompletionMessage:
        logger.warning("Tool calling with Qwen VLM has limited support. Falling back to regular completion.")

        # For now, just do regular completion and wrap in ChatCompletionMessage format
        response_text = await self.create_completion(messages, False, temperature, max_tokens, **kwargs)

        # Create a mock ChatCompletionMessage
        class MockMessage:
            def __init__(self, content):
                self.content = content
                self.tool_calls = None

        return MockMessage(response_text)


# ============================================================================
# FACTORY CLASSES
# ============================================================================

class ModelFactory:

    @staticmethod
    def create_openai(api_key: Optional[str] = None,
                     model: str = "gpt-4",
                     config: ModelConfig = None,
                     api_type: str = "openai",
                     **kwargs) -> OpenAIProvider:
        """Create OpenAI provider with unified interface"""
        return OpenAIProvider(api_key, model, config, api_type, **kwargs)

    @staticmethod
    def create_gpt4o(api_key: Optional[str] = None,
                     config: ModelConfig = None,
                     **kwargs) -> OpenAIProvider:
        """Create GPT-4o provider"""
        return OpenAIProvider(api_key, "gpt-4o", config, "openai", **kwargs)

    @staticmethod
    def create_gpt4o_mini(api_key: Optional[str] = None,
                          config: ModelConfig = None,
                          **kwargs) -> OpenAIProvider:
        """Create GPT-4o-mini provider"""
        return OpenAIProvider(api_key, "gpt-4o-mini", config, "openai", **kwargs)

    @staticmethod
    def create_gpt5(api_key: Optional[str] = None,
                    config: ModelConfig = None,
                    **kwargs) -> OpenAIProvider:
        """Create GPT-5 provider"""
        return OpenAIProvider(api_key, "gpt-5", config, "openai", **kwargs)

    @staticmethod
    def create_azure_openai(api_key: str,
                           base_url: str,
                           api_version: str,
                           model: str = "gpt-4",
                           config: ModelConfig = None) -> OpenAIProvider:
        """Create Azure OpenAI provider"""
        return OpenAIProvider(
            api_key, model, config, "azure", base_url=base_url, api_version=api_version
        )

    @staticmethod
    def create_qwen25vl(model_name: str = "7b",
                       device_map: str = "auto",
                       torch_dtype: str = "auto",
                       flash_attention: bool = False,
                       config: ModelConfig = None) -> Qwen25VLProvider:
        """Create a Qwen 2.5-VL provider.

        Args:
            model_name: Model variant ("3b", "7b") or full model path
            device_map: Device mapping for model loading
            torch_dtype: PyTorch data type for model
            flash_attention: Whether to use flash attention
            config: Model configuration
        """
        return Qwen25VLProvider(
            model_name=model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            flash_attention=flash_attention,
            config=config
        )

    @staticmethod
    def create_vllm(model_name: str,
                   tensor_parallel_size: int = 1,
                   gpu_memory_utilization: float = 0.95,
                   **kwargs) -> VLLMProvider:
        """Create vLLM provider for efficient local inference"""
        return VLLMProvider(
            model_name=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            **kwargs
        )

    @staticmethod
    def create_vllm_wrapper(model_name: str, **kwargs) -> VLLMModelProvider:
        """Create vLLM model provider wrapper"""
        return VLLMModelProvider(model_name, **kwargs)

    @staticmethod
    def create_dashscope(model: str,
                        api_base: str,
                        api_key: str,
                        **kwargs) -> DashScopeProvider:
        """Create DashScope provider"""
        return DashScopeProvider(model, api_base, api_key, **kwargs)


# ============================================================================
# EXAMPLE USAGE AND BACKWARD COMPATIBILITY
# ============================================================================

if __name__ == "__main__":
    # Create a conversation
    conversation = Conversation(
        system_prompt="You are a helpful AI assistant that can analyze images and answer questions."
    )

    # Add a text message
    conversation.add_user_message("Hello, can you help me?")

    print("=== Testing Model Providers ===")

    # Example 1: OpenAI GPT-4o (BaseModelProvider interface)
    try:
        print("\n1. Testing OpenAI GPT-4o (BaseModelProvider)...")
        gpt4o_provider = ModelFactory.create_gpt4o()
        response = gpt4o_provider.chat_completion(conversation)
        print("GPT-4o Response:", response)
    except Exception as e:
        print(f"GPT-4o Error: {e}")

    # Example 2: OpenAI GPT-4o (ModelProvider async interface)
    try:
        print("\n2. Testing OpenAI GPT-4o (ModelProvider async)...")
        gpt4o_provider = ModelFactory.create_gpt4o()
        messages = conversation_to_messages(conversation)
        import asyncio
        response = asyncio.run(gpt4o_provider.create_completion(messages))
        if hasattr(response, 'choices'):
            print("GPT-4o Async Response:", response.choices[0].message.content)
        else:
            print("GPT-4o Async Response:", response)
    except Exception as e:
        print(f"GPT-4o Async Error: {e}")

    # Example 3: vLLM Provider
    try:
        print("\n3. Testing vLLM Provider...")
        if VLLM_AVAILABLE:
            vllm_provider = ModelFactory.create_vllm("Qwen/Qwen2.5-VL-3B-Instruct")
            messages = conversation_to_messages(conversation)
            response = asyncio.run(vllm_provider.create_completion(messages))
            print("vLLM Response:", response)
        else:
            print("vLLM not available - install with: pip install vllm")
    except Exception as e:
        print(f"vLLM Error: {e}")

    # Example 4: Create Qwen 2.5-VL provider (7B model)
    try:
        print("\n4. Testing Qwen 2.5-VL Provider...")
        qwen_provider = ModelFactory.create_qwen25vl(model_name="7b")
        response = qwen_provider.chat_completion(conversation)
        print("Qwen 2.5-VL Response:", response)
    except Exception as e:
        print(f"Qwen Error: {e}")

    # Example 5: Multimodal conversation with image
    try:
        print("\n5. Testing Multimodal Conversation...")
        # Add image message
        image_content = [
            MessageContent(type=ContentType.IMAGE_URL, image_url="path/to/image.jpg"),
            MessageContent(type=ContentType.TEXT, text="What do you see in this image?")
        ]
        conversation.add_user_message(image_content)

        # Test with GPT-4o (supports vision)
        gpt4o_provider = ModelFactory.create_gpt4o()
        response = gpt4o_provider.chat_completion(conversation)
        print("GPT-4o Multimodal Response:", response)

    except Exception as e:
        print(f"Multimodal Error: {e}")

    print("\n=== All Tests Completed ===")


# ============================================================================
# EXPORTS FOR BACKWARD COMPATIBILITY
# ============================================================================

__all__ = [
    # Abstract base classes
    'BaseModelProvider',
    'ModelProvider',

    # Provider implementations
    'OpenAIProvider',
    'Qwen25VLProvider',
    'QwenProvider',
    'VLLMProvider',
    'VLLMModelProvider',
    'DashScopeProvider',

    # Factory
    'ModelFactory',

    # Utility functions
    'conversation_to_messages',
    'messages_to_conversation',

    # Availability flags
    'VLLM_AVAILABLE'
]