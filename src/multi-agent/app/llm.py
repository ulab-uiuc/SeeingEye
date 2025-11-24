"""
Simplified LLM interface that uses consolidated providers from src/model.py

This module provides the LLM class interface for multi-agent applications
while using the unified model providers from src/model.py
"""

import sys
import os
import requests
from typing import Dict, List, Optional, Union

# Add src directory to Python path for model.py imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from openai import (
    APIError,
    AsyncAzureOpenAI,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

# Import model providers with optimized loading
def _import_model_providers():
    """Import model providers, handling missing packages gracefully"""
    try:
        from model import (
            ModelFactory, BaseModelProvider, ModelProvider,
            DashScopeProvider, conversation_to_messages, messages_to_conversation,
            VLLM_AVAILABLE
        )
        # Try to import other providers, but don't fail if they're not available
        try:
            from model import OpenAIProvider, Qwen25VLProvider, QwenProvider
        except ImportError:
            OpenAIProvider = None
            Qwen25VLProvider = None
            QwenProvider = None

        try:
            from model import VLLMProvider, VLLMModelProvider
        except ImportError:
            VLLMProvider = None
            VLLMModelProvider = None

        return {
            'ModelFactory': ModelFactory,
            'BaseModelProvider': BaseModelProvider,
            'ModelProvider': ModelProvider,
            'OpenAIProvider': OpenAIProvider,
            'Qwen25VLProvider': Qwen25VLProvider,
            'QwenProvider': QwenProvider,
            'VLLMProvider': VLLMProvider,
            'VLLMModelProvider': VLLMModelProvider,
            'DashScopeProvider': DashScopeProvider,
            'conversation_to_messages': conversation_to_messages,
            'messages_to_conversation': messages_to_conversation,
            'VLLM_AVAILABLE': VLLM_AVAILABLE,
            'available': True
        }
    except ImportError as e:
        import warnings
        warnings.warn(f"Model providers not available: {e}. Limited functionality.")
        return {
            'ModelFactory': None,
            'BaseModelProvider': None,
            'ModelProvider': None,
            'OpenAIProvider': None,
            'Qwen25VLProvider': None,
            'QwenProvider': None,
            'VLLMProvider': None,
            'VLLMModelProvider': None,
            'DashScopeProvider': None,
            'conversation_to_messages': None,
            'messages_to_conversation': None,
            'VLLM_AVAILABLE': False,
            'available': False
        }

# Import providers
_imports = _import_model_providers()
ModelFactory = _imports['ModelFactory']
BaseModelProvider = _imports['BaseModelProvider']
ModelProvider = _imports['ModelProvider']
OpenAIProvider = _imports['OpenAIProvider']
Qwen25VLProvider = _imports['Qwen25VLProvider']
QwenProvider = _imports['QwenProvider']
VLLMProvider = _imports['VLLMProvider']
VLLMModelProvider = _imports['VLLMModelProvider']
DashScopeProvider = _imports['DashScopeProvider']
conversation_to_messages = _imports['conversation_to_messages']
messages_to_conversation = _imports['messages_to_conversation']
VLLM_AVAILABLE = _imports['VLLM_AVAILABLE']
MPU_IMPORTS_AVAILABLE = _imports['available']

# Import local app modules
from .token_counter import TokenCounter, TokenTracker

try:
    from .bedrock import BedrockClient
    BEDROCK_AVAILABLE = True
except ImportError:
    BedrockClient = None
    BEDROCK_AVAILABLE = False

from .config import LLMSettings, config
from .exceptions import TokenLimitExceeded
from .logger import logger
from .schema import (
    ROLE_VALUES,
    TOOL_CHOICE_TYPE,
    TOOL_CHOICE_VALUES,
    Message,
    ToolChoice,
)

REASONING_MODELS = ["o1", "o3-mini"]
MULTIMODAL_MODELS = [
    "gpt-4-vision-preview",
    "gpt-4o",
    "gpt-4o-mini",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
]


class LLM:
    """
    Unified LLM interface using consolidated providers from model.py

    This class provides the main interface for the multi-agent system while
    using the consolidated model providers for actual LLM interactions.
    """
    _instances: Dict[str, "LLM"] = {}

    def __new__(
        cls, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if config_name not in cls._instances:
            instance = super().__new__(cls)
            instance.__init__(config_name, llm_config)
            cls._instances[config_name] = instance
        return cls._instances[config_name]

    @classmethod
    def force_new_instance(
        cls, config_name: str, llm_config: Optional[LLMSettings] = None
    ) -> "LLM":
        """Force create a new LLM instance, bypassing singleton cache."""
        # Clear existing instance if it exists
        if config_name in cls._instances:
            del cls._instances[config_name]

        # Create new instance
        instance = super().__new__(cls)
        instance.__init__(config_name, llm_config)
        cls._instances[config_name] = instance
        return instance

    def __init__(
        self, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if not hasattr(self, "client"):  # Only initialize if not already initialized
            llm_config = llm_config or config.llm
            llm_config = llm_config.get(config_name, llm_config["default"])
            self.model = llm_config.model
            self.max_tokens = llm_config.max_tokens
            self.temperature = llm_config.temperature
            self.api_type = llm_config.api_type
            self.api_key = llm_config.api_key
            self.api_version = llm_config.api_version
            self.base_url = llm_config.base_url

            # Store config name for vLLM cluster integration
            self.config_name = config_name

            # Initialize token management using modular components
            max_input_tokens = (
                llm_config.max_input_tokens
                if hasattr(llm_config, "max_input_tokens")
                else None
            )
            self.token_tracker = TokenTracker(max_input_tokens)
            self.token_counter = TokenCounter(self.model)

            # Initialize the appropriate provider using consolidated model.py
            # Pass config name for vLLM cluster integration
            llm_config._config_name = config_name
            self.provider = self._create_provider(llm_config)

    def _create_provider(self, llm_config):
        """Factory method to create the appropriate provider using model.py"""

        # Check if this is a local vLLM API configuration and ensure cluster access
        if (self.api_type == "openai" and
            self.base_url and
            "localhost" in self.base_url and
            any(port in self.base_url for port in [":8000", ":8001"])):

            # This is a local vLLM API config - check if cluster access is already set up
            from .utils.vllm_setup import quick_vllm_setup, get_cached_vllm_servers, is_vllm_setup_complete
            import os

            # Determine config type and ports from current URL
            config_name = getattr(llm_config, '_config_name', 'unknown')

            # Extract port from current base_url
            if ":8000" in self.base_url:
                config_name = "translator_api"
                vision_port = 8000
                text_port = 8001  # Default for discovery
            elif ":8001" in self.base_url:
                config_name = "reasoning_api"
                vision_port = 8000  # Default for discovery
                text_port = 8001
            else:
                # Try to extract custom port
                import re
                port_match = re.search(r':(\d+)', self.base_url)
                if port_match:
                    current_port = int(port_match.group(1))
                    if current_port % 2 == 0:  # Even port = vision
                        config_name = "translator_api"
                        vision_port = current_port
                        text_port = current_port + 1
                    else:  # Odd port = text
                        config_name = "reasoning_api"
                        vision_port = current_port - 1
                        text_port = current_port
                else:
                    vision_port = 8000
                    text_port = 8001

            # Store ports for reconnection
            self._vllm_vision_port = vision_port
            self._vllm_text_port = text_port

            # Check if vLLM is already set up for these ports
            if is_vllm_setup_complete(vision_port, text_port):
                # Use cached server configuration
                active_servers = get_cached_vllm_servers(vision_port, text_port)
                if active_servers and config_name in active_servers:
                    correct_base_url = active_servers[config_name]['base_url']
                    if correct_base_url != self.base_url:
                        logger.info(f"🔄 Using cached vLLM config: {self.base_url} → {correct_base_url}")
                        self.base_url = correct_base_url
                else:
                    logger.debug(f"✅ vLLM already configured, keeping current URL: {self.base_url}")
            else:
                # First time setup - run the full setup process
                logger.info(f"🔗 First-time vLLM setup for {config_name}")

                # Try to set up vLLM access and get the correct base URL
                try:
                    success, active_servers = quick_vllm_setup(vision_port=vision_port, text_port=text_port)

                    if success and config_name in active_servers:
                        # Update the base URL to use the correct endpoint (localhost or direct node)
                        correct_base_url = active_servers[config_name]['base_url']
                        if correct_base_url != self.base_url:
                            logger.info(f"🔄 Updating base URL from {self.base_url} to {correct_base_url}")
                            self.base_url = correct_base_url
                    else:
                        logger.warning(f"⚠️  Could not establish vLLM cluster access for {config_name}")
                        logger.warning("Falling back to direct connection attempt")
                except Exception as e:
                    logger.warning(f"⚠️  Error setting up vLLM access: {e}")
                    logger.warning("Proceeding with original configuration")

        if not MPU_IMPORTS_AVAILABLE:
            raise ImportError("Model providers not available. Check model.py imports.")

        if self.api_type == "vllm":
            # Use vLLM for efficient local inference - check availability when needed
            # Import the lazy loading function
            from model import _import_vllm
            if not _import_vllm():
                raise ImportError("vLLM support requires vllm package. Install with: pip install vllm")

            # Extract vLLM specific parameters from config
            vllm_kwargs = {}
            if hasattr(llm_config, 'tensor_parallel_size'):
                vllm_kwargs['tensor_parallel_size'] = llm_config.tensor_parallel_size
            if hasattr(llm_config, 'gpu_memory_utilization'):
                vllm_kwargs['gpu_memory_utilization'] = llm_config.gpu_memory_utilization

            # GLIBC compatibility fixes
            if hasattr(llm_config, 'disable_custom_all_reduce'):
                vllm_kwargs['disable_custom_all_reduce'] = llm_config.disable_custom_all_reduce
            if hasattr(llm_config, 'enforce_eager'):
                vllm_kwargs['enforce_eager'] = llm_config.enforce_eager
            if hasattr(llm_config, 'max_model_len'):
                vllm_kwargs['max_model_len'] = llm_config.max_model_len

            return ModelFactory.create_vllm_wrapper(llm_config.model, **vllm_kwargs)

        elif self.api_type == "dashscope":
            # Use DashScope API
            dashscope_kwargs = {}
            if hasattr(llm_config, 'timeout'):
                dashscope_kwargs['timeout'] = llm_config.timeout
            if hasattr(llm_config, 'retry'):
                dashscope_kwargs['retry'] = llm_config.retry
            if hasattr(llm_config, 'wait'):
                dashscope_kwargs['wait'] = llm_config.wait

            # Use API key from config or environment variable
            api_key = llm_config.api_key or os.environ.get('DASHSCOPE_API_KEY', '')
            if not api_key:
                raise ValueError("DashScope API key not found. Set DASHSCOPE_API_KEY environment variable or api_key in config")

            return ModelFactory.create_dashscope(
                model=llm_config.model,
                api_base=self.base_url,
                api_key=api_key,
                **dashscope_kwargs
            )

        elif self.api_type == "qwen25vl":
            # Use model.py factory for Qwen VLM models
            qwen_provider = ModelFactory.create_qwen25vl(
                model_name=llm_config.model,
                device_map="auto"
            )
            return QwenProvider(qwen_provider, self.token_counter)

        else:
            # Create OpenAI-compatible client using model.py
            if self.api_type == "azure":
                return ModelFactory.create_azure_openai(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    api_version=self.api_version,
                    model=self.model
                )
            elif self.api_type == "aws":
                if BEDROCK_AVAILABLE:
                    # For AWS Bedrock, we'd need to create a BedrockProvider in model.py
                    # For now, fall back to OpenAI-compatible
                    client = BedrockClient()
                    return OpenAIProvider(client, self.model)
                else:
                    raise ImportError("AWS Bedrock support requires boto3. Install with: pip install boto3")
            else:
                # Standard OpenAI API
                return ModelFactory.create_openai(
                    api_key=self.api_key,
                    model=self.model,
                    base_url=self.base_url
                )

    def count_tokens(self, text: str) -> int:
        """Calculate the number of tokens in a text"""
        return self.token_counter.count_text(text)

    def count_message_tokens(self, messages: List[dict]) -> int:
        return self.token_counter.count_message_tokens(messages)


    def update_token_count(self, input_tokens: int, completion_tokens: int = 0) -> None:
        """Update token counts"""
        self.token_tracker.update_token_count(input_tokens, completion_tokens)
        usage = self.token_tracker.get_usage_summary()
        logger.info(
            f"Token usage: Input={input_tokens}, Completion={completion_tokens}, "
            f"Cumulative Input={usage['input_tokens']}, Cumulative Completion={usage['completion_tokens']}, "
            f"Total={input_tokens + completion_tokens}, Cumulative Total={usage['total_tokens']}"
        )

    def check_token_limit(self, input_tokens: int) -> bool:
        """Check if token limits are exceeded"""
        return self.token_tracker.check_token_limit(input_tokens)

    def get_limit_error_message(self, input_tokens: int) -> str:
        """Generate error message for token limit exceeded"""
        return self.token_tracker.get_limit_error_message(input_tokens)

    def _extract_sampling_params(self) -> tuple[dict, dict]:
        """Extract sampling parameters from config for OpenAI-compatible APIs.

        Returns:
            tuple[dict, dict]: (standard_openai_params, vllm_specific_params)
        """
        standard_params = {}
        vllm_params = {}

        if self.api_type == "openai" and self.base_url and "localhost" in self.base_url:
            # This is a local vLLM API - pass sampling parameters
            llm_config_dict = config.llm.get(self.config_name, config.llm["default"])

            # Convert LLMSettings object to dict if needed
            if hasattr(llm_config_dict, '__dict__'):
                llm_config_values = llm_config_dict.__dict__
            else:
                llm_config_values = llm_config_dict
            if 'top_p' in llm_config_values and llm_config_values['top_p'] is not None:
                standard_params['top_p'] = llm_config_values['top_p']
            if 'frequency_penalty' in llm_config_values and llm_config_values['frequency_penalty'] is not None:
                standard_params['frequency_penalty'] = llm_config_values['frequency_penalty']
            if 'presence_penalty' in llm_config_values and llm_config_values['presence_penalty'] is not None:
                standard_params['presence_penalty'] = llm_config_values['presence_penalty']

            # vLLM-specific parameters (passed via extra_body)
            if 'top_k' in llm_config_values and llm_config_values['top_k'] is not None:
                vllm_params['top_k'] = llm_config_values['top_k']
            if 'repetition_penalty' in llm_config_values and llm_config_values['repetition_penalty'] is not None:
                vllm_params['repetition_penalty'] = llm_config_values['repetition_penalty']

            # # Debug: Log the parameters being sent to vLLM
            # if standard_params or vllm_params:
            #     logger.info(f"🎯 Standard OpenAI parameters: {standard_params}")
            #     logger.info(f"🎯 vLLM-specific parameters: {vllm_params}")

        return standard_params, vllm_params

    def _test_connection_health(self) -> bool:
        """Test if current API endpoint is responsive"""
        try:
            if (self.api_type == "openai" and
                self.base_url and
                "localhost" in self.base_url):
                health_url = self.base_url.replace('/v1', '/health')
                response = requests.get(health_url, timeout=5)
                return response.status_code == 200
        except Exception:
            pass
        return True  # Assume healthy for non-vLLM endpoints

    def _attempt_reconnection(self) -> bool:
        """Attempt to reestablish vLLM cluster connection"""
        if (self.api_type == "openai" and
            self.base_url and
            "localhost" in self.base_url and
            hasattr(self, '_vllm_vision_port') and
            hasattr(self, '_vllm_text_port')):

            from .utils.vllm_setup import reconnect_vllm_servers

            logger.info(f"🔄 Attempting vLLM reconnection for {self.config_name}")

            try:
                success, active_servers = reconnect_vllm_servers(
                    self._vllm_vision_port,
                    self._vllm_text_port
                )

                if success and self.config_name in active_servers:
                    old_url = self.base_url
                    self.base_url = active_servers[self.config_name]['base_url']
                    logger.info(f"✅ Reconnected: {old_url} → {self.base_url}")

                    # Recreate provider with new URL
                    llm_config = config.llm.get(self.config_name, config.llm["default"])
                    llm_config._config_name = self.config_name
                    self.provider = self._create_provider(llm_config)

                    return True
                else:
                    logger.warning(f"❌ Reconnection failed for {self.config_name}")

            except Exception as e:
                logger.warning(f"❌ Reconnection error: {e}")

        return False

    @staticmethod
    def format_messages(
        messages: List[Union[dict, Message]], supports_images: bool = False
    ) -> List[dict]:
        """
        Format messages for LLM by converting them to OpenAI message format.

        Args:
            messages: List of messages that can be either dict or Message objects
            supports_images: Flag indicating if the target model supports image inputs

        Returns:
            List[dict]: List of formatted messages in OpenAI format

        Raises:
            ValueError: If messages are invalid or missing required fields
            TypeError: If unsupported message types are provided

        Examples:
            >>> msgs = [
            ...     Message.system_message("You are a helpful assistant"),
            ...     {"role": "user", "content": "Hello"},
            ...     Message.user_message("How are you?")
            ... ]
            >>> formatted = LLM.format_messages(msgs)
        """
        formatted_messages = []

        for message in messages:
            # Convert Message objects to dictionaries
            if isinstance(message, Message):
                message = message.to_dict()

            if isinstance(message, dict):
                # If message is a dict, ensure it has required fields
                if "role" not in message:
                    raise ValueError("Message dict must contain 'role' field")

                # Process base64 images if present and model supports images
                if supports_images and message.get("base64_image"):
                    # Initialize or convert content to appropriate format
                    if not message.get("content"):
                        message["content"] = []
                    elif isinstance(message["content"], str):
                        message["content"] = [
                            {"type": "text", "text": message["content"]}
                        ]
                    elif isinstance(message["content"], list):
                        # Convert string items to proper text objects
                        message["content"] = [
                            (
                                {"type": "text", "text": item}
                                if isinstance(item, str)
                                else item
                            )
                            for item in message["content"]
                        ]

                    # Add the image to content
                    message["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{message['base64_image']}"
                            },
                        }
                    )

                    # Remove the base64_image field
                    del message["base64_image"]
                # If model doesn't support images but message has base64_image, handle gracefully
                elif not supports_images and message.get("base64_image"):
                    # Just remove the base64_image field and keep the text content
                    del message["base64_image"]

                if "content" in message or "tool_calls" in message:
                    formatted_messages.append(message)
                # else: do not include the message
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")

        # Validate all messages have required fields
        for msg in formatted_messages:
            if msg["role"] not in ROLE_VALUES:
                raise ValueError(f"Invalid role: {msg['role']}")

        return formatted_messages

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # Don't retry TokenLimitExceeded
    )
    async def ask(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = True,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a prompt to the LLM and get the response.

        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            stream (bool): Whether to stream the response
            temperature (float): Sampling temperature for the response

        Returns:
            str: The generated response

        Raises:
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If messages are invalid or response is empty
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            # Check if the model supports images
            supports_images = self.model in MULTIMODAL_MODELS or self.api_type in ["qwen25vl", "vllm", "dashscope"]

            # Format system and user messages with image support check
            if system_msgs:
                system_msgs = self.format_messages(system_msgs, supports_images)
                messages = system_msgs + self.format_messages(messages, supports_images)
            else:
                messages = self.format_messages(messages, supports_images)

            # Calculate input token count
            input_tokens = self.count_message_tokens(messages)

            # Check if token limits are exceeded
            if not self.check_token_limit(input_tokens):
                error_message = self.get_limit_error_message(input_tokens)
                # Raise a special exception that won't be retried
                raise TokenLimitExceeded(error_message)

            # Use provider to create completion
            temperature = temperature if temperature is not None else self.temperature

            # Extract sampling parameters for vLLM API
            standard_params, vllm_params = self._extract_sampling_params()
            completion_kwargs = {**standard_params, **vllm_params}

            response = await self.provider.create_completion(
                messages, stream=stream, temperature=temperature, max_tokens=self.max_tokens, **completion_kwargs
            )

            # Handle response based on provider type
            if isinstance(response, str):
                # Qwen provider returns string directly
                completion_tokens = self.count_tokens(response)
                self.update_token_count(input_tokens, completion_tokens)
                return response

            # OpenAI provider returns ChatCompletion
            if not stream:
                # Non-streaming request
                if not response.choices or not response.choices[0].message.content:
                    raise ValueError("Empty or invalid response from LLM")

                # Update token counts
                self.update_token_count(
                    response.usage.prompt_tokens, response.usage.completion_tokens
                )

                return response.choices[0].message.content
            else:
                # Streaming request - For streaming, update estimated token count before making the request
                self.update_token_count(input_tokens)

                collected_messages = []
                completion_text = ""
                async for chunk in response:
                    chunk_message = chunk.choices[0].delta.content or ""
                    collected_messages.append(chunk_message)
                    completion_text += chunk_message
                    print(chunk_message, end="", flush=True)

                print()  # Newline after streaming
                full_response = "".join(collected_messages).strip()
                if not full_response:
                    raise ValueError("Empty response from streaming LLM")

                # estimate completion tokens for streaming response
                completion_tokens = self.count_tokens(completion_text)
                logger.info(
                    f"Estimated completion tokens for streaming response: {completion_tokens}"
                )
                # Update token count using the proper method
                self.update_token_count(0, completion_tokens)

                return full_response

        except TokenLimitExceeded:
            # Re-raise token limit errors without logging
            raise
        except ValueError:
            logger.exception(f"Validation error")
            raise
        except OpenAIError as oe:
            logger.exception(f"OpenAI API error")
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
                # Check if this might be a connection issue that we can fix
                if any(keyword in str(oe).lower() for keyword in ['connection', 'timeout', 'network']):
                    logger.info("🔗 Detected connection-related API error, checking health...")
                    if not self._test_connection_health():
                        logger.info("🔄 Connection unhealthy, attempting reconnection...")
                        if self._attempt_reconnection():
                            logger.info("✅ Reconnection successful, retrying will occur")
                        else:
                            logger.warning("❌ Reconnection failed")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error in ask")
            # Check for connection-related exceptions
            if any(keyword in str(e).lower() for keyword in ['connection', 'timeout', 'network']):
                logger.info("🔗 Detected connection-related error, checking health...")
                if not self._test_connection_health():
                    logger.info("🔄 Connection unhealthy, attempting reconnection...")
                    if self._attempt_reconnection():
                        logger.info("✅ Reconnection successful, retrying will occur")
                    else:
                        logger.warning("❌ Reconnection failed")
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # Don't retry TokenLimitExceeded
    )
    async def ask_with_images(
        self,
        messages: List[Union[dict, Message]],
        images: List[Union[str, dict]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a prompt with images to the LLM and get the response.

        Args:
            messages: List of conversation messages
            images: List of image URLs or image data dictionaries
            system_msgs: Optional system messages to prepend
            stream (bool): Whether to stream the response
            temperature (float): Sampling temperature for the response

        Returns:
            str: The generated response

        Raises:
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If messages are invalid or response is empty
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            # For ask_with_images, we always set supports_images to True because
            # this method should only be called with models that support images
            # Allow Qwen VLM and vLLM models for image support
            if self.model not in MULTIMODAL_MODELS and self.api_type not in ["qwen25vl", "vllm", "dashscope"]:
                raise ValueError(
                    f"Model {self.model} does not support images. Use a model from {MULTIMODAL_MODELS} or use vLLM/Qwen VLM/DashScope"
                )

            # Format messages with image support
            formatted_messages = self.format_messages(messages, supports_images=True)

            # Ensure the last message is from the user to attach images
            if not formatted_messages or formatted_messages[-1]["role"] != "user":
                raise ValueError(
                    "The last message must be from the user to attach images"
                )

            # Process the last user message to include images
            last_message = formatted_messages[-1]

            # Convert content to multimodal format if needed
            content = last_message["content"]
            multimodal_content = (
                [{"type": "text", "text": content}]
                if isinstance(content, str)
                else content
                if isinstance(content, list)
                else []
            )

            # Add images to content
            for image in images:
                if isinstance(image, str):
                    multimodal_content.append(
                        {"type": "image_url", "image_url": {"url": image}}
                    )
                elif isinstance(image, dict) and "url" in image:
                    multimodal_content.append({"type": "image_url", "image_url": image})
                elif isinstance(image, dict) and "image_url" in image:
                    multimodal_content.append(image)
                else:
                    raise ValueError(f"Unsupported image format: {image}")

            # Update the message with multimodal content
            last_message["content"] = multimodal_content

            # Add system messages if provided
            if system_msgs:
                all_messages = (
                    self.format_messages(system_msgs, supports_images=True)
                    + formatted_messages
                )
            else:
                all_messages = formatted_messages

            # Calculate tokens and check limits
            input_tokens = self.count_message_tokens(all_messages)
            if not self.check_token_limit(input_tokens):
                raise TokenLimitExceeded(self.get_limit_error_message(input_tokens))

            # Use provider to create completion
            temperature = temperature if temperature is not None else self.temperature

            # Extract sampling parameters for vLLM API
            standard_params, vllm_params = self._extract_sampling_params()
            completion_kwargs = {**standard_params, **vllm_params}

            response = await self.provider.create_completion(
                all_messages, stream=stream, temperature=temperature, max_tokens=self.max_tokens, **completion_kwargs
            )

            # Handle response based on provider type
            if isinstance(response, str):
                # Qwen provider returns string directly
                completion_tokens = self.count_tokens(response)
                self.update_token_count(input_tokens, completion_tokens)
                return response

            # OpenAI provider returns ChatCompletion
            if not stream:
                if not response.choices or not response.choices[0].message.content:
                    raise ValueError("Empty or invalid response from LLM")

                self.update_token_count(response.usage.prompt_tokens)
                return response.choices[0].message.content
            else:
                # Handle streaming request
                self.update_token_count(input_tokens)

                collected_messages = []
                async for chunk in response:
                    chunk_message = chunk.choices[0].delta.content or ""
                    collected_messages.append(chunk_message)
                    print(chunk_message, end="", flush=True)

                print()  # Newline after streaming
                full_response = "".join(collected_messages).strip()

                if not full_response:
                    raise ValueError("Empty response from streaming LLM")

                return full_response

        except TokenLimitExceeded:
            raise
        except ValueError as ve:
            logger.error(f"Validation error in ask_with_images: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API error: {oe}")
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
                # Check if this might be a connection issue that we can fix
                if any(keyword in str(oe).lower() for keyword in ['connection', 'timeout', 'network']):
                    logger.info("🔗 Detected connection-related API error, checking health...")
                    if not self._test_connection_health():
                        logger.info("🔄 Connection unhealthy, attempting reconnection...")
                        if self._attempt_reconnection():
                            logger.info("✅ Reconnection successful, retrying will occur")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask_with_images: {e}")
            # Check for connection-related exceptions
            if any(keyword in str(e).lower() for keyword in ['connection', 'timeout', 'network']):
                logger.info("🔗 Detected connection-related error, checking health...")
                if not self._test_connection_health():
                    logger.info("🔄 Connection unhealthy, attempting reconnection...")
                    if self._attempt_reconnection():
                        logger.info("✅ Reconnection successful, retrying will occur")
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # Don't retry TokenLimitExceeded
    )
    async def ask_tool(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        timeout: int = 300,
        tools: Optional[List[dict]] = None,
        tool_choice: TOOL_CHOICE_TYPE = ToolChoice.AUTO,  # type: ignore
        temperature: Optional[float] = None,
        **kwargs,
    ) -> ChatCompletionMessage | None:
        """
        Ask LLM using functions/tools and return the response.

        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            timeout: Request timeout in seconds
            tools: List of tools to use
            tool_choice: Tool choice strategy
            temperature: Sampling temperature for the response
            **kwargs: Additional completion arguments

        Returns:
            ChatCompletionMessage: The model's response

        Raises:
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If tools, tool_choice, or messages are invalid
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            # Validate tool_choice
            if tool_choice not in TOOL_CHOICE_VALUES:
                raise ValueError(f"Invalid tool_choice: {tool_choice}")

            # Check if the model supports images
            supports_images = self.model in MULTIMODAL_MODELS or self.api_type in ["qwen25vl", "vllm", "dashscope"]

            # Format messages
            if system_msgs:
                system_msgs = self.format_messages(system_msgs, supports_images)
                messages = system_msgs + self.format_messages(messages, supports_images)
            else:
                messages = self.format_messages(messages, supports_images)

            # Calculate input token count
            input_tokens = self.count_message_tokens(messages)

            # If there are tools, calculate token count for tool descriptions
            tools_tokens = 0
            if tools:
                for tool in tools:
                    tools_tokens += self.count_tokens(str(tool))

            input_tokens += tools_tokens

            # Check if token limits are exceeded
            if not self.check_token_limit(input_tokens):
                error_message = self.get_limit_error_message(input_tokens)
                # Raise a special exception that won't be retried
                raise TokenLimitExceeded(error_message)

            # Validate tools if provided
            if tools:
                for tool in tools:
                    if not isinstance(tool, dict) or "type" not in tool:
                        raise ValueError("Each tool must be a dict with 'type' field")

            # Use provider to create tool completion
            temperature = temperature if temperature is not None else self.temperature

            # Extract sampling parameters for vLLM API and merge with kwargs
            standard_params, vllm_params = self._extract_sampling_params()
            completion_kwargs = {**kwargs, **standard_params, **vllm_params}

            # Debug logging: output raw messages before API call
            logger.debug(f"🔍 DEBUG: About to send {len(messages)} messages to API")
            for i, msg in enumerate(messages):
                logger.debug(f"🔍 DEBUG: Message {i}: {msg}")

            response_message = await self.provider.create_tool_completion(
                messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                max_tokens=self.max_tokens,
                timeout=timeout,
                **completion_kwargs
            )

            # Check if response is valid
            if not response_message:
                return None

            # Update token counts - estimate for Qwen, use actual for OpenAI
            if isinstance(response_message.content, str):
                completion_tokens = self.count_tokens(response_message.content)
                self.update_token_count(input_tokens, completion_tokens)
            else:
                # For OpenAI, tokens should be tracked in the provider
                pass

            return response_message

        except TokenLimitExceeded:
            # Re-raise token limit errors without logging
            raise
        except ValueError as ve:
            logger.error(f"Validation error in ask_tool: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API error: {oe}")

            # Output raw messages that caused the error for debugging
            logger.error(f"🚨 ERROR DEBUG: Failed API call had {len(messages)} messages:")
            for i, msg in enumerate(messages):
                logger.error(f"🚨 ERROR DEBUG: Message {i}: {msg}")

            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
                # Check if this might be a connection issue that we can fix
                if any(keyword in str(oe).lower() for keyword in ['connection', 'timeout', 'network']):
                    logger.info("🔗 Detected connection-related API error, checking health...")
                    if not self._test_connection_health():
                        logger.info("🔄 Connection unhealthy, attempting reconnection...")
                        if self._attempt_reconnection():
                            logger.info("✅ Reconnection successful, retrying will occur")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask_tool: {e}")
            # Check for connection-related exceptions
            if any(keyword in str(e).lower() for keyword in ['connection', 'timeout', 'network']):
                logger.info("🔗 Detected connection-related error, checking health...")
                if not self._test_connection_health():
                    logger.info("🔄 Connection unhealthy, attempting reconnection...")
                    if self._attempt_reconnection():
                        logger.info("✅ Reconnection successful, retrying will occur")
            raise