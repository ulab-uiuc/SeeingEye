"""
vLLM Provider for efficient local inference
Supports text, vision-language, and tool calling capabilities
"""

import base64
import json
from io import BytesIO
from typing import Dict, List, Optional, Union
from PIL import Image

try:
    from vllm import LLM as vLLM_Engine, SamplingParams
    from vllm.multimodal import MultiModalDataDict
    VLLM_AVAILABLE = True
except ImportError:
    vLLM_Engine = None
    SamplingParams = None
    MultiModalDataDict = None
    VLLM_AVAILABLE = False
    import warnings
    warnings.warn("vLLM not available. Install with: pip install vllm")

from app.logger import logger


class VLLMProvider:
    """vLLM provider implementation for efficient local inference"""

    def __init__(self, model_name: str, tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.95, **kwargs):
        if not VLLM_AVAILABLE:
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


# Utility functions for testing
def create_vllm_provider(model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", **kwargs) -> VLLMProvider:
    """Factory function to create vLLM provider"""
    return VLLMProvider(model_name, **kwargs)


def format_image_message(text: str, image_path: str) -> List[dict]:
    """Helper to format image message for testing"""
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
            }
        ]
    }]