"""
Base Benchmark Inference System

This module provides a unified base class for all benchmark inference implementations.
It consolidates common functionality like image processing, logging, error handling,
and token tracking that is shared across MMMU, MIA, and other benchmarks.
"""

import os
import sys
import json
import hashlib
import requests
import io
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path

# Image processing
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    Image = None
    PIL_AVAILABLE = False

# Import flow and logging components
try:
    from ..flow.flow_executor import FlowExecutor
    from .log_save import LogSave
    from ..config import config
    FLOW_AVAILABLE = True
except ImportError:
    FlowExecutor = None
    LogSave = None
    config = None
    FLOW_AVAILABLE = False


class BenchmarkInfer(ABC):
    """
    Abstract base class for benchmark inference implementations.

    Provides common functionality for:
    - Flow execution with multi-agent systems
    - Comprehensive logging and session management
    - Image processing and caching
    - Error handling and retry logic
    - Token usage tracking
    - Result generation
    """

    def __init__(self,
                 benchmark_name: str,
                 image_cache_dir: Optional[str] = None):
        """
        Initialize benchmark inference system.

        Args:
            benchmark_name: Name of the benchmark (e.g., 'mmmu', 'mia')
            image_cache_dir: Custom directory for image caching
        """
        if not FLOW_AVAILABLE:
            raise ImportError("Flow system not available. Check imports.")

        self.benchmark_name = benchmark_name

        # Initialize flow executor (reads config directly)
        print(f"🤖 Initializing {benchmark_name.upper()} Flow System...")
        self.flow_executor = FlowExecutor()

        # Initialize logging system
        self.log_save = LogSave(benchmark_name=benchmark_name)
        self.session_id = None

        # Setup image cache directory
        if image_cache_dir:
            self.image_cache_dir = Path(image_cache_dir)
        else:
            self.image_cache_dir = Path(__file__).parent.parent.parent / "benchmark_evaluation" / benchmark_name / "image_cache"

        self.image_cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"📝 Logging initialized for benchmark: {benchmark_name}")
        print(f"🖼️ Image cache: {self.image_cache_dir}")
        print(f"✅ {benchmark_name.upper()} Flow System ready")

    # ==================== Session Management ====================

    def start_logging_session(self, experiment_config: Optional[Dict[str, Any]] = None) -> str:
        """Start a new logging session for the benchmark run."""
        self.session_id = self.log_save.start_question_session(experiment_config=experiment_config)
        print(f"📝 Started logging session: {self.session_id}")
        return self.session_id

    def finish_logging_session(self):
        """Finish the current logging session."""
        if self.session_id:
            self.log_save.finish_question_session()
            print(f"📝 Finished logging session: {self.session_id}")
            self.session_id = None

    @property
    def session_dir(self) -> Optional[Path]:
        """Get current session directory."""
        if self.session_id and hasattr(self.log_save, 'current_session'):
            return self.log_save.current_session.session_dir
        return None

    # ==================== Data Processing ====================

    def _convert_numpy_types(self, obj: Any) -> Any:
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj

    def clean_sample_data(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Clean sample data for JSON serialization."""
        cleaned = {}
        for k, v in sample.items():
            if isinstance(v, (np.integer, np.floating)):
                cleaned[k] = float(v) if isinstance(v, np.floating) else int(v)
            else:
                cleaned[k] = v
        return cleaned

    # ==================== Image Processing ====================

    def load_and_cache_image(self, image_url_or_path: str) -> str:
        """
        Load image from URL or local path and cache it locally.

        Args:
            image_url_or_path: URL or local path to image

        Returns:
            Local cached image path
        """
        if not PIL_AVAILABLE:
            raise ImportError("PIL is required for image processing")

        try:
            # Generate cache filename from URL/path
            cache_name = hashlib.md5(image_url_or_path.encode()).hexdigest() + ".jpg"
            cache_path = self.image_cache_dir / cache_name

            # Check if already cached
            if cache_path.exists():
                return str(cache_path)

            # Load image
            if image_url_or_path.startswith(('http://', 'https://')):
                # Load from URL
                response = requests.get(image_url_or_path, timeout=30)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content))
            else:
                # Load from local path
                if not os.path.isabs(image_url_or_path):
                    # Make relative to benchmark data directory
                    benchmark_dir = Path(__file__).parent.parent.parent / "benchmark_evaluation" / self.benchmark_name
                    image_path = benchmark_dir / "data" / image_url_or_path
                else:
                    image_path = Path(image_url_or_path)

                if not image_path.exists():
                    raise FileNotFoundError(f"Image file not found: {image_path}")

                image = Image.open(image_path)

            # Convert to RGB and save to cache
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Resize if too large to avoid API limits
            max_size = (2048, 2048)  # Reasonable limit for most APIs
            if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                image.thumbnail(max_size, Image.Resampling.LANCZOS)

            image.save(cache_path, format='JPEG', quality=85)
            return str(cache_path)

        except Exception as e:
            print(f"❌ Error processing image {image_url_or_path}: {e}")
            raise

    # ==================== Token Tracking ====================

    def _collect_token_usage_from_flow(self) -> Dict[str, Any]:
        """Collect per-question token usage statistics from all agents in the flow."""
        try:
            from .log_save import collect_flow_token_usage
            return collect_flow_token_usage(self.flow_executor)
        except ImportError:
            return {"input_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _collect_detailed_token_usage_from_flow(self) -> Dict[str, Any]:
        """Collect per-question token usage from all agents in the flow."""
        try:
            from .log_save import collect_flow_token_usage_detailed

            return collect_flow_token_usage_detailed(self.flow_executor)
        except ImportError:
            return {"input_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _reset_token_counters(self):
        """Reset token counters before starting a new question."""
        try:
            from .log_save import reset_flow_token_counters
            reset_flow_token_counters(self.flow_executor)
        except ImportError:
            pass

    # ==================== Error Handling ====================

    def _clean_error_message(self, error_msg: str) -> str:
        """Clean error message to remove base64 image data and other sensitive content."""
        # Remove base64 image data
        clean_msg = error_msg.replace('"base64_image":', '"base64_image": "[EXCLUDED]",')

        # Remove long base64-like strings
        import re
        clean_msg = re.sub(r'"[A-Za-z0-9+/]{30,}={0,2}"', '"[BASE64_DATA_EXCLUDED]"', clean_msg)

        # Truncate extremely long error messages
        if len(clean_msg) > 1000:
            clean_msg = clean_msg[:1000] + "... [TRUNCATED]"

        return clean_msg

    def _create_error_info(self, error_str: str) -> tuple[str, Dict[str, Any]]:
        """
        Create standardized error information.

        Returns:
            Tuple of (clean_error_message, error_info_dict)
        """
        critical_error_info = {
            "timestamp": datetime.now().isoformat(),
            "error_message": self._clean_error_message(error_str),
            "full_traceback": error_str if len(error_str) < 1000 else error_str[:1000] + "...[truncated]"
        }

        if "APIConnectionError" in error_str:
            clean_error_msg = "Error: API Connection Error (network/server issue)"
            critical_error_info.update({
                "error_type": "APIConnectionError",
                "severity": "high",
                "rerun_recommended": True
            })
        elif "RetryError" in error_str:
            clean_error_msg = "Error: API request failed after retries"
            critical_error_info.update({
                "error_type": "RetryError",
                "severity": "high",
                "rerun_recommended": True
            })
        elif "TokenLimitExceeded" in error_str:
            clean_error_msg = "Error: Token limit exceeded"
            critical_error_info.update({
                "error_type": "TokenLimitExceeded",
                "severity": "medium",
                "rerun_recommended": False
            })
        else:
            clean_error_str = self._clean_error_message(error_str)
            clean_error_msg = f"Error: {clean_error_str}"
            critical_error_info.update({
                "error_type": "UnknownError",
                "severity": "medium",
                "rerun_recommended": True
            })

        return clean_error_msg, critical_error_info

    def _detect_final_iteration_tool_errors(self, response: str) -> List[Dict[str, Any]]:
        """
        Detect if tool errors occurred in the final iteration that led to termination.

        Args:
            response: The final response from the flow execution

        Returns:
            List of critical error dictionaries
        """
        critical_errors = []

        # Check if the response indicates termination due to max steps/iterations
        # and contains tool errors
        if "Terminated: Reached max" in response or "Terminated: Max" in response:

            # Look for tool error patterns in the final steps
            lines = response.split('\n')

            # Get the last few steps to check for tool errors
            final_steps = []
            for line in lines:
                if line.startswith('Step '):
                    final_steps.append(line)

            # Check if the last step(s) contain tool errors
            if final_steps:
                last_step = final_steps[-1]

                # Pattern: "Step X: Error: ⚠️ Tool 'name' encountered a problem: details"
                if ("Error: ⚠️ Tool" in last_step and "encountered a problem" in last_step):
                    # Extract tool name and error details
                    tool_error_start = last_step.find("Tool '") + 6
                    tool_error_end = last_step.find("'", tool_error_start)
                    tool_name = last_step[tool_error_start:tool_error_end] if tool_error_start > 5 else "unknown"

                    error_detail_start = last_step.find("encountered a problem: ") + 23
                    error_details = last_step[error_detail_start:] if error_detail_start > 22 else "unknown error"

                    critical_error = {
                        "error_type": "FinalIterationToolFailure",
                        "error_message": f"Tool '{tool_name}' failed in final iteration: {error_details}",
                        "tool_name": tool_name,
                        "step_details": last_step,
                        "severity": "high",
                        "rerun_recommended": True,
                        "termination_cause": "max_iterations_with_tool_failure"
                    }
                    critical_errors.append(critical_error)

        return critical_errors

    # ==================== Flow Execution ====================

    def execute_on_sample(self,
                         sample: Dict[str, Any],
                         question_metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Execute the flow on a single sample with comprehensive logging.

        Args:
            sample: Sample data dictionary
            question_metadata: Optional metadata for logging

        Returns:
            Response string from the flow
        """
        try:
            # Extract question information for logging
            question = self.extract_question(sample)
            options = self.extract_options(sample)
            expected_answer = self.extract_expected_answer(sample)
            custom_question_id = self.extract_question_id(sample)

            # Start individual question logging
            if self.session_id:
                # Clean metadata of numpy types before logging
                metadata = question_metadata or self.extract_metadata(sample)
                clean_metadata = self._convert_numpy_types(metadata)

                self.log_save.start_individual_question(
                    question=question,
                    options=options,
                    expected_answer=expected_answer,
                    question_metadata=clean_metadata,
                    custom_question_id=custom_question_id
                )

                # Reset token counters for per-question tracking
                self._reset_token_counters()

            # Build flow input from sample (benchmark-specific)
            flow_input = self.build_flow_input(sample)

            # Execute using the modular flow executor with log_save context
            result = self.flow_executor.execute(flow_input, log_save=self.log_save)
            response = result["response"]

            # Finish individual question logging
            if self.session_id:
                # Extract evaluation results if available
                evaluation_results = {
                    'execution_time': result.get('execution_time', 0),
                    'metadata': result.get('metadata', {})
                }

                # Collect per-question token usage
                per_question_usage = self._collect_detailed_token_usage_from_flow()

                # Check for critical errors in final iteration (tool failures that led to termination)
                critical_errors = self._detect_final_iteration_tool_errors(response)

                self.log_save.finish_individual_question(
                    model_response=response,
                    evaluation_results=evaluation_results,
                    token_usage=per_question_usage,
                    critical_errors=critical_errors
                )

            return response

        except Exception as e:
            # Handle errors with comprehensive logging
            error_str = str(e)
            clean_error_msg, critical_error_info = self._create_error_info(error_str)

            print(f"Error during {self.benchmark_name} sample execution: {self._clean_error_message(error_str)}")

            # Log the error if session is active
            if self.session_id and hasattr(self, 'log_save') and self.log_save.current_question:
                # Try to collect per-question token usage even in error case
                per_question_usage = self._collect_detailed_token_usage_from_flow()

                self.log_save.finish_individual_question(
                    model_response=clean_error_msg,
                    evaluation_results={'error': True, 'error_message': self._clean_error_message(error_str)},
                    token_usage=per_question_usage,
                    critical_errors=[critical_error_info]
                )

            return clean_error_msg

    # ==================== Experiment Configuration ====================

    def build_experiment_config(self, args) -> Dict[str, Any]:
        """Build experiment configuration for logging."""
        try:
            from ..config import config
            # Get max_iterations from the flow executor
            max_iterations = getattr(self.flow_executor.underlying_flow, 'max_iterations', 3) if hasattr(self.flow_executor, 'underlying_flow') else 3

            experiment_config = {
                'benchmark_name': self.benchmark_name,
                'max_iterations': max_iterations,
                'models': {}
            }

            # Add arguments to config
            if hasattr(args, '__dict__'):
                for key, value in args.__dict__.items():
                    if not key.startswith('_') and value is not None:
                        experiment_config[key] = value

            # Extract model configurations used by the flow
            used_models = ['translator_api', 'reasoning_api']

            for name in used_models:
                if hasattr(config, 'llm') and name in config.llm:
                    llm_config = config.llm[name]
                    if hasattr(llm_config, 'model') and llm_config.model:
                        experiment_config['models'][name] = {
                            'model': llm_config.model,
                            'api_type': llm_config.api_type,
                            'base_url': llm_config.base_url if hasattr(llm_config, 'base_url') else None,
                            'temperature': llm_config.temperature if hasattr(llm_config, 'temperature') else None,
                            'max_tokens': llm_config.max_tokens if hasattr(llm_config, 'max_tokens') else None
                        }

            return experiment_config

        except Exception as e:
            print(f"⚠️ Could not gather full experiment config: {e}")
            # Fallback max_iterations
            fallback_max_iterations = 3
            try:
                fallback_max_iterations = getattr(self.flow_executor.underlying_flow, 'max_iterations', 3) if hasattr(self.flow_executor, 'underlying_flow') else 3
            except:
                pass
            return {
                'benchmark_name': self.benchmark_name,
                'max_iterations': fallback_max_iterations,
                'error': str(e)
            }

    # ==================== Abstract Methods (Benchmark-Specific) ====================

    @abstractmethod
    def build_flow_input(self, sample: Dict[str, Any]) -> str:
        """
        Convert benchmark sample to flow input format.

        Must be implemented by each benchmark.

        Args:
            sample: Sample data dictionary

        Returns:
            Formatted flow input string
        """
        pass

    @abstractmethod
    def extract_question(self, sample: Dict[str, Any]) -> str:
        """Extract question text from sample."""
        pass

    @abstractmethod
    def extract_options(self, sample: Dict[str, Any]) -> List[str]:
        """Extract answer options from sample (if any)."""
        pass

    @abstractmethod
    def extract_expected_answer(self, sample: Dict[str, Any]) -> str:
        """Extract expected answer from sample."""
        pass

    @abstractmethod
    def extract_question_id(self, sample: Dict[str, Any]) -> str:
        """Extract unique question ID from sample."""
        pass

    @abstractmethod
    def extract_metadata(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from sample for logging."""
        pass

    @abstractmethod
    def load_dataset(self, data_path: str, **kwargs) -> pd.DataFrame:
        """
        Load benchmark dataset.

        Args:
            data_path: Path to dataset file
            **kwargs: Additional benchmark-specific arguments

        Returns:
            DataFrame with standardized columns
        """
        pass

    # ==================== Utility Methods ====================

    def filter_dataset(self,
                      data: pd.DataFrame,
                      start_sample: Optional[int] = None,
                      end_sample: Optional[int] = None,
                      max_samples: Optional[int] = None) -> pd.DataFrame:
        """
        Filter dataset based on sample range or maximum count.

        Args:
            data: Input DataFrame
            start_sample: Starting sample index
            end_sample: Ending sample index (inclusive)
            max_samples: Maximum number of samples

        Returns:
            Filtered DataFrame
        """
        original_size = len(data)

        if start_sample is not None or end_sample is not None:
            start = start_sample or 0
            end = end_sample or len(data)
            # Make end_sample inclusive by adding 1 to the slice end
            if end_sample is not None:
                end = end_sample + 1
            data = data.iloc[start:end]
            # Print the actual inclusive range being processed
            actual_end = min(start + len(data) - 1, len(data) - 1) if len(data) > 0 else start
            print(f"Processing samples {start} to {actual_end} ({len(data)} samples)")
        elif max_samples and max_samples > 0:
            data = data.iloc[:max_samples]
            print(f"Limited to {max_samples} samples")
        else:
            print(f"Processing all {len(data)} samples")

        if len(data) != original_size:
            print(f"📊 Dataset filtered: {original_size} → {len(data)} samples")

        return data

    def save_session_info(self, output_file: Optional[str] = None):
        """Save session information to file for external reference."""
        if not self.session_dir:
            return

        session_info_file = Path("session_info.txt")
        with session_info_file.open('w') as f:
            f.write(f"SESSION_DIR={self.session_dir}\n")
            f.write(f"BENCHMARK={self.benchmark_name}\n")
            if output_file:
                f.write(f"OUTPUT_FILE={output_file}\n")

    def print_execution_summary(self,
                              sample_idx: int,
                              total_samples: int,
                              response: str,
                              execution_time: float,
                              sample: Optional[Dict[str, Any]] = None):
        """Print detailed execution summary."""
        print("🎯 Flow Execution Result:")
        print("="*70)
        print(response)
        print("="*70)
        print(f"📊 Question {sample_idx+1}/{total_samples} | Time: {execution_time:.2f}s")

        # Add benchmark-specific info
        if sample:
            self.print_sample_info(sample)

        print('-' * 70)

    def print_sample_info(self, sample: Dict[str, Any]):
        """Print benchmark-specific sample information. Override in subclasses."""
        pass