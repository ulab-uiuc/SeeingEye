import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from app.config import PROJECT_ROOT


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy and pandas data types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict()
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        return super().default(obj)


class LogMessage:
    """Represents a single log message with timestamp and metadata"""

    def __init__(
        self,
        role: str,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.timestamp = datetime.now()
        self.role = role
        self.content = content
        self.tool_calls = tool_calls or []
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        message_dict = {
            "timestamp": self.timestamp.isoformat(),
            "role": self.role,
            "content": self.content
        }

        if self.tool_calls:
            # Clean base64 data from tool_calls
            cleaned_tool_calls = []
            for tool_call in self.tool_calls:
                cleaned_call = tool_call.copy()
                if isinstance(cleaned_call, dict) and 'base64_image' in cleaned_call:
                    cleaned_call['base64_image'] = "[IMAGE_DATA_EXCLUDED]"
                cleaned_tool_calls.append(cleaned_call)
            message_dict["tool_calls"] = cleaned_tool_calls
        if self.metadata:
            # Clean base64 data from metadata
            cleaned_metadata = self._clean_base64_from_dict(self.metadata)
            message_dict["metadata"] = cleaned_metadata

        return message_dict

    def _clean_base64_from_dict(self, data: Any) -> Any:
        """Recursively clean base64_image fields from nested dictionaries"""
        if isinstance(data, dict):
            cleaned = {}
            for key, value in data.items():
                if key == 'base64_image':
                    cleaned[key] = "[IMAGE_DATA_EXCLUDED]"
                else:
                    cleaned[key] = self._clean_base64_from_dict(value)
            return cleaned
        elif isinstance(data, list):
            return [self._clean_base64_from_dict(item) for item in data]
        else:
            return data


class QuestionLog:
    """Manages logging for a single question"""

    def __init__(
        self,
        question_id: str,
        question: str,
        options: Optional[List[str]] = None,
        expected_answer: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.question_id = question_id
        self.start_time = datetime.now()
        self.end_time = None
        self.question = question
        self.options = options or []
        self.expected_answer = expected_answer
        self.metadata = metadata or {}
        self.messages: List[LogMessage] = []
        self.model_response = None
        self.critical_errors = []  # Track critical errors that occurred during execution
        self.evaluation_results = {}
        self.token_usage = {}  # Per-question token usage

    def add_message(
        self,
        role: str,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a message to this question's log"""
        message = LogMessage(role, content, tool_calls, metadata)
        self.messages.append(message)

    def finish(
        self,
        model_response: Optional[str] = None,
        evaluation_results: Optional[Dict[str, Any]] = None,
        token_usage: Optional[Dict[str, Any]] = None,
        critical_errors: Optional[List[Dict[str, Any]]] = None
    ):
        """Finalize the question log"""
        self.end_time = datetime.now()
        self.model_response = model_response
        self.critical_errors = critical_errors or []
        self.evaluation_results = evaluation_results or {}
        self.token_usage = token_usage or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        duration = None
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()

        return {
            "question_id": self.question_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": duration,
            "raw_question": self.question,
            "options": self.options,
            "expected_answer": self.expected_answer,
            "model_response": self.model_response,
            "critical_errors": self.critical_errors,
            "evaluation_results": self.evaluation_results,
            "token_usage_this_question": self.token_usage,
            "metadata": self.metadata,
            "messages": [msg.to_dict() for msg in self.messages],
            "total_messages": len(self.messages)
        }


class SessionLog:
    """Manages logging for an entire session of multiple questions"""

    def __init__(
        self,
        session_id: str,
        benchmark_name: str,
        experiment_config: Optional[Dict[str, Any]] = None
    ):
        self.session_id = session_id
        self.benchmark_name = benchmark_name
        self.start_time = datetime.now()
        self.end_time = None
        self.experiment_config = experiment_config or {}
        self.questions: List[QuestionLog] = []
        self.current_question: Optional[QuestionLog] = None
        self.session_dir = PROJECT_ROOT / "logs" / benchmark_name / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def start_question(
        self,
        question: str,
        options: Optional[List[str]] = None,
        expected_answer: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        custom_question_id: Optional[str] = None
    ) -> str:
        """Start logging a new question"""
        if custom_question_id:
            question_id = custom_question_id
        else:
            question_number = len(self.questions) + 1
            question_id = f"question_{question_number}"

        self.current_question = QuestionLog(
            question_id=question_id,
            question=question,
            options=options,
            expected_answer=expected_answer,
            metadata=metadata
        )
        self.questions.append(self.current_question)
        return question_id

    def add_message(
        self,
        role: str,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a message to the current question"""
        if not self.current_question:
            raise RuntimeError("No active question to add message to")

        self.current_question.add_message(role, content, tool_calls, metadata)

    def finish_question(
        self,
        model_response: Optional[str] = None,
        evaluation_results: Optional[Dict[str, Any]] = None,
        token_usage: Optional[Dict[str, Any]] = None,
        critical_errors: Optional[List[Dict[str, Any]]] = None
    ):
        """Finish the current question and save it"""
        if not self.current_question:
            raise RuntimeError("No current question to finish")

        self.current_question.finish(model_response, evaluation_results, token_usage, critical_errors)

        # Save individual question file
        question_file = self.session_dir / f"{self.current_question.question_id}.json"
        with question_file.open('w', encoding='utf-8') as f:
            json.dump(self.current_question.to_dict(), f, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder)

        print(f"📝 Question {len(self.questions)} log saved to: {question_file}")
        self.current_question = None

    def finish_session(self):
        """Finish the session and save metadata"""
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()

        # Calculate total token usage across all questions
        total_token_usage = {
            "total_input_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "question_count": len(self.questions),
            "per_question_breakdown": []
        }

        for question in self.questions:
            if question.token_usage:
                total_token_usage["total_input_tokens"] += question.token_usage.get("total_input_tokens", 0)
                total_token_usage["total_completion_tokens"] += question.token_usage.get("total_completion_tokens", 0)
                total_token_usage["per_question_breakdown"].append({
                    "question_id": question.question_id,
                    "token_usage": question.token_usage
                })

        total_token_usage["total_tokens"] = total_token_usage["total_input_tokens"] + total_token_usage["total_completion_tokens"]

        # Session metadata
        session_metadata = {
            "session_id": self.session_id,
            "benchmark_name": self.benchmark_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": duration,
            "total_questions": len(self.questions),
            "experiment_config": self.experiment_config,
            "total_token_usage": total_token_usage
        }

        # Save session metadata
        session_file = self.session_dir / "session_metadata.json"
        with session_file.open('w', encoding='utf-8') as f:
            json.dump(session_metadata, f, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder)

        print(f"📝 Session metadata saved to: {session_file}")
        print(f"📊 Session completed: {len(self.questions)} questions in {duration:.1f}s")


class LogSave:
    """
    Simplified, modular logging system for multi-agent conversations.

    Supports only the individual question logging paradigm for clarity and modularity.
    """

    def __init__(self, benchmark_name: str = "default", log_subdir: str = ""):
        self.benchmark_name = benchmark_name
        self.log_subdir = log_subdir
        self.current_session: Optional[SessionLog] = None

        # Create base logs directory
        if log_subdir:
            self.base_log_dir = PROJECT_ROOT / "logs" / benchmark_name / log_subdir
        else:
            self.base_log_dir = PROJECT_ROOT / "logs" / benchmark_name
        self.base_log_dir.mkdir(parents=True, exist_ok=True)

    def start_question_session(
        self,
        session_id: Optional[str] = None,
        experiment_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start a new question session"""
        if not session_id:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}"

        self.current_session = SessionLog(
            session_id=session_id,
            benchmark_name=self.benchmark_name,
            experiment_config=experiment_config
        )
        return session_id

    def start_individual_question(
        self,
        question: str,
        options: Optional[List[str]] = None,
        expected_answer: Optional[str] = None,
        question_metadata: Optional[Dict[str, Any]] = None,
        custom_question_id: Optional[str] = None
    ) -> str:
        """Start logging for an individual question"""
        if not self.current_session:
            raise RuntimeError("Must call start_question_session() first")

        return self.current_session.start_question(
            question=question,
            options=options,
            expected_answer=expected_answer,
            metadata=question_metadata,
            custom_question_id=custom_question_id
        )

    def add_individual_question_message(
        self,
        role: str,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a message to the current question"""
        if not self.current_session:
            raise RuntimeError("No active session")

        self.current_session.add_message(role, content, tool_calls, metadata)

    def finish_individual_question(
        self,
        model_response: Optional[str] = None,
        evaluation_results: Optional[Dict[str, Any]] = None,
        token_usage: Optional[Dict[str, Any]] = None,
        critical_errors: Optional[List[Dict[str, Any]]] = None
    ):
        """Finish the current question"""
        if not self.current_session:
            raise RuntimeError("No active session")

        self.current_session.finish_question(model_response, evaluation_results, token_usage, critical_errors)

    def finish_question_session(self):
        """Finish the current session"""
        if not self.current_session:
            return

        self.current_session.finish_session()
        self.current_session = None

    @property
    def current_question(self) -> Optional[QuestionLog]:
        """Get the current question for compatibility with agent code"""
        if self.current_session:
            return self.current_session.current_question
        return None

    def get_log_files(self) -> List[str]:
        """Get list of session directories"""
        if not self.base_log_dir.exists():
            return []

        return [str(d) for d in self.base_log_dir.iterdir() if d.is_dir()]

    def load_session_metadata(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session metadata"""
        session_file = self.base_log_dir / session_id / "session_metadata.json"
        if not session_file.exists():
            return None

        with session_file.open('r', encoding='utf-8') as f:
            return json.load(f)

    def load_question(self, session_id: str, question_id: str) -> Optional[Dict[str, Any]]:
        """Load individual question data"""
        question_file = self.base_log_dir / session_id / f"{question_id}.json"
        if not question_file.exists():
            return None

        with question_file.open('r', encoding='utf-8') as f:
            return json.load(f)


def reset_flow_token_counters(flow_executor) -> None:
    """
    Reset token counters for all agents in a flow executor.

    Call this before starting a new question to get per-question token usage.

    Args:
        flow_executor: Flow executor instance with access to agents
    """
    try:
        # Get the underlying flow which has access to agents
        flow = flow_executor.underlying_flow

        # Reset token counters for all agents
        if hasattr(flow, 'agents') and flow.agents:
            for agent_name, agent in flow.agents.items():
                if hasattr(agent, 'llm') and hasattr(agent.llm, 'token_tracker'):
                    agent.llm.token_tracker.reset()
                    print(f"🔄 Reset token counter for {agent_name}")

    except Exception as e:
        print(f"Warning: Could not reset token counters: {e}")


def collect_flow_token_usage_detailed(flow_executor) -> Dict[str, Any]:
    """
    Collect per-question token usage from all agents in a flow executor.

    Args:
        flow_executor: Flow executor instance with access to agents

    Returns:
        Dictionary with per-question token usage
    """
    try:
        # Get per-question usage (current state of counters)
        return collect_flow_token_usage(flow_executor)

    except Exception as e:
        print(f"Warning: Could not collect token usage: {e}")
        return {"total_input_tokens": 0, "total_completion_tokens": 0, "total_tokens": 0, "agents": {}}


def collect_flow_token_usage(flow_executor) -> Dict[str, Any]:
    """
    Collect token usage statistics from all agents in a flow executor.

    This utility function leverages the existing token infrastructure:
    - app.token_counter.TokenTracker for tracking tokens
    - app.llm.LLMProvider.token_tracker for LLM-level tracking
    - app.agent.react.ReactAgent.get_token_usage() for agent-level collection

    Args:
        flow_executor: Flow executor instance with access to agents

    Returns:
        Dictionary with aggregated token usage statistics from all agents
    """
    try:
        # Get the underlying flow which has access to agents
        flow = flow_executor.underlying_flow
        token_usage = {
            "total_input_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "agents": {}
        }

        # Collect from all agents in the flow using existing get_token_usage method
        if hasattr(flow, 'agents') and flow.agents:
            for agent_name, agent in flow.agents.items():
                if hasattr(agent, 'get_token_usage'):
                    agent_usage = agent.get_token_usage()
                    if agent_usage:
                        token_usage["agents"][agent_name] = agent_usage
                        # Use the existing field names from TokenTracker.get_usage_summary()
                        token_usage["total_input_tokens"] += agent_usage.get("input_tokens", 0)
                        token_usage["total_completion_tokens"] += agent_usage.get("completion_tokens", 0)

        token_usage["total_tokens"] = token_usage["total_input_tokens"] + token_usage["total_completion_tokens"]
        return token_usage
    except Exception as e:
        print(f"Warning: Could not collect token usage: {e}")
        return {
            "total_input_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "agents": {},
            "error": str(e)
        }