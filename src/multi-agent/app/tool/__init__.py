"""
Tool module with lazy imports to improve startup time.

Tools are imported only when actually accessed, avoiding the overhead
of importing heavy dependencies (like boto3 for Bedrock) at module load time.
"""

# Always import BaseTool since it's lightweight and commonly needed
from app.tool.base import BaseTool

# Make ToolCollection also lazy to avoid importing logger at module load time
# from app.tool.tool_collection import ToolCollection

# Define what tools are available (for IDE autocomplete and documentation)
__all__ = [
    "BaseTool",
    "Bash",
    "Terminate",
    "TerminateAndAnswer",
    "Think",
    "StrReplaceEditor",
    "WebSearch",
    "ToolCollection",
    "CreateChatCompletion",
    "PlanningTool",
    "Crawl4aiTool",
    "SmartGridCaption",
    "ReadTable",
    "OCR",
    "CropAndCaption",
    "SplitPatch",
]

# Lazy import mapping - tools are imported only when accessed
_LAZY_IMPORTS = {
    "ToolCollection": "app.tool.tool_collection",
    "Bash": "app.tool.bash",
    "CreateChatCompletion": "app.tool.create_chat_completion",
    "PlanningTool": "app.tool.planning",
    "StrReplaceEditor": "app.tool.str_replace_editor",
    "Terminate": "app.tool.terminate",
    "TerminateAndAnswer": "app.tool.terminate_and_answer",
    "Think": "app.tool.think",
    "WebSearch": "app.tool.web_search",
    "Crawl4aiTool": "app.tool.crawl4ai",
    "SplitPatch": "app.tool.split_patch",
    "ReadTable": "app.tool.read_table",
    "CropAndCaption": "app.tool.crop_and_caption",
    "OCR": "app.tool.ocr",
    "SmartGridCaption": "app.tool.smart_grid_caption",
}


def __getattr__(name):
    """
    Lazy import tools only when accessed.

    This significantly speeds up module import time by deferring
    expensive imports (like boto3 via crop_and_caption -> llm -> bedrock)
    until the tool is actually used.
    """
    if name in _LAZY_IMPORTS:
        module_path = _LAZY_IMPORTS[name]
        try:
            # Import the module
            import importlib
            module = importlib.import_module(module_path)
            # Get the class with the same name as the key
            tool_class = getattr(module, name)
            # Cache it in globals for next access
            globals()[name] = tool_class
            return tool_class
        except ImportError as e:
            raise ImportError(f"Failed to import {name} from {module_path}: {e}")

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# For type checking and IDE support
if False:  # TYPE_CHECKING - this block never runs but helps IDEs
    from app.tool.tool_collection import ToolCollection
    from app.tool.bash import Bash
    from app.tool.create_chat_completion import CreateChatCompletion
    from app.tool.planning import PlanningTool
    from app.tool.str_replace_editor import StrReplaceEditor
    from app.tool.terminate import Terminate
    from app.tool.terminate_and_answer import TerminateAndAnswer
    from app.tool.think import Think
    from app.tool.web_search import WebSearch
    from app.tool.crawl4ai import Crawl4aiTool
    from app.tool.split_patch import SplitPatch
    from app.tool.read_table import ReadTable
    from app.tool.crop_and_caption import CropAndCaption
    from app.tool.ocr import OCR
    from app.tool.smart_grid_caption import SmartGridCaption
