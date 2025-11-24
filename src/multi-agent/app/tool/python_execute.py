import multiprocessing
import sys
from io import StringIO
from typing import Dict

from app.tool.base import BaseTool, ToolResult


class PythonExecute(BaseTool):
    """A tool for executing Python code with timeout and safety restrictions."""

    name: str = "python_execute"
    description: str = "Executes Python code string. IMPORTANT: Always add print() statements in your code to display results, calculations, variable values, and function outputs that you want to see. Only printed content will be visible in the output."
    parameters: dict = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute. Remember to include print() statements for any values, results, or outputs you want to see displayed.",
            },
        },
        "required": ["code"],
    }

    def _run_code(self, code: str, result_dict: dict, safe_globals: dict) -> None:
        original_stdout = sys.stdout
        try:
            output_buffer = StringIO()
            sys.stdout = output_buffer
            
            # Add common variables that might be referenced in MMMU context
            # This helps avoid "name 'choices' is not defined" errors
            safe_globals.update({
                'choices': [],  # Default empty list for choices
                'options': [],  # Alternative name for choices
                'question': '',  # Default empty string for question
                'answer': '',    # Default empty string for answer
            })
            
            # Simply execute the code as-is, relying on user/AI to add print statements
            exec(code, safe_globals, safe_globals)
            
            result_dict["observation"] = output_buffer.getvalue()
            result_dict["success"] = True
        except Exception as e:
            result_dict["observation"] = str(e)
            result_dict["success"] = False
        finally:
            sys.stdout = original_stdout

    async def execute(
        self,
        code: str,
        timeout: int = 5,
    ) -> ToolResult:
        """
        Executes the provided Python code with a timeout.

        Args:
            code (str): The Python code to execute.
            timeout (int): Execution timeout in seconds.

        Returns:
            ToolResult: Contains execution output or error message.
        """

        with multiprocessing.Manager() as manager:
            result = manager.dict({"observation": "", "success": False})
            if isinstance(__builtins__, dict):
                safe_globals = {"__builtins__": __builtins__}
            else:
                safe_globals = {"__builtins__": __builtins__.__dict__.copy()}
            proc = multiprocessing.Process(
                target=self._run_code, args=(code, result, safe_globals)
            )
            proc.start()
            proc.join(timeout)

            # timeout process
            if proc.is_alive():
                proc.terminate()
                proc.join(1)
                return ToolResult(error=f"Execution timeout after {timeout} seconds")
            
            # Convert result to ToolResult
            if result.get("success", False):
                observation = result.get("observation", "")
                if observation:
                    return ToolResult(output=observation)
                else:
                    return ToolResult(output="Code executed successfully with no output")
            else:
                error_msg = result.get("observation", "Unknown error occurred")
                return ToolResult(error=error_msg)
