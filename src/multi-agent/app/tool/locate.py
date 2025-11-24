"""Locate tool for selecting relevant patches from grid-overlay images."""

import base64
from pathlib import Path
from typing import List, Tuple, Optional

from app.tool.base import BaseTool, ToolResult
from app.llm import LLM


class Locate(BaseTool):
    """Tool for selecting patches most relevant to the question from grid-overlay images"""
    
    name: str = "locate"
    description: str = "Select the most relevant patches from a grid-overlay image based on the question. Returns patch coordinates for cropping."
    
    # Define tool parameters (JSON Schema format)
    parameters: dict = {
        "type": "object",
        "properties": {
            "grid_image_path": {
                "type": "string",
                "description": "Path to the image with 4x4 grid overlay (numbered 0-15)"
            },
            "question": {
                "type": "string",
                "description": "The question that needs to be answered"
            },
            "context": {
                "type": "string",
                "description": "Additional context or previous analysis",
                "default": ""
            }
        },
        "required": ["grid_image_path", "question"]
    }
    
    async def execute(
        self, 
        grid_image_path: str,
        question: str,
        context: str = ""
    ) -> ToolResult:
        """Execute patch selection and return coordinates list"""
        
        try:
            # Validate input image
            grid_image_file = Path(grid_image_path)
            if not grid_image_file.exists():
                return ToolResult(error=f"Grid image file does not exist: {grid_image_path}")
            
            # Convert image to base64 for LLM
            try:
                with open(grid_image_file, "rb") as f:
                    image_data = f.read()
                image_b64 = base64.b64encode(image_data).decode('utf-8')
            except Exception as e:
                return ToolResult(error=f"Failed to read grid image: {str(e)}")
            
            # Create LLM prompt for patch selection
            selection_prompt = self._create_selection_prompt(question, context)
            
            # Prepare LLM request
            llm = LLM()
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": selection_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ]
            
            # Call LLM to select patches
            try:
                response = await llm.ask_with_images(
                    messages=messages,
                    images=[],  # Image already included in messages
                    temperature=0.1
                )
                
                # Parse the response to extract patch coordinates
                patch_coordinates = self._parse_patch_selection(response)
                
                # Prepare result
                result_data = {
                    "grid_image_path": grid_image_path,
                    "question": question,
                    "context": context,
                    "selected_patches": patch_coordinates,
                    "llm_response": response
                }
                
                output_text = f"""Patch selection completed:

Grid Image: {grid_image_path}
Question: {question}
Selected Regions: {patch_coordinates}

Format: [[top_left, bottom_right], [single_patch, -1], ...]
Ready for CropAndCaption processing."""
                
                return ToolResult(output=output_text, data=result_data)
                
            except Exception as e:
                return ToolResult(error=f"LLM processing error: {str(e)}")
            
        except Exception as e:
            return ToolResult(error=f"Error in locate processing: {str(e)}")
    
    def _create_selection_prompt(self, question: str, context: str) -> str:
        """Create the patch selection prompt for LLM"""
        
        context_section = f"\n\nPrevious Context:\n{context}" if context else ""
        
        prompt = f"""You need to select the most relevant patches from this 4x4 grid-overlay image to answer the given question.

Question: {question}{context_section}

The image shows a 4x4 grid with patches numbered 0-15:
- Row 1: Patches 0, 1, 2, 3 (top row)
- Row 2: Patches 4, 5, 6, 7
- Row 3: Patches 8, 9, 10, 11
- Row 4: Patches 12, 13, 14, 15 (bottom row)

Your task:
1. Analyze which patch(es) contain information most relevant to answering the question
2. You can select:
   - Single patches: [patch_number, -1]
   - Rectangular regions: [top_left_patch, bottom_right_patch]
   - Multiple separate regions

Instructions:
- Select the MINIMUM number of patches that contain the relevant information
- Prioritize quality over quantity - better to crop precisely than include noise
- Consider text, charts, diagrams, numbers, or other visual elements needed for the question
- If the whole image is needed, you can select the entire grid: [0, 15]

Format your response EXACTLY as a Python list:
[[top_left1, bottom_right1], [top_left2, bottom_right2], [single_patch, -1]]

Examples:
- Single patch 5: [[5, -1]]
- Rectangle from patch 1 to 6: [[1, 6]]
- Two separate regions: [[2, 6], [10, -1]]
- Whole image: [[0, 15]]

Your selection:"""
        
        return prompt
    
    def _parse_patch_selection(self, llm_response: str) -> List[List[int]]:
        """Parse LLM response to extract patch coordinates"""
        
        try:
            # Look for list pattern in the response
            import re
            import ast
            
            # Find the list pattern in the response
            list_pattern = r'\[\s*\[.*?\]\s*\]'
            matches = re.findall(list_pattern, llm_response, re.DOTALL)
            
            if matches:
                # Try to parse the first match
                list_str = matches[0]
                try:
                    parsed_list = ast.literal_eval(list_str)
                    
                    # Validate the format
                    if isinstance(parsed_list, list):
                        validated_list = []
                        for item in parsed_list:
                            if isinstance(item, list) and len(item) == 2:
                                top_left, bottom_right = item
                                # Validate patch numbers
                                if (0 <= top_left <= 15 and 
                                    (bottom_right == -1 or (0 <= bottom_right <= 15))):
                                    validated_list.append([top_left, bottom_right])
                        
                        if validated_list:
                            return validated_list
                except:
                    pass
            
            # Fallback: try to find individual numbers
            numbers = re.findall(r'\d+', llm_response)
            if numbers:
                # Try to group them in pairs
                patch_numbers = [int(n) for n in numbers if 0 <= int(n) <= 15]
                if patch_numbers:
                    # If we have at least one valid patch, create a selection
                    if len(patch_numbers) >= 2:
                        return [[patch_numbers[0], patch_numbers[1]]]
                    else:
                        return [[patch_numbers[0], -1]]
            
            # Ultimate fallback: select center patches
            return [[5, 10]]  # Center region as fallback
            
        except Exception as e:
            # Error fallback: select center patches
            return [[5, 10]]
    
    def _validate_patch_coordinates(self, coordinates: List[List[int]]) -> bool:
        """Validate that patch coordinates are in correct format"""
        
        if not isinstance(coordinates, list):
            return False
        
        for coord_pair in coordinates:
            if not isinstance(coord_pair, list) or len(coord_pair) != 2:
                return False
            
            top_left, bottom_right = coord_pair
            
            # Validate individual patch numbers
            if not (0 <= top_left <= 15):
                return False
            
            if bottom_right != -1 and not (0 <= bottom_right <= 15):
                return False
            
            # For rectangular regions, validate that top_left <= bottom_right
            if bottom_right != -1:
                top_left_row, top_left_col = top_left // 4, top_left % 4
                bottom_right_row, bottom_right_col = bottom_right // 4, bottom_right % 4
                
                if (top_left_row > bottom_right_row or 
                    top_left_col > bottom_right_col):
                    return False
        
        return True
