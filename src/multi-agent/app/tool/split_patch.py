"""Ask to locate tool for generating patch-numbered images for LLM analysis."""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

from app.tool.base import BaseTool, ToolResult


class SplitPatch(BaseTool):
    """Tool for generating 4x4 patch grid overlay on images for LLM to select crop regions"""
    
    name: str = "split_patch"
    description: str = "**PRIORITY TOOL - USE FIRST**: Generate a 4x4 patch grid overlay on an image with numbered areas (0-15). This should be the first tool called when analyzing any image to create a grid-based reference system for detailed regional analysis."
    
    # Define tool parameters (JSON Schema format)
    parameters: dict = {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Path to the input image file"
            }
        },
        "required": ["image_path"]
    }
    
    async def execute(
        self, 
        image_path: str
    ) -> ToolResult:
        """Execute patch numbering and save to with_grid folder"""
        
        try:
            # Validate input image
            image_file = Path(image_path)
            if not image_file.exists():
                return ToolResult(error=f"Image file does not exist: {image_path}")
            
            # Read image
            image = cv2.imread(str(image_file))
            if image is None:
                return ToolResult(error=f"Could not load image: {image_path}")
            
            height, width = image.shape[:2]
            
            # Create overlay for patch grid
            overlay = image.copy()
            
            # Calculate patch dimensions
            patch_height = height // 4
            patch_width = width // 4
            
            # Colors for grid and text
            grid_color = (0, 255, 0)  # Green grid lines
            text_color = (255, 0, 0)  # Red text
            background_color = (255, 255, 255)  # White background for text
            
            # Draw grid lines
            for i in range(1, 4):
                # Vertical lines
                cv2.line(overlay, (i * patch_width, 0), (i * patch_width, height), grid_color, 3)
                # Horizontal lines  
                cv2.line(overlay, (0, i * patch_height), (width, i * patch_height), grid_color, 3)
            
            # Add patch numbers (0-15)
            patch_id = 0
            for row in range(4):
                for col in range(4):
                    # Calculate center position for text
                    center_x = col * patch_width + patch_width // 2
                    center_y = row * patch_height + patch_height // 2
                    
                    # Text properties
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 2.0
                    thickness = 3
                    
                    # Get text size for background rectangle
                    text = str(patch_id)
                    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                    
                    # Draw text background rectangle (semi-transparent)
                    bg_x1 = center_x - text_width // 2 - 10
                    bg_y1 = center_y - text_height // 2 - 10
                    bg_x2 = center_x + text_width // 2 + 10
                    bg_y2 = center_y + text_height // 2 + 10
                    
                    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), background_color, -1)
                    
                    # Draw text
                    text_x = center_x - text_width // 2
                    text_y = center_y + text_height // 2
                    cv2.putText(overlay, text, (text_x, text_y), font, font_scale, text_color, thickness)
                    
                    patch_id += 1
            
            # Blend original image with overlay (50% transparency for overlay)
            alpha = 0.5  # 50% transparency
            blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
            
            # Create with_grid folder in the same directory as the original image
            image_dir = image_file.parent
            with_grid_dir = image_dir / "with_grid"
            with_grid_dir.mkdir(exist_ok=True)
            
            # Generate output filename based on original image name
            original_name = image_file.stem
            output_filename = f"{original_name}_with_grid.png"
            output_path = with_grid_dir / output_filename
            
            # Save processed image
            success = cv2.imwrite(str(output_path), blended)
            if not success:
                return ToolResult(error=f"Failed to save processed image to: {output_path}")
            
            # Prepare result
            result_data = {
                "processed_image_path": str(output_path),
                "original_image_path": image_path,
                "grid_size": "4x4",
                "total_patches": 16,
                "image_dimensions": {
                    "width": width,
                    "height": height
                },
                "patch_dimensions": {
                    "width": patch_width,
                    "height": patch_height
                }
            }
            
            output_text = f"""Grid overlay completed:

Original Image: {image_path}
Grid Image: {output_path}
Grid: 4x4 patches (numbered 0-15)
Transparency: 50%

The image with numbered patches is ready for LLM analysis to select crop regions."""
            
            return ToolResult(output=output_text, data=result_data)
            
        except Exception as e:
            return ToolResult(error=f"Error in ask_to_locate processing: {str(e)}")
    
    
    def _get_patch_coordinates(self, patch_id: int, image_width: int, image_height: int) -> Dict[str, int]:
        """Get the coordinates of a specific patch"""
        
        if patch_id < 0 or patch_id > 15:
            raise ValueError("Patch ID must be between 0 and 15")
        
        patch_width = image_width // 4
        patch_height = image_height // 4
        
        # Convert patch_id (0-15) to row, col (0-3, 0-3)
        patch_index = patch_id
        row = patch_index // 4
        col = patch_index % 4
        
        return {
            "x_start": col * patch_width,
            "y_start": row * patch_height,
            "x_end": (col + 1) * patch_width if col < 3 else image_width,
            "y_end": (row + 1) * patch_height if row < 3 else image_height,
            "row": row,
            "col": col
        }
