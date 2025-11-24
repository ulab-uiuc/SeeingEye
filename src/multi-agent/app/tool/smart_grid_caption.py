"""Smart Grid Caption tool - combines split_patch + locate + crop_and_caption workflow."""

import cv2
import numpy as np
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.tool.base import BaseTool, ToolResult
from app.llm import LLM


class SmartGridCaption(BaseTool):
    """Combined tool that performs grid overlay, region selection, and cropping with caption generation"""
    
    name: str = "smart_grid_caption"
    description: str = "Intelligent image analysis tool that creates grid overlay, selects relevant regions via LLM, and generates detailed captions. If you do not find any relevant information in the image, please use this tool to analyze the image and generate a caption."
    
    # Define tool parameters (JSON Schema format)
    parameters: dict = {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Path to the input image file"
            },
            "question": {
                "type": "string",
                "description": "The question that needs to be answered"
            },
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of answer options (optional)"
            },
            "context": {
                "type": "string",
                "description": "Additional context or previous analysis",
                "default": ""
            },
            "output_dir": {
                "type": "string",
                "description": "Directory to save intermediate and final outputs",
                "default": "./smart_grid_output"
            }
        },
        "required": ["image_path", "question"]
    }
    
    async def execute(
        self, 
        image_path: str,
        question: str,
        options: Optional[List[str]] = None,
        context: str = "",
        output_dir: str = "./smart_grid_output"
    ) -> ToolResult:
        """Execute the complete smart grid caption workflow"""
        
        try:
            # Create output directory
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            
            # Format options for display
            options_text = ""
            if options:
                options_text = f"\nOptions: {options}"
            
            # Step 1: Generate grid overlay
            grid_result = await self._generate_grid_overlay(image_path)
            if grid_result.error:
                return ToolResult(error=f"Grid generation failed: {grid_result.error}")
            
            # Extract data from output (assuming it's a dict stored in output)
            grid_data = grid_result.output if isinstance(grid_result.output, dict) else {}
            grid_image_path = grid_data.get("processed_image_path", "")
            
            # Step 2: LLM selects relevant regions
            locate_result = await self._locate_relevant_regions(
                grid_image_path, question, options, context
            )
            if locate_result.error:
                return ToolResult(error=f"Region location failed: {locate_result.error}")
            
            # Extract data from output
            locate_data = locate_result.output if isinstance(locate_result.output, dict) else {}
            patch_coordinates = locate_data.get("selected_patches", [])
            
            # Step 3: Crop and generate captions
            crop_result = await self._crop_and_generate_captions(
                image_path, patch_coordinates, question, options, context, output_dir
            )
            if crop_result.error:
                return ToolResult(error=f"Cropping and captioning failed: {crop_result.error}")
            
            # Extract crop result data
            crop_data = crop_result.output if isinstance(crop_result.output, dict) else {}
            
            # Combine all results
            result_data = {
                "original_image_path": image_path,
                "question": question,
                "options": options,
                "context": context,
                "grid_image_path": grid_image_path,
                "selected_patches": patch_coordinates,
                "cropped_regions": crop_data.get("cropped_regions", []),
                "combined_analysis": crop_data.get("combined_analysis", ""),
                "workflow_steps": {
                    "1_grid_generation": grid_data,
                    "2_region_selection": locate_data,
                    "3_crop_and_caption": crop_data
                }
            }
            
            # Generate final output text
            output_text = f"""Smart Grid Caption Analysis Completed:

🔍 Question: {question}{options_text}

📊 Workflow Results:
1. ✅ Grid Overlay Generated: {grid_image_path}
2. ✅ Regions Selected: {patch_coordinates}
3. ✅ Crops Generated: {len(crop_data.get('cropped_regions', []))} regions

🖼️ Cropped Images:
{chr(10).join([f"- {region['cropped_image_path']}" for region in crop_data.get('cropped_regions', [])])}

📝 Combined Analysis:
{crop_data.get('combined_analysis', '')}

🎯 This analysis provides focused visual information relevant to answering the question."""
            
            return ToolResult(output=output_text)
            
        except Exception as e:
            return ToolResult(error=f"Error in smart_grid_caption workflow: {str(e)}")
    
    async def _generate_grid_overlay(self, image_path: str) -> ToolResult:
        """Step 1: Generate 4x4 grid overlay on the image"""
        
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
            grid_color = (255, 0, 0)  # Red grid lines (changed from green)
            text_color = (255, 255, 255)  # White text (better contrast with red grid)
            background_color = (0, 0, 0)  # Black background for text (better contrast)
            
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
                    
                    # Text properties - adaptive font size based on image dimensions
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    # Calculate font scale based on image size (smaller for larger images)
                    base_font_scale = min(width, height) / 600.0  # Normalize to reasonable size
                    font_scale = max(0.8, min(3.0, base_font_scale))  # Clamp between 0.8 and 3.0
                    thickness = max(2, int(font_scale * 1.5))  # Adaptive thickness
                    
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
            
            return ToolResult(output=result_data)
            
        except Exception as e:
            return ToolResult(error=f"Error in grid generation: {str(e)}")
    
    async def _locate_relevant_regions(
        self, 
        grid_image_path: str, 
        question: str, 
        options: Optional[List[str]], 
        context: str
    ) -> ToolResult:
        """Step 2: Use LLM to select most relevant patches"""
        
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
            selection_prompt = self._create_selection_prompt(question, options, context)
            
            # Prepare LLM request with vision-capable model
            llm = LLM()  # Use default OpenAI config (gpt-4o-mini) which supports vision
            
            # Call LLM to select patches
            response = await llm.ask_with_images(
                messages=[{
                    "role": "user",
                    "content": selection_prompt
                }],
                images=[f"data:image/png;base64,{image_b64}"],
                temperature=0.1
            )
            
            # Parse the response to extract patch coordinates
            patch_coordinates = self._parse_patch_selection(response)
            
            # Prepare result
            result_data = {
                "grid_image_path": grid_image_path,
                "question": question,
                "options": options,
                "context": context,
                "selected_patches": patch_coordinates,
                "llm_response": response
            }
            
            return ToolResult(output=result_data)
            
        except Exception as e:
            return ToolResult(error=f"Error in region location: {str(e)}")
    
    def _create_selection_prompt(
        self, 
        question: str, 
        options: Optional[List[str]], 
        context: str
    ) -> str:
        """Create the patch selection prompt for LLM"""
        
        options_section = ""
        if options:
            options_section = f"\nAnswer Options: {', '.join(options)}"
        
        context_section = f"\n\nPrevious Context:\n{context}" if context else ""
        
        prompt = f"""You need to select the most relevant patches from this 4x4 grid-overlay image to answer the given question.

Question: {question}{options_section}{context_section}

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
    
    async def _crop_and_generate_captions(
        self,
        original_image_path: str,
        patch_coordinates: List[List[int]],
        question: str,
        options: Optional[List[str]],
        context: str,
        output_dir: str
    ) -> ToolResult:
        """Step 3: Crop regions and generate detailed captions"""
        
        try:
            # Validate input image
            image_file = Path(original_image_path)
            if not image_file.exists():
                return ToolResult(error=f"Original image file does not exist: {original_image_path}")
            
            # Read original image
            image = cv2.imread(str(image_file))
            if image is None:
                return ToolResult(error=f"Could not load original image: {original_image_path}")
            
            height, width = image.shape[:2]
            
            # Create cropped_regions subdirectory
            crop_dir = Path(output_dir) / "cropped_regions"
            crop_dir.mkdir(parents=True, exist_ok=True)
            
            # Validate and fix patch coordinates
            valid_coordinates = self._validate_and_fix_coordinates(patch_coordinates)
            if not valid_coordinates:
                return ToolResult(error=f"All patch coordinates are invalid: {patch_coordinates}")
            
            # Process each patch coordinate
            cropped_regions = []
            captions = []
            
            for i, (top_left, bottom_right) in enumerate(valid_coordinates):
                # Calculate crop area
                crop_coords = self._calculate_crop_coordinates(
                    top_left, bottom_right, width, height
                )
                
                # Crop the region
                cropped_image = image[
                    crop_coords["y_start"]:crop_coords["y_end"],
                    crop_coords["x_start"]:crop_coords["x_end"]
                ]
                
                # Validate cropped image is not empty
                if cropped_image.size == 0:
                    return ToolResult(error=f"Cropped region {i} is empty. Invalid coordinates: [{top_left},{bottom_right}]. Crop coords: {crop_coords}")
                
                # Save cropped image
                crop_filename = f"cropped_region_{i}_{top_left}_{bottom_right}.png"
                crop_path = crop_dir / crop_filename
                
                success = cv2.imwrite(str(crop_path), cropped_image)
                if not success:
                    return ToolResult(error=f"Failed to save cropped region {i}: {crop_path}. Image shape: {cropped_image.shape}")
                
                # Generate contextual caption for this crop
                caption = await self._generate_contextual_caption(
                    str(crop_path), question, options, context, i, top_left, bottom_right
                )
                
                region_info = {
                    "region_id": i,
                    "patch_coordinates": [top_left, bottom_right],
                    "pixel_coordinates": crop_coords,
                    "cropped_image_path": str(crop_path),
                    "caption": caption
                }
                
                cropped_regions.append(region_info)
                captions.append(caption)
            
            # Combine all captions into comprehensive analysis
            combined_analysis = self._combine_captions(captions, question, options, context)
            
            # Prepare result
            result_data = {
                "original_image_path": original_image_path,
                "question": question,
                "options": options,
                "context": context,
                "patch_coordinates": patch_coordinates,
                "cropped_regions": cropped_regions,
                "individual_captions": captions,
                "combined_analysis": combined_analysis,
                "output_directory": str(crop_dir)
            }
            
            return ToolResult(output=result_data)
            
        except Exception as e:
            return ToolResult(error=f"Error in cropping and captioning: {str(e)}")
    
    def _calculate_crop_coordinates(self, top_left: int, bottom_right: int, width: int, height: int) -> Dict[str, int]:
        """Calculate pixel coordinates for cropping based on patch numbers"""
        
        patch_width = width // 4
        patch_height = height // 4
        
        if bottom_right == -1:
            # Single patch
            row = top_left // 4
            col = top_left % 4
            
            x_start = col * patch_width
            x_end = (col + 1) * patch_width if col < 3 else width
            y_start = row * patch_height
            y_end = (row + 1) * patch_height if row < 3 else height
        else:
            # Rectangular region
            top_left_row, top_left_col = top_left // 4, top_left % 4
            bottom_right_row, bottom_right_col = bottom_right // 4, bottom_right % 4
            
            x_start = top_left_col * patch_width
            x_end = (bottom_right_col + 1) * patch_width if bottom_right_col < 3 else width
            y_start = top_left_row * patch_height
            y_end = (bottom_right_row + 1) * patch_height if bottom_right_row < 3 else height
        
        return {
            "x_start": x_start,
            "x_end": x_end,
            "y_start": y_start,
            "y_end": y_end
        }
    
    async def _generate_contextual_caption(
        self, 
        crop_path: str, 
        question: str, 
        options: Optional[List[str]], 
        context: str, 
        region_id: int,
        top_left: int,
        bottom_right: int
    ) -> str:
        """Generate detailed caption with full context for the cropped region"""
        
        try:
            # Convert cropped image to base64
            with open(crop_path, "rb") as f:
                image_data = f.read()
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Create contextual caption prompt
            caption_prompt = self._create_caption_prompt(
                question, options, context, region_id, top_left, bottom_right
            )
            
            # Prepare LLM request with vision-capable model
            llm = LLM()  # Use default OpenAI config (gpt-4o-mini) which supports vision
            
            # Generate caption
            caption = await llm.ask_with_images(
                messages=[{
                    "role": "user", 
                    "content": caption_prompt
                }],
                images=[f"data:image/png;base64,{image_b64}"],
                temperature=0.1
            )
            
            return caption
            
        except Exception as e:
            return f"Caption generation failed: {str(e)}"
    
    def _create_caption_prompt(
        self, 
        question: str, 
        options: Optional[List[str]], 
        context: str, 
        region_id: int,
        top_left: int,
        bottom_right: int
    ) -> str:
        """Create contextual caption prompt for the cropped region"""
        
        patch_description = f"patch {top_left}" if bottom_right == -1 else f"patches {top_left} to {bottom_right}"
        
        options_section = ""
        if options:
            options_section = f"\nAnswer Options: {', '.join(options)}"
        
        context_section = f"\n\nPrevious Analysis Context:\n{context}" if context else ""
        
        prompt = f"""Analyze this image region to answer the question: {question}{options_section}{context_section}

This is region {region_id} containing {patch_description}.

Describe exactly what you see in this image:

**Visual Description:**
What objects, people, colors, and details are visible in this image?

**Text/Numbers Extracted:**
Any text, labels, numbers, or written content in the image (write "None visible" if no text).

**Structural Analysis:**
How are elements arranged and positioned in the image?

**Relevance to Question:**
How does what you see help answer: {question}

**Option Support:**
Which answer option does this image support and why?

Be direct and specific about what you observe in the image."""
        
        return prompt
    
    def _combine_captions(
        self, 
        captions: List[str], 
        question: str, 
        options: Optional[List[str]], 
        context: str
    ) -> str:
        """Combine individual captions into comprehensive analysis"""
        
        options_text = ""
        if options:
            options_text = f"\nAnswer Options: {', '.join(options)}"
        
        header = f"""COMPREHENSIVE VISUAL ANALYSIS

Question: {question}{options_text}

DETAILED REGION ANALYSIS:
"""
        
        combined = header
        
        for i, caption in enumerate(captions):
            combined += f"\n{'='*50}\nREGION {i}:\n{'='*50}\n{caption}\n"
        
        summary = f"""
You can now improve your caption based on region caption.
"""
        
        combined += summary
        
        return combined.strip()
    
    def _validate_and_fix_coordinates(self, patch_coordinates: List[List[int]]) -> List[List[int]]:
        """Validate and fix patch coordinates to ensure they form valid rectangles"""
        valid_coordinates = []
        
        for coord_pair in patch_coordinates:
            if not isinstance(coord_pair, list) or len(coord_pair) != 2:
                continue
            
            top_left, bottom_right = coord_pair
            
            # Validate individual patch numbers
            if not (0 <= top_left <= 15):
                continue
            
            # Handle single patch case
            if bottom_right == -1:
                valid_coordinates.append([top_left, -1])
                continue
            
            # Handle same patch case (should be single patch)
            if top_left == bottom_right:
                valid_coordinates.append([top_left, -1])
                continue
            
            if not (0 <= bottom_right <= 15):
                continue
            
            # For rectangular regions, validate that coordinates form a valid rectangle
            top_left_row, top_left_col = top_left // 4, top_left % 4
            bottom_right_row, bottom_right_col = bottom_right // 4, bottom_right % 4
            
            # Check if it forms a valid rectangle (top-left to bottom-right)
            if (top_left_row <= bottom_right_row and 
                top_left_col <= bottom_right_col):
                valid_coordinates.append([top_left, bottom_right])
            else:
                # Try to fix invalid rectangles by converting to single patches
                # Add both patches as individual crops
                valid_coordinates.append([top_left, -1])
                if top_left != bottom_right:
                    valid_coordinates.append([bottom_right, -1])
        
        return valid_coordinates
