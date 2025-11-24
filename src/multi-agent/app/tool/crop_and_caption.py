"""CropAndCaption tool for cropping selected patches and generating contextual captions."""

import cv2
import numpy as np
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.tool.base import BaseTool, ToolResult
from app.llm import LLM


class CropAndCaption(BaseTool):
    """Tool for cropping selected patches and generating detailed captions with full context"""
    
    name: str = "crop_and_caption"
    description: str = "Crop selected patch regions from original image and generate detailed captions with full context for translator agent"
    
    # Define tool parameters (JSON Schema format)
    parameters: dict = {
        "type": "object",
        "properties": {
            "original_image_path": {
                "type": "string",
                "description": "Path to the original image (without grid overlay)"
            },
            "patch_coordinates": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {
                        "type": "integer"
                    },
                    "minItems": 2,
                    "maxItems": 2
                },
                "description": "List of patch coordinate pairs [[top_left, bottom_right], [single, -1], ...] from locate tool"
            },
            "question": {
                "type": "string",
                "description": "The question that needs to be answered"
            },
            "context": {
                "type": "string",
                "description": "Previous analysis context from translator agent",
                "default": ""
            },
            "output_dir": {
                "type": "string",
                "description": "Directory to save cropped images",
                "default": "./cropped_regions"
            }
        },
        "required": ["original_image_path", "patch_coordinates", "question"]
    }
    
    async def execute(
        self, 
        original_image_path: str,
        patch_coordinates: List[List[int]],
        question: str,
        context: str = "",
        output_dir: str = "./cropped_regions"
    ) -> ToolResult:
        """Execute cropping and captioning with full context"""
        
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
            
            # Create output directory
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            
            # Process each patch coordinate
            cropped_regions = []
            captions = []
            
            for i, (top_left, bottom_right) in enumerate(patch_coordinates):
                # Calculate crop area
                crop_coords = self._calculate_crop_coordinates(
                    top_left, bottom_right, width, height
                )
                
                # Crop the region
                cropped_image = image[
                    crop_coords["y_start"]:crop_coords["y_end"],
                    crop_coords["x_start"]:crop_coords["x_end"]
                ]
                
                # Save cropped image
                crop_filename = f"cropped_region_{i}_{top_left}_{bottom_right}.png"
                crop_path = output_dir_path / crop_filename
                
                success = cv2.imwrite(str(crop_path), cropped_image)
                if not success:
                    return ToolResult(error=f"Failed to save cropped region {i}: {crop_path}")
                
                # Generate contextual caption for this crop
                caption = await self._generate_contextual_caption(
                    str(crop_path), question, context, i, top_left, bottom_right
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
            combined_analysis = self._combine_captions(captions, question, context)
            
            # Prepare result
            result_data = {
                "original_image_path": original_image_path,
                "question": question,
                "context": context,
                "patch_coordinates": patch_coordinates,
                "cropped_regions": cropped_regions,
                "individual_captions": captions,
                "combined_analysis": combined_analysis,
                "output_directory": str(output_dir_path)
            }
            
            output_text = f"""CropAndCaption completed:

Original Image: {original_image_path}
Question: {question}
Processed Regions: {len(cropped_regions)}

Cropped Images:
{chr(10).join([f"- {region['cropped_image_path']}" for region in cropped_regions])}

Combined Analysis:
{combined_analysis}

Ready for translator agent to update SIR with contextual analysis."""
            
            return ToolResult(output=output_text, data=result_data)
            
        except Exception as e:
            return ToolResult(error=f"Error in crop_and_caption processing: {str(e)}")
    
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
                question, context, region_id, top_left, bottom_right
            )
            
            # Prepare LLM request
            llm = LLM()
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": caption_prompt
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
            
            # Generate caption
            caption = await llm.ask_with_images(
                messages=messages,
                images=[],
                temperature=0.1
            )
            
            return caption
            
        except Exception as e:
            return f"Caption generation failed: {str(e)}"
    
    def _create_caption_prompt(
        self, 
        question: str, 
        context: str, 
        region_id: int,
        top_left: int,
        bottom_right: int
    ) -> str:
        """Create contextual caption prompt for translator agent"""
        
        patch_description = f"patch {top_left}" if bottom_right == -1 else f"patches {top_left} to {bottom_right}"
        context_section = f"\n\nPrevious Analysis Context:\n{context}" if context else ""
        
        prompt = f"""You are analyzing a cropped region from an image to help answer a specific question. This analysis will be used by a translator agent to update the Structured Intermediate Representations (SIR).

Question: {question}{context_section}

Cropped Region: {patch_description} (Region {region_id})

Please provide a detailed analysis of this cropped region focusing on:

1. **Visual Content**: Describe all visible elements, objects, text, numbers, charts, diagrams, etc.
2. **Text Content**: Extract and transcribe any text, labels, numbers, or values exactly as they appear
3. **Structural Elements**: Describe layouts, relationships, connections between elements
4. **Question Relevance**: Explain how the content in this region relates to answering the question
5. **Contextual Integration**: How this region fits with the overall image context

Format your response as:
**Visual Description:**
[Detailed description of what you see]

**Text/Numbers Extracted:**
[Exact transcription of any text or numerical values]

**Structural Analysis:**
[Layout, relationships, organization of elements]

**Relevance to Question:**
[How this content helps answer the question]

**Integration Notes:**
[How this relates to the broader image context]

Provide precise, detailed information that will help the translator agent create an accurate SIR representation."""
        
        return prompt
    
    def _combine_captions(self, captions: List[str], question: str, context: str) -> str:
        """Combine individual captions with region labels"""
        
        combined = ""
        
        for i, caption in enumerate(captions):
            combined += f"REGION {i}: {caption}\n\n"
        
        return combined.strip()
