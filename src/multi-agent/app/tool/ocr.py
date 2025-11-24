"""OCR tool for extracting text from images using Azure Computer Vision API."""

import os
import requests
from pathlib import Path
from typing import Optional

from app.tool.base import BaseTool, ToolResult


class OCR(BaseTool):
    """OCR tool for extracting text from images using Azure Computer Vision API"""
    
    name: str = "ocr"
    description: str = "Extract text content from image files, supports text recognition in multiple languages"
    
    # Define tool parameters (JSON Schema format)
    parameters: dict = {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Path to the image file (supports relative and absolute paths)"
            },
            "language": {
                "type": "string",
                "description": "Language code for recognition, e.g., 'zh-Hans' (Simplified Chinese), 'en' (English), 'unk' (auto-detect)",
                "default": "unk"
            },
            "detect_orientation": {
                "type": "boolean",
                "description": "Whether to detect image orientation",
                "default": True
            }
        },
        "required": ["image_path"]
    }
    
    # Azure Computer Vision API configuration
    subscription_key: str = ""
    endpoint: str = ""
    ocr_url: str = endpoint + "vision/v2.1/ocr"
    headers: dict = {
            "Ocp-Apim-Subscription-Key": subscription_key,
            "Content-Type": "application/octet-stream"
        }
    

    
    async def execute(self, image_path: str, language: str = "unk", detect_orientation: bool = True) -> ToolResult:
        """Execute OCR recognition"""
        try:
            # Check if image file exists
            image_file = Path(image_path)
            if not image_file.exists():
                return ToolResult(error=f"Image file does not exist: {image_path}")
            
            # Check if file is in supported image format
            allowed_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'}
            if image_file.suffix.lower() not in allowed_extensions:
                return ToolResult(error=f"Unsupported image format: {image_file.suffix}. Supported formats: {', '.join(allowed_extensions)}")
            
            # Set request parameters
            params = {
                "language": language,
                "detectOrientation": str(detect_orientation).lower()
            }
            
            # Read image file (binary mode)
            with open(image_file, "rb") as f:
                image_data = f.read()
            
            # Call Azure OCR API
            response = requests.post(
                self.ocr_url, 
                headers=self.headers, 
                params=params, 
                data=image_data,
                timeout=30
            )
            
            # Check response status
            if response.status_code != 200:
                return ToolResult(error=f"OCR API call failed: {response.status_code} - {response.text}")
            
            result = response.json()
            
            # Extract text content
            extracted_text = self._extract_text_from_result(result)
            
            # Format output as string for better compatibility
            language_detected = result.get("language", "unknown")
            text_angle = result.get("textAngle", 0)
            orientation = result.get("orientation", "NotDetected")
            regions_count = len(result.get("regions", []))
            
            output_text = f"""OCR Results:
Extracted Text:
{extracted_text}

"""
            
            return ToolResult(output=output_text)
            
        except FileNotFoundError:
            return ToolResult(error=f"File not found: {image_path}")
        except requests.exceptions.RequestException as e:
            return ToolResult(error=f"Network request error: {str(e)}")
        except Exception as e:
            return ToolResult(error=f"OCR processing error: {str(e)}")
    
    def _extract_text_from_result(self, result: dict) -> str:
        """Extract plain text from API result"""
        text_lines = []
        
        for region in result.get("regions", []):
            for line in region.get("lines", []):
                line_text = ""
                for word in line.get("words", []):
                    line_text += word.get("text", "") + " "
                if line_text.strip():
                    text_lines.append(line_text.strip())
        
        return "\n".join(text_lines)