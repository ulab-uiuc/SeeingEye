"""Table extraction tool for reading tables from images using img2table and Tesseract OCR."""

import os
import pandas as pd
from pathlib import Path
from typing import Optional, List
from img2table.document import Image
from img2table.ocr import TesseractOCR

from app.tool.base import BaseTool, ToolResult


class ReadTable(BaseTool):
    """Tool for extracting tables from images using img2table and Tesseract OCR"""
    
    name: str = "read_table"
    description: str = "Extract table data from image files and convert to structured format (CSV/Excel), supports both bordered and borderless tables"
    
    # Define tool parameters (JSON Schema format)
    parameters: dict = {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Path to the image file containing tables (supports relative and absolute paths)"
            },
            "language": {
                "type": "string",
                "description": "OCR language code: 'eng' (English), 'chi_sim' (Simplified Chinese), 'chi_tra' (Traditional Chinese), etc.",
                "default": "eng"
            },
            "export_xlsx": {
                "type": "string",
                "description": "Optional path to export all tables to Excel file (e.g., 'tables.xlsx')",
                "default": None
            },
            "borderless_tables": {
                "type": "boolean",
                "description": "Whether to detect borderless tables",
                "default": True
            },
            "implicit_rows": {
                "type": "boolean",
                "description": "Whether to infer implicit rows through alignment",
                "default": True
            }
        },
        "required": ["image_path"]
    }
    
    async def execute(
        self, 
        image_path: str, 
        language: str = "eng",
        export_xlsx: Optional[str] = None,
        borderless_tables: bool = True,
        implicit_rows: bool = True
    ) -> ToolResult:
        """Execute table extraction from image"""
        try:
            # Check if image file exists
            image_file = Path(image_path)
            if not image_file.exists():
                return ToolResult(error=f"Image file does not exist: {image_path}")
            
            # Check if file is in supported image format
            allowed_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.pdf'}
            if image_file.suffix.lower() not in allowed_extensions:
                return ToolResult(error=f"Unsupported image format: {image_file.suffix}. Supported formats: {', '.join(allowed_extensions)}")
            
            # Create document object
            doc = Image(str(image_file))
            
            # Initialize OCR engine
            try:
                ocr_engine = TesseractOCR(lang=language)
            except Exception as e:
                return ToolResult(error=f"Failed to initialize OCR engine with language '{language}': {str(e)}")
            
            # Extract tables
            tables = doc.extract_tables(
                ocr=ocr_engine,
                borderless_tables=borderless_tables,
                implicit_rows=implicit_rows
            )
            
            if not tables:
                return ToolResult(output="No tables detected in the image.")
            
            # Process tables and collect DataFrames
            table_results = []
            dfs = []
            
            for i, table in enumerate(tables, start=1):
                df = table.df  # Get pandas DataFrame
                dfs.append(df)
                
                # Format table info as text
                table_info = {
                    "table_number": i,
                    "rows": df.shape[0],
                    "columns": df.shape[1],
                    "data": self._format_table_as_text(df)
                }
                table_results.append(table_info)
            
            # Optional Excel export
            excel_export_info = ""
            if export_xlsx and dfs:
                try:
                    with pd.ExcelWriter(export_xlsx) as writer:
                        for i, df in enumerate(dfs, start=1):
                            sheet_name = f"Table_{i}"
                            df.to_excel(writer, sheet_name=sheet_name, index=False)
                    excel_export_info = f"All tables exported to Excel: {export_xlsx}"
                except Exception as e:
                    excel_export_info = f"Excel export failed: {str(e)}"
            
            # Format output
            output_text = f"""Table Extraction Results:
Found {len(tables)} table(s) in the image.

"""
            
            for table_info in table_results:
                output_text += f"""Table #{table_info['table_number']} ({table_info['rows']} rows × {table_info['columns']} columns):
{table_info['data']}

"""
            
            if excel_export_info:
                output_text += f"{excel_export_info}\n"
            
            return ToolResult(output=output_text.strip())
            
        except ImportError as e:
            return ToolResult(error=f"Missing required dependencies. Please install: pip install img2table tesseract. Error: {str(e)}")
        except Exception as e:
            return ToolResult(error=f"Table extraction error: {str(e)}")
    
    def _format_table_as_text(self, df: pd.DataFrame) -> str:
        """Format DataFrame as readable text table"""
        if df.empty:
            return "[Empty table]"
        
        # Convert DataFrame to a clean text format
        text_lines = []
        
        # Add column headers
        headers = [str(col) for col in df.columns]
        header_line = " | ".join(headers)
        text_lines.append(header_line)
        text_lines.append("-" * len(header_line))
        
        # Add data rows
        for _, row in df.iterrows():
            row_values = [str(val) if pd.notna(val) else "" for val in row]
            row_line = " | ".join(row_values)
            text_lines.append(row_line)
        
        return "\n".join(text_lines)
