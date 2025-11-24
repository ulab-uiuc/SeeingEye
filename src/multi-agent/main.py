import argparse
import asyncio
import base64
from pathlib import Path

from app.agent.translator import TranslatorAgent
from app.logger import logger


async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Translator agent with a prompt and image")
    parser.add_argument(
        "--prompt", type=str, required=False, help="Input prompt for the agent"
    )
    parser.add_argument(
        "--image", type=str, required=False, help="Path to image file for analysis"
    )
    args = parser.parse_args()

    # Create and initialize Translator agent
    agent = TranslatorAgent()
    try:
        # Use command line prompt if provided, otherwise ask for input
        prompt = args.prompt if args.prompt else input("Enter your prompt/question: ")
        if not prompt.strip():
            logger.warning("Empty prompt provided.")
            return

        # Handle image input
        image_path = args.image if args.image else input("Enter image path (optional, press Enter to skip): ")
        
        base64_image = None
        if image_path and image_path.strip():
            image_file = Path(image_path.strip())
            if image_file.exists():
                logger.info(f"Loading image from: {image_file}")
                with open(image_file, "rb") as f:
                    image_data = f.read()
                    base64_image = base64.b64encode(image_data).decode('utf-8')
            else:
                logger.warning(f"Image file not found: {image_file}")
                return

        logger.info("Processing your request with Translator agent...")
        
        # Add message with image to agent's memory
        if base64_image:
            agent.update_memory("user", prompt, base64_image=base64_image)
        else:
            agent.update_memory("user", prompt)
            
        # Run the agent
        result = await agent.run()
        
        logger.info("Request processing completed.")
        logger.info(f"Result: {result}")
        
    except KeyboardInterrupt:
        logger.warning("Operation interrupted.")
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
    finally:
        # Ensure agent resources are cleaned up before exiting
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
