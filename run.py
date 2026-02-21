"""
Main script to run the PDF Chatbot application
"""
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Find project root (where .env file is)
def find_project_root():
    """Find project root directory (where .env file is located)"""
    current = Path(__file__).parent.absolute()
    
    # Look for .env file in current directory or parent directories
    for path in [current] + list(current.parents):
        if (path / '.env').exists():
            return path
    
    # If .env not found, use current directory
    return current

PROJECT_ROOT = find_project_root()

# Change to project root
os.chdir(PROJECT_ROOT)

# Load environment variables from project root
load_dotenv(PROJECT_ROOT / '.env')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_requirements():
    """Check if all requirements are met"""
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        logger.warning(".env file not found! Creating from .env.example...")
        if os.path.exists('.env.example'):
            import shutil
            shutil.copy('.env.example', '.env')
            logger.info("Please edit .env file with your OpenAI API key")
            return False
        else:
            logger.error("Neither .env nor .env.example found!")
            return False
    
    # Check OpenAI API key
    from config import settings
    if not settings.openai_api_key:
        logger.error("OpenAI API key not set in .env file!")
        return False
    
    # Check PDF file
    if not os.path.exists(settings.pdf_file_path):
        logger.error(f"PDF file not found: {settings.pdf_file_path}")
        logger.info("Please place your PDF file in the project directory")
        return False
    
    return True


def main():
    """Main entry point"""
    
    print("="*70)
    print("  PDF CHATBOT - Intelligent Document Q&A System")
    print("="*70)
    print()
    
    # Check requirements
    if not check_requirements():
        logger.error("Requirements check failed. Please fix the issues above.")
        sys.exit(1)
    
    logger.info("All requirements met!")
    logger.info("")
    
    # Check if database is initialized
    from config import settings
    db_initialized = os.path.exists(settings.vector_db_path) and \
                    os.listdir(settings.vector_db_path)
    
    if not db_initialized:
        logger.info("Vector database not found.")
        logger.info("You have two options:")
        logger.info("  1. Run 'python initialize_db.py' to initialize the database first (recommended)")
        logger.info("  2. Use the web interface to initialize after starting the server")
        logger.info("")
        
        response = input("Start server anyway? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            logger.info("Please run 'python initialize_db.py' first")
            sys.exit(0)
    
    # Start server
    logger.info("")
    logger.info("Starting FastAPI server...")
    logger.info(f"Server will be available at: http://{settings.host}:{settings.port}")
    logger.info(f"API Documentation: http://{settings.host}:{settings.port}/docs")
    logger.info("")
    logger.info("Press Ctrl+C to stop the server")
    logger.info("="*70)
    logger.info("")
    
    import uvicorn
    uvicorn.run(
        "api:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nServer stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

