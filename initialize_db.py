"""
Standalone script to initialize the vector database
Run this before starting the API server
"""
import os
import sys
import logging
from dotenv import load_dotenv

from config import settings
from pdf_processor import PDFProcessor
from vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main initialization function"""
    
    # Get script directory (where .env file should be)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load environment variables from script directory
    env_path = os.path.join(script_dir, '.env')
    load_dotenv(env_path)
    
    # Change to script directory to ensure relative paths work
    os.chdir(script_dir)
    
    logger.info("="*60)
    logger.info("PDF Chatbot - Database Initialization")
    logger.info("="*60)
    
    # Validate API key
    if not settings.openai_api_key:
        logger.error("OpenAI API key not found! Please set OPENAI_API_KEY in .env file")
        sys.exit(1)
    
    # Resolve PDF path - try relative to script dir, then absolute
    pdf_path = settings.pdf_file_path
    if not os.path.isabs(pdf_path):
        # Try relative to script directory
        pdf_path = os.path.join(script_dir, pdf_path)
    
    # Validate PDF file
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        logger.error(f"Current directory: {os.getcwd()}")
        logger.error(f"Script directory: {script_dir}")
        logger.error(f"Looking for: {settings.pdf_file_path}")
        sys.exit(1)
    
    # Update settings with resolved path
    settings.pdf_file_path = pdf_path
    
    logger.info(f"PDF File: {pdf_path}")
    logger.info(f"Vector DB Path: {settings.vector_db_path}")
    logger.info(f"Collection Name: {settings.collection_name}")
    logger.info(f"Chunk Size: {settings.chunk_size}")
    logger.info(f"Chunk Overlap: {settings.chunk_overlap}")
    logger.info("")
    
    try:
        # Step 1: Initialize vector store
        logger.info("Step 1: Initializing vector store...")
        vector_store = VectorStore(
            api_key=settings.openai_api_key,
            db_path=settings.vector_db_path,
            collection_name=settings.collection_name,
            embedding_model=settings.embedding_model
        )
        
        # Check if already initialized
        stats = vector_store.get_collection_stats()
        if stats['document_count'] > 0:
            logger.warning(f"Collection already contains {stats['document_count']} documents!")
            
            # Check if FORCE_REPROCESS environment variable is set
            force_reprocess = os.getenv('FORCE_REPROCESS', 'no').lower() in ['yes', 'true', '1']
            
            if force_reprocess:
                logger.info("FORCE_REPROCESS is set, resetting collection...")
                vector_store.reset_collection()
                logger.info("Collection reset.")
            else:
                # Check if running in non-interactive mode (like Render)
                if not sys.stdin.isatty():
                    logger.info("Running in non-interactive mode. Using existing collection.")
                    sys.exit(0)
                
                response = input("Do you want to reset and reprocess? (yes/no): ")
                
                if response.lower() in ['yes', 'y']:
                    vector_store.reset_collection()
                    logger.info("Collection reset.")
                else:
                    logger.info("Using existing collection. Exiting.")
                    sys.exit(0)
        
        # Step 2: Process PDF
        logger.info("\nStep 2: Processing PDF document...")
        processor = PDFProcessor(pdf_path)
        
        chunks = processor.process(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        
        logger.info(f"Created {len(chunks)} chunks from PDF")

        # Step 2: Add to vector store
        logger.info("\nStep 2: Creating embeddings and storing in database...")
        logger.info("This may take a while depending on document size...")
        
        vector_store.add_documents(chunks, batch_size=50)
        
        # Step 3: Verify
        logger.info("\nStep 3: Verifying database...")
        final_stats = vector_store.get_collection_stats()
        
        logger.info("")
        logger.info("="*60)
        logger.info("Initialization Complete!")
        logger.info("="*60)
        logger.info(f"Collection: {final_stats['collection_name']}")
        logger.info(f"Total Documents: {final_stats['document_count']}")
        logger.info(f"Embedding Model: {final_stats['embedding_model']}")
        logger.info(f"Database Path: {final_stats['db_path']}")
        logger.info("")
        logger.info("You can now start the API server with: python api.py")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

