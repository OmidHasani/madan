"""
Test script to verify the system is working correctly
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()


def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")
    
    try:
        from config import settings
        from pdf_processor import PDFProcessor
        from vector_store import VectorStore
        from rag_engine import RAGEngine
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False


def test_config():
    """Test configuration"""
    print("\nTesting configuration...")
    
    try:
        from config import settings
        
        if not settings.openai_api_key:
            print("✗ OpenAI API key not set")
            return False
        
        print(f"✓ API Key: {settings.openai_api_key[:20]}...")
        print(f"✓ PDF Path: {settings.pdf_file_path}")
        print(f"✓ Embedding Model: {settings.embedding_model}")
        print(f"✓ Chat Model: {settings.chat_model}")
        
        return True
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False


def test_pdf_file():
    """Test if PDF file exists"""
    print("\nTesting PDF file...")
    
    try:
        from config import settings
        
        if not os.path.exists(settings.pdf_file_path):
            print(f"✗ PDF file not found: {settings.pdf_file_path}")
            return False
        
        file_size = os.path.getsize(settings.pdf_file_path)
        print(f"✓ PDF file exists: {file_size / 1024 / 1024:.2f} MB")
        
        return True
    except Exception as e:
        print(f"✗ PDF file error: {e}")
        return False


def test_pdf_extraction():
    """Test PDF extraction (on first page only)"""
    print("\nTesting PDF extraction...")
    
    try:
        from config import settings
        from pdf_processor import PDFProcessor
        
        if not os.path.exists(settings.pdf_file_path):
            print("⊘ Skipping (PDF not found)")
            return True
        
        processor = PDFProcessor(settings.pdf_file_path)
        
        # Extract just first page for testing
        pages = processor.extract_text_pymupdf()
        
        if not pages:
            print("✗ No text extracted")
            return False
        
        first_page_text = pages[0][1] if pages else ""
        print(f"✓ Extracted {len(pages)} pages")
        print(f"✓ First page has {len(first_page_text)} characters")
        
        if len(first_page_text) > 50:
            print(f"✓ Sample: {first_page_text[:100]}...")
        
        return True
    except Exception as e:
        print(f"✗ PDF extraction error: {e}")
        return False


def test_openai_connection():
    """Test OpenAI API connection"""
    print("\nTesting OpenAI API connection...")
    
    try:
        from openai import OpenAI
        from config import settings
        
        client = OpenAI(api_key=settings.openai_api_key)
        
        # Test with a simple embedding
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input="test"
        )
        
        embedding = response.data[0].embedding
        print(f"✓ API connection successful")
        print(f"✓ Embedding dimension: {len(embedding)}")
        
        return True
    except Exception as e:
        print(f"✗ OpenAI API error: {e}")
        return False


def test_vector_db():
    """Test vector database initialization"""
    print("\nTesting vector database...")
    
    try:
        from config import settings
        from vector_store import VectorStore
        
        vector_store = VectorStore(
            api_key=settings.openai_api_key,
            db_path=settings.vector_db_path,
            collection_name=settings.collection_name + "_test",
            embedding_model=settings.embedding_model
        )
        
        stats = vector_store.get_collection_stats()
        print(f"✓ Vector DB initialized")
        print(f"✓ Collection: {stats['collection_name']}")
        print(f"✓ Document count: {stats['document_count']}")
        
        # Cleanup test collection
        vector_store.reset_collection()
        
        return True
    except Exception as e:
        print(f"✗ Vector DB error: {e}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("  PDF CHATBOT - System Test")
    print("="*60)
    print()
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("PDF File", test_pdf_file),
        ("PDF Extraction", test_pdf_extraction),
        ("OpenAI Connection", test_openai_connection),
        ("Vector Database", test_vector_db),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("  Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("  1. Run: python initialize_db.py")
        print("  2. Run: python run.py")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())



