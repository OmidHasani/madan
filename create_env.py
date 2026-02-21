"""Create .env file with configuration"""
import os
import sys

# Fix encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

env_content = """# OpenAI Configuration
OPENAI_API_KEY=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

# PDF File Path
PDF_FILE_PATH=shopmanual.pdf

# Application Settings
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4-turbo-preview
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5

# Server Settings
HOST=0.0.0.0
PORT=8000

# Vector Database
VECTOR_DB_PATH=vector_db
COLLECTION_NAME=shop_manual
"""

# Check if .env already exists
if os.path.exists('.env'):
    print("Warning: .env file already exists! Overwriting...")

# Write .env file
with open('.env', 'w', encoding='utf-8') as f:
    f.write(env_content)

print("SUCCESS: .env file created!")
print("\nSettings:")
print("   - OpenAI API Key: Configured")
print("   - PDF File Path: shopmanual.pdf")
print("\nNote: If your PDF has a different name, edit the .env file.")

