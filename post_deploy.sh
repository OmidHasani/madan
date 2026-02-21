#!/bin/bash
# Post-deploy script for Render.com
# This script runs after the build to initialize the database

echo "Running post-deploy script..."

# Check if database already exists
if [ -d "vector_db" ] && [ "$(ls -A vector_db)" ]; then
    echo "Vector database already exists, skipping initialization..."
else
    echo "Initializing vector database..."
    python initialize_db.py || echo "Warning: Database initialization failed or already exists"
fi

echo "Post-deploy script completed!"
