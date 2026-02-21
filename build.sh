#!/bin/bash
# Build script for Render.com

set -e  # Exit on error

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Checking if database needs initialization..."
# Try to initialize database, but don't fail if it already exists
python initialize_db.py || echo "Database initialization skipped (may already exist)"

echo "Build completed!"
