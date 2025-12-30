#!/usr/bin/env python3
"""RAG MCP Server - Main entry point."""

import sys
import os
from pathlib import Path

# Add the app directory to the Python path
app_dir = Path(__file__).parent / "app"
sys.path.insert(0, str(app_dir))

from app.server import main

if __name__ == "__main__":
    main()
