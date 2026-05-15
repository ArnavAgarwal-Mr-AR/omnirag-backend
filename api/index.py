import os
import sys

# CRITICAL: Add parent directory to path BEFORE importing any project modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now imports from the root directory will work correctly
from main import app

# Vercel needs the app object to be named 'app'
# It is already imported as 'app' from main