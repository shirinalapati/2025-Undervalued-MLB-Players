"""
WSGI entry point for PythonAnywhere deployment.
This file makes the Dash app accessible via WSGI.
"""

import sys
import os

# Add the project directory to the path
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

# Change to project directory
os.chdir(project_dir)

# Import the Dash app
from frontend.table_dashboard import app

# PythonAnywhere requires the app to be named 'application'
# Dash apps expose their Flask server via app.server
application = app.server

