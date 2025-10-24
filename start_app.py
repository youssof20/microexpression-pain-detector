"""
Startup script for the Microexpression Pain Detector.
This script starts the Streamlit application on port 8502.
"""

import subprocess
import sys
import webbrowser
import time

def start_application():
    """Start the Streamlit application."""
    print("Starting Microexpression Pain Detector...")
    print("GitHub: https://github.com/youssof20/microexpression-pain-detector")
    print("Application will be available at: http://localhost:8502")
    print("Press Ctrl+C to stop the application")
    
    try:
        # Start Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py", 
            "--server.port", "8502",
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"Error starting application: {e}")

if __name__ == "__main__":
    start_application()
