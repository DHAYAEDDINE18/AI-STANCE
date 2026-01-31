import subprocess
import sys
import os

def install_requirements():
    """
    Installs requirements from requirements.txt using pip.
    """
    req_file = "requirements.txt"
    
    if not os.path.exists(req_file):
        print(f"Error: '{req_file}' not found in the current directory.")
        return

    print(f"--- Starting Installation using {sys.executable} ---")
    print(f"Reading from {req_file}...\n")

    try:
        # Using sys.executable -m pip ensures we install to the current environment
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
        
        print("\n" + "="*40)
        print("SUCCESS: All requirements installed successfully!")
        print("="*40)
    except subprocess.CalledProcessError as e:
        print("\n" + "!"*40)
        print(f"ERROR: Installation failed with exit code {e.returncode}")
        print("Please check your internet connection or permissions.")
        print("!"*40)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    install_requirements()
    input("\nPress Enter to close this window...")
