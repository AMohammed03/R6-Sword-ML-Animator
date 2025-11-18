import os
import subprocess
import sys

def install_pyinstaller():
    try:
        import PyInstaller
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

def build_executable():
    cmd = [
        "pyinstaller",
        "--onefile",  # Single executable file
        "--name=R6-Sword-Animator",  # Output name
        "--add-data=trained_model.pth;.",  # Include the model file
        "--windowed",  # No console window (GUI mode)
        "--icon=icon.ico" if os.path.exists("icon.ico") else "",  # Add icon if available
        "--hidden-import=torch",
        "--hidden-import=tkinter",
        "--hidden-import=pyperclip",
        "--clean",  # Clean cache
        "generate.py"  # Main script
    ]

    try:
        subprocess.check_call(cmd)
    except subproccess.CalledProcessError as e:
        print(f"\n Build failed: {e}")
        sys.exit(1)

def main():
    install_pyinstaller()
    build_executable()


if __name__ == "__main__":
    main()
