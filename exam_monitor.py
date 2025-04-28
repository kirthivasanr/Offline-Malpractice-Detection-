import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import subprocess # Using subprocess can sometimes be cleaner for independent processes

class ExamMonitorLauncher(tk.Tk):
    def __init__(self):
        super().__init__()

        # Basic window setup
        self.title("Exam Monitor System")
        self.geometry("550x450")
        self.configure(bg="#f8f8f8")
        self.minsize(500, 400)

        # Force window visibility on startup
        self.focus_force()
        self.lift()
        self.attributes('-topmost', True)
        self.after(100, lambda: self.attributes('-topmost', False)) # Allow other windows on top later

        # Padding
        self.padx, self.pady = 25, 25

        # Main Frame
        main_frame = ttk.Frame(self, padding=(self.padx, self.pady))
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(main_frame,
                               text="Exam Malpractice Monitor",
                               font=("Arial", 18, "bold"),
                               foreground="#2c3e50")
        title_label.pack(pady=(0, 20))

        # Description
        desc_text = """Detects potential malpractice behaviors:
• Head Pose indicating Peeking
• Hand Gestures for Communication
• Suspicious Object Movement/Presence"""
        desc_label = ttk.Label(main_frame, text=desc_text,
                              justify=tk.LEFT,
                              font=("Arial", 11),
                              wraplength=480) # Adjusted wrap length
        desc_label.pack(pady=(0, 25), fill=tk.X)

        # Buttons Frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X, pady=15)

        # Button Style
        style = ttk.Style(self)
        style.configure("TButton", font=("Arial", 11), padding=8)
        button_width = 20

        # Button Container (for centering)
        button_container = ttk.Frame(buttons_frame)
        button_container.pack() # Center alignment by default

        # --- Buttons ---
        monitor_btn = ttk.Button(button_container,
                               text="Start Monitoring",
                               command=self.start_monitoring,
                               width=button_width)
        monitor_btn.pack(pady=7)

        review_btn = ttk.Button(button_container,
                              text="Review Violations",
                              command=self.review_violations,
                              width=button_width)
        review_btn.pack(pady=7)

        exit_btn = ttk.Button(button_container,
                            text="Exit",
                            command=self.quit_app, # Use a dedicated quit method
                            width=button_width)
        exit_btn.pack(pady=7)

        # Status Bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self,
                              textvariable=self.status_var,
                              relief=tk.SUNKEN,
                              anchor=tk.W,
                              padding=(10, 5),
                              background="#e0e0e0", # Slightly different background
                              font=("Arial", 9))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def run_script(self, script_name, module_name):
        """Handles running the detection or reviewer script."""
        script_path = os.path.join(os.path.dirname(__file__), script_name)

        # --- Explicitly define the Python interpreter path ---
        # Make SURE this path is correct for your global Python 3.10 install where OpenCV exists
        python_executable = "C:\\Users\\kirth\\AppData\\Local\\Programs\\Python\\Python310\\python.exe"
        # Or using forward slashes:
        # python_executable = "C:/Users/kirth/AppData/Local/Programs/Python/Python310/python.exe"

        # Verify the executable exists before proceeding (optional but good check)
        if not os.path.exists(python_executable):
            messagebox.showerror("Error", f"Python executable not found at specified path:\n{python_executable}\nPlease correct the path in exam_monitor.py")
            self.status_var.set("Error: Python path incorrect")
            return

        # --- Keep the rest of the function the same ---
        if not os.path.exists(script_path):
             messagebox.showerror("Error", f"{script_name} not found.")
             self.status_var.set(f"Error: {script_name} not found")
             return

        # Basic dependency check (runs in *this* environment, may not reflect child)
        # It's less critical now that we force the interpreter, but keep it for basic check
        try:
            if module_name == "detection":
                 import cv2 # Check if launcher can import (might differ from child)
                 import mediapipe
            elif module_name == "reviewer":
                 from PIL import Image, ImageTk
                 import cv2
        except ImportError as e:
            # This error might be misleading if the global env is different,
            # but still useful if basic libraries are missing everywhere.
             messagebox.showwarning("Dependency Check",
                                 f"Launcher check failed for: {e.name}. "
                                 f"Ensuring child process uses correct Python at:\n{python_executable}")
            # Don't return here, let the child process try

        # Run the script using the *explicit* python executable
        try:
            self.status_var.set(f"Launching {module_name} with {os.path.basename(python_executable)}...")
            self.update()
            self.iconify()

            # Use the explicitly defined python_executable
            process = subprocess.Popen([python_executable, script_path])
            process.wait()

        except Exception as e:
            messagebox.showerror("Execution Error", f"Failed to start {module_name} using\n{python_executable}:\n{str(e)}")
            self.status_var.set(f"Error launching {module_name}")
        finally:
            self.deiconify()
            self.lift()
            self.focus_force()
            self.status_var.set(f"{module_name.capitalize()} closed. Ready.")


    def start_monitoring(self):
        """Start the malpractice detection system"""
        self.run_script("malpractice_detection.py", "detection")

    def review_violations(self):
        """Open the violation reviewer"""
        self.run_script("violation_reviewer.py", "reviewer")

    def quit_app(self):
        """Cleanly exit the application."""
        self.quit()
        self.destroy()

def main():
    # Check if required files exist before starting GUI
    required_files = ["malpractice_detection.py", "violation_reviewer.py", "requirements.txt"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"ERROR: Missing required file(s): {', '.join(missing_files)}")
        print("Please ensure all script files and requirements.txt are in the same directory.")
        # Optionally show a simple tkinter error message if GUI is desired even on error
        root = tk.Tk()
        root.withdraw() # Hide the main window
        messagebox.showerror("Startup Error", f"Missing required file(s): {', '.join(missing_files)}\nPlease restore them and restart.")
        root.destroy()
        return # Exit if files are missing

    app = ExamMonitorLauncher()
    app.mainloop()

if __name__ == "__main__":
    main() 