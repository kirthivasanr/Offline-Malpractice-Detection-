import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2 # Keep using cv2 for loading, as detection saved with it

class ViolationReviewer(tk.Tk):
    def __init__(self, violations_base_folder="violations"):
        super().__init__()

        self.title("Violation Reviewer")
        self.geometry("1000x750") # Slightly larger default size
        self.configure(bg="#e8e8e8") # Slightly different BG
        self.minsize(800, 600)

        self.violations_folder = violations_base_folder
        if not os.path.exists(self.violations_folder):
            try:
                # Try creating base and default subfolders if they don't exist
                os.makedirs(self.violations_folder)
                default_types = ["peeking", "hand_signs", "passing_object"]
                for dt in default_types:
                    os.makedirs(os.path.join(self.violations_folder, dt), exist_ok=True)
                messagebox.showinfo("Setup", f"Created violations folder structure at: {self.violations_folder}")
            except Exception as e:
                 messagebox.showerror("Error", f"Could not create violations folder: {e}\nPlease check permissions.")
                 self.destroy()
                 return


        self.violation_types = self.get_violation_types()
        if not self.violation_types:
             # Handle case where folder exists but is empty
             self.violation_types = ["(No Types Found)"]


        self.current_type = tk.StringVar(value=self.violation_types[0])
        self.current_images = []
        self.current_index = -1 # Start at -1 for "no image selected" state
        self.tk_img_ref = None # Keep reference to the current PhotoImage

        self.create_ui()
        self.load_images() # Load images for the default type

        # Bind window resize event to update image display
        self.bind("<Configure>", self.on_resize)


    def get_violation_types(self):
        """Get subfolder names from the violations directory."""
        try:
            return sorted([f for f in os.listdir(self.violations_folder)
                           if os.path.isdir(os.path.join(self.violations_folder, f))])
        except Exception as e:
            messagebox.showerror("Error", f"Could not read violation types: {e}")
            return []

    def create_ui(self):
        """Create the user interface."""
        main_frame = ttk.Frame(self, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Configure grid weights for resizing
        main_frame.grid_rowconfigure(1, weight=1) # Image frame row
        main_frame.grid_columnconfigure(0, weight=1) # Full width column

        # --- Header ---
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 15))

        ttk.Label(header_frame, text="Violation Type:", font=("Arial", 10)).pack(side=tk.LEFT, padx=(0, 5))
        self.type_combo = ttk.Combobox(header_frame, textvariable=self.current_type,
                                 values=self.violation_types, state="readonly", width=25, font=("Arial", 10))
        self.type_combo.pack(side=tk.LEFT, padx=5)
        self.type_combo.bind("<<ComboboxSelected>>", lambda e: self.load_images())

        self.counter_label = ttk.Label(header_frame, text="Image 0 / 0", font=("Arial", 10, "bold"))
        self.counter_label.pack(side=tk.RIGHT, padx=5)

        # --- Image Display ---
        # Using a Canvas for potentially better centering/panning in future
        self.image_canvas = tk.Canvas(main_frame, bg="#ffffff", relief=tk.SUNKEN, bd=1)
        self.image_canvas.grid(row=1, column=0, sticky="nsew")
        # Add a label inside the canvas to hold the image
        self.image_label = ttk.Label(self.image_canvas, anchor="center", background="#ffffff")
        self.image_canvas.create_window(0, 0, window=self.image_label, anchor="nw")

        # --- Navigation ---
        nav_frame = ttk.Frame(main_frame)
        nav_frame.grid(row=2, column=0, sticky="ew", pady=(15, 0))

        style = ttk.Style(self)
        style.configure("Nav.TButton", font=("Arial", 10), padding=5)

        ttk.Button(nav_frame, text="<< Previous", command=self.show_previous, style="Nav.TButton").pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="Next >>", command=self.show_next, style="Nav.TButton").pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="Delete Current", command=self.delete_current, style="Nav.TButton").pack(side=tk.RIGHT, padx=2)

        # --- Status Bar ---
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=(5, 3), font=("Arial", 9))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)


    def load_images(self):
        """Load image file paths for the selected violation type."""
        violation_type = self.current_type.get()
        if not violation_type or violation_type == "(No Types Found)":
            self.current_images = []
            self.current_index = -1
            self.update_image_display()
            self.status_var.set("No violation type selected or found.")
            return

        folder_path = os.path.join(self.violations_folder, violation_type)

        if not os.path.isdir(folder_path):
            # This case shouldn't happen if get_violation_types worked, but good practice
            self.current_images = []
            self.current_index = -1
            self.update_image_display()
            self.status_var.set(f"Error: Folder not found for '{violation_type}'")
            return

        try:
            image_files = sorted([
                os.path.join(folder_path, f) for f in os.listdir(folder_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
            ])
            self.current_images = image_files
            self.current_index = 0 if self.current_images else -1
            self.status_var.set(f"Loaded {len(self.current_images)} images for '{violation_type}'")
        except Exception as e:
             self.current_images = []
             self.current_index = -1
             self.status_var.set(f"Error reading images for '{violation_type}': {e}")

        self.update_image_display()


    def update_image_display(self):
        """Update the displayed image on the canvas."""
        num_images = len(self.current_images)
        display_idx = self.current_index + 1 if self.current_index != -1 else 0
        self.counter_label.config(text=f"Image {display_idx} / {num_images}")

        if self.current_index == -1 or not self.current_images:
            # Clear canvas/label
            self.image_label.config(image=None, text="No image selected")
            self.tk_img_ref = None
            # Resize internal label to canvas size
            self.image_canvas.itemconfig(1, width=self.image_canvas.winfo_width(), height=self.image_canvas.winfo_height())
            self.image_canvas.coords(1, self.image_canvas.winfo_width()//2, self.image_canvas.winfo_height()//2) # Center text
            self.image_label.place(relx=0.5, rely=0.5, anchor="center") # Center using place
            if num_images == 0:
                 self.status_var.set(f"No images found for '{self.current_type.get()}'")
            return

        image_path = self.current_images[self.current_index]

        # Check if file exists before loading
        if not os.path.exists(image_path):
             messagebox.showwarning("File Not Found", f"Image file seems to be missing:\n{image_path}\n\nRemoving from list.")
             self.current_images.pop(self.current_index)
             # Adjust index safely
             if num_images - 1 == 0: # Was the last image
                 self.current_index = -1
             elif self.current_index >= len(self.current_images): # If it was the last one in the list
                 self.current_index = len(self.current_images) - 1
             # No change needed if deleting from middle/start

             self.update_image_display() # Reload display
             return


        try:
            # Load with OpenCV (handles various formats, consistent with saving)
            cv_img = cv2.imread(image_path)
            if cv_img is None:
                raise ValueError(f"OpenCV could not read image: {os.path.basename(image_path)}")

            img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            # --- Scaling ---
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()

            if canvas_width <= 1 or canvas_height <= 1:
                # Canvas not yet rendered, postpone update
                self.after(50, self.update_image_display)
                return

            img_w, img_h = pil_img.size
            scale = min(canvas_width / img_w, canvas_height / img_h)

            # Optional: Prevent upscaling beyond original size
            # scale = min(scale, 1.0)

            new_w = int(img_w * scale)
            new_h = int(img_h * scale)

            # Use Pillow's high-quality resize
            resized_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            self.tk_img_ref = ImageTk.PhotoImage(resized_img)

            # Update label inside canvas
            self.image_label.config(image=self.tk_img_ref, text="") # Remove any placeholder text
            self.image_label.image = self.tk_img_ref # Keep reference

            # Position label in the center of the canvas
            # label_x = (canvas_width - new_w) // 2
            # label_y = (canvas_height - new_h) // 2
            # self.image_canvas.coords(1, label_x, label_y) # Move window item
            # self.image_canvas.itemconfig(1, width=new_w, height=new_h) # Resize window item

            # Using place is often simpler for centering
            self.image_label.place(relx=0.5, rely=0.5, anchor="center")

            self.status_var.set(f"Displaying: {os.path.basename(image_path)}")

        except Exception as e:
            error_msg = f"Error loading/displaying {os.path.basename(image_path)}: {e}"
            print(error_msg) # Print detailed error to console
            self.image_label.config(image=None, text=f"Error:\n{e}")
            self.tk_img_ref = None
            self.image_label.place(relx=0.5, rely=0.5, anchor="center")
            self.status_var.set("Error loading image.")


    def on_resize(self, event=None):
        """Handle window resize events to rescale the image."""
        # Debounce resizing slightly using 'after'
        if hasattr(self, '_resize_job'):
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(150, self.update_image_display)


    def show_previous(self):
        """Show the previous image."""
        if not self.current_images: return
        num_images = len(self.current_images)
        if num_images == 0: return

        self.current_index = (self.current_index - 1 + num_images) % num_images
        self.update_image_display()

    def show_next(self):
        """Show the next image."""
        if not self.current_images: return
        num_images = len(self.current_images)
        if num_images == 0: return

        self.current_index = (self.current_index + 1) % num_images
        self.update_image_display()

    def delete_current(self):
        """Delete the currently displayed image file."""
        if self.current_index == -1 or not self.current_images:
            messagebox.showinfo("Delete Image", "No image selected to delete.")
            return

        image_path = self.current_images[self.current_index]
        filename = os.path.basename(image_path)

        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to permanently delete this image?\n\n{filename}"):
            try:
                os.remove(image_path)
                self.status_var.set(f"Deleted: {filename}")
                print(f"Deleted: {image_path}")

                # Remove from list and update display
                self.current_images.pop(self.current_index)
                num_images = len(self.current_images)

                if num_images == 0:
                    self.current_index = -1
                elif self.current_index >= num_images: # If it was the last one
                    self.current_index = num_images - 1
                # Else: index remains valid for the next image

                self.update_image_display()

            except Exception as e:
                messagebox.showerror("Delete Error", f"Failed to delete image:\n{e}")
                self.status_var.set(f"Error deleting {filename}")


def main():
    app = ViolationReviewer()
    app.mainloop()

if __name__ == "__main__":
    main()