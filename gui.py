import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import subprocess
import os
from PIL import Image, ImageTk
import threading

class RaytracerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CUDA Raytracer Control Center")
        self.root.geometry("1100x700")
        
        self.scene_data = {}
        self.current_scene_path = ""
        
        self.setup_ui()
        self.load_default_scene()

    def setup_ui(self):
        # Main Layout
        self.main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_pane.pack(fill=tk.BOTH, expand=True)

        # Left Sidebar (Controls)
        self.sidebar = ttk.Frame(self.main_pane, width=300)
        self.main_pane.add(self.sidebar, weight=0)

        # Right Area (Preview & Log)
        self.right_area = ttk.Frame(self.main_pane)
        self.main_pane.add(self.right_area, weight=1)

        self.setup_sidebar()
        self.setup_preview_area()

    def setup_sidebar(self):
        padding = {'padx': 10, 'pady': 5}
        
        # Scene Selection
        ttk.Label(self.sidebar, text="Scene Configuration", font=('Helvetica', 12, 'bold')).pack(**padding)
        
        scene_frame = ttk.LabelFrame(self.sidebar, text="File")
        scene_frame.pack(fill=tk.X, **padding)
        
        self.btn_load = ttk.Button(scene_frame, text="Load Scene JSON", command=self.open_scene)
        self.btn_load.pack(fill=tk.X, padx=5, pady=5)
        
        self.lbl_scene_name = ttk.Label(scene_frame, text="No scene loaded", wraplength=250)
        self.lbl_scene_name.pack(fill=tk.X, padx=5, pady=5)

        # Render Settings
        render_frame = ttk.LabelFrame(self.sidebar, text="Render Settings")
        render_frame.pack(fill=tk.X, **padding)

        ttk.Label(render_frame, text="Width:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.ent_width = ttk.Entry(render_frame, width=10)
        self.ent_width.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        self.ent_width.insert(0, "400")

        ttk.Label(render_frame, text="Height:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.ent_height = ttk.Entry(render_frame, width=10)
        self.ent_height.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        self.ent_height.insert(0, "300")

        ttk.Label(render_frame, text="Samples:").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.ent_samples = ttk.Entry(render_frame, width=10)
        self.ent_samples.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        self.ent_samples.insert(0, "10")

        # Camera Settings
        cam_frame = ttk.LabelFrame(self.sidebar, text="Camera")
        cam_frame.pack(fill=tk.X, **padding)

        ttk.Label(cam_frame, text="Position:").grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=5)
        self.cam_pos_x = ttk.Entry(cam_frame, width=7); self.cam_pos_x.grid(row=1, column=0, padx=2)
        self.cam_pos_y = ttk.Entry(cam_frame, width=7); self.cam_pos_y.grid(row=1, column=1, padx=2)
        self.cam_pos_z = ttk.Entry(cam_frame, width=7); self.cam_pos_z.grid(row=1, column=2, padx=2)

        ttk.Label(cam_frame, text="Look At:").grid(row=2, column=0, columnspan=3, sticky=tk.W, padx=5, pady=(5,0))
        self.cam_look_x = ttk.Entry(cam_frame, width=7); self.cam_look_x.grid(row=3, column=0, padx=2)
        self.cam_look_y = ttk.Entry(cam_frame, width=7); self.cam_look_y.grid(row=3, column=1, padx=2)
        self.cam_look_z = ttk.Entry(cam_frame, width=7); self.cam_look_z.grid(row=3, column=2, padx=2)

        ttk.Label(cam_frame, text="FOV:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.ent_fov = ttk.Entry(cam_frame, width=10)
        self.ent_fov.grid(row=4, column=1, sticky=tk.W, padx=5)

        # Actions
        self.btn_render = ttk.Button(self.sidebar, text="START RENDER", command=self.start_render)
        self.btn_render.pack(fill=tk.X, padx=10, pady=20)
        
        self.progress = ttk.Progressbar(self.sidebar, mode='indeterminate')
        self.progress.pack(fill=tk.X, padx=10)

    def setup_preview_area(self):
        # Preview Image
        self.canvas = tk.Canvas(self.right_area, bg="#222")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Log Output
        self.log_text = tk.Text(self.right_area, height=8, bg="#111", fg="#0f0", font=('Consolas', 9))
        self.log_text.pack(fill=tk.X, padx=10, pady=(0, 10))

    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def open_scene(self):
        path = filedialog.askopenfilename(initialdir="assets/scenes", filetypes=[("JSON files", "*.json")])
        if path:
            self.load_scene(path)

    def load_scene(self, path):
        try:
            with open(path, 'r') as f:
                self.scene_data = json.load(f)
            self.current_scene_path = path
            self.lbl_scene_name.config(text=os.path.basename(path))
            
            # Fill inputs
            cam = self.scene_data.get('camera', {})
            pos = cam.get('position', [0,0,0])
            look = cam.get('look_at', [0,0,0])
            
            self.cam_pos_x.delete(0, tk.END); self.cam_pos_x.insert(0, str(pos[0]))
            self.cam_pos_y.delete(0, tk.END); self.cam_pos_y.insert(0, str(pos[1]))
            self.cam_pos_z.delete(0, tk.END); self.cam_pos_z.insert(0, str(pos[2]))
            
            self.cam_look_x.delete(0, tk.END); self.cam_look_x.insert(0, str(look[0]))
            self.cam_look_y.delete(0, tk.END); self.cam_look_y.insert(0, str(look[1]))
            self.cam_look_z.delete(0, tk.END); self.cam_look_z.insert(0, str(look[2]))
            
            self.ent_fov.delete(0, tk.END); self.ent_fov.insert(0, str(cam.get('fov', 40)))
            
            self.log(f"Loaded scene: {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load scene: {e}")

    def load_default_scene(self):
        default = "assets/scenes/demo_scene_advanced.json"
        if os.path.exists(default):
            self.load_scene(default)

    def start_render(self):
        if not self.current_scene_path:
            messagebox.showwarning("Warning", "Please load a scene first")
            return
            
        # Update JSON with current UI values
        try:
            self.scene_data['camera']['position'] = [
                float(self.cam_pos_x.get()), float(self.cam_pos_y.get()), float(self.cam_pos_z.get())
            ]
            self.scene_data['camera']['look_at'] = [
                float(self.cam_look_x.get()), float(self.cam_look_y.get()), float(self.cam_look_z.get())
            ]
            self.scene_data['camera']['fov'] = float(self.ent_fov.get())
            
            # Save temporary scene
            temp_scene = "assets/scenes/temp_gui_scene.json"
            with open(temp_scene, 'w') as f:
                json.dump(self.scene_data, f, indent=2)
                
            width = self.ent_width.get()
            height = self.ent_height.get()
            samples = self.ent_samples.get()
            
            self.btn_render.config(state=tk.DISABLED)
            self.progress.start()
            
            # Run in thread to keep GUI responsive
            thread = threading.Thread(target=self.run_process, args=(temp_scene, width, height, samples))
            thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {e}")

    def run_process(self, scene_path, width, height, samples):
        try:
            cmd = [".\\main.exe", scene_path, width, height, samples]
            self.log(f"Running: {' '.join(cmd)}")
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                self.root.after(0, self.log, line.strip())
            
            process.wait()
            self.root.after(0, self.finish_render)
        except Exception as e:
            self.root.after(0, self.log, f"Error: {e}")
            self.root.after(0, self.finish_render)

    def finish_render(self):
        self.progress.stop()
        self.btn_render.config(state=tk.NORMAL)
        self.update_preview()

    def update_preview(self):
        if os.path.exists("output.ppm"):
            try:
                img = Image.open("output.ppm")
                
                # Resize to fit canvas
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                
                if canvas_width > 1 and canvas_height > 1:
                    img.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
                
                self.photo = ImageTk.PhotoImage(img)
                self.canvas.create_image(canvas_width//2, canvas_height//2, image=self.photo)
                self.log("Preview updated.")
            except Exception as e:
                self.log(f"Failed to load preview: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = RaytracerGUI(root)
    root.mainloop()
