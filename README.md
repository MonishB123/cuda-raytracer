# CUDA Raytracer

A high-performance path-tracing renderer implemented in CUDA.

## Features
- **Path Tracing:** Realistic lighting, soft shadows, and global illumination.
- **Multiple Primitives:** Support for Triangle Meshes (.obj) and Spheres.
- **Advanced Materials:** Lambertian, Metal (reflective), and Dielectric (refractive/glass).
- **GUI Control Center:** A Python-based interface to adjust parameters and preview results.

## Requirements
- NVIDIA GPU with CUDA support.
- Visual Studio 2022 (with MSVC v143).
- Python 3.12+ (for GUI).
- Pillow (Python library): `pip install Pillow`.

## How to Build
Run the `start.bat` file to initialize the environment and build the project:
```powershell
.\start.bat
```

## How to Use

### Using the GUI (Recommended)
Launch the interactive control center:
```powershell
python gui.py
```
From the GUI, you can:
1. Load a scene JSON file.
2. Adjust resolution (Width/Height) and Samples.
3. Modify Camera position and FOV.
4. Click **START RENDER** to see the result.

### Using the Command Line
```powershell
.\main.exe <scene.json> <width> <height> <samples>
```
Example:
```powershell
.\main.exe assets/scenes/demo_scene_advanced.json 400 300 10
```

## Scene File Format
Scenes are defined in JSON. See `assets/scenes/demo_scene_advanced.json` for a complete example including multiple materials and objects.
