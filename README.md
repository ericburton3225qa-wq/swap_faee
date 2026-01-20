# Deep Fake - Live Camera Face Swapping

A real-time face swapping application using InsightFace and GFPGAN. This project provides multiple tools for swapping faces in live camera feeds or static images with optional AI-powered face enhancement.

## Features

- **Live Camera Face Swapping**: Real-time face swap from webcam with CLI and GUI interfaces
- **Multiple Tools**:
  - `test_insight.py` - CLI tool for live camera face swapping
  - `test_insight_ui.py` - Tkinter GUI with visual controls
  - `test_camera.py` - Simple camera viewer with FPS overlay
  - `test_insight_face.py` - Image-to-image face swapping

- **Performance Optimizations**:
  - GPU acceleration (CUDA support with automatic detection)
  - Frame skipping (process every 2nd frame for ~50% speedup)
  - Float16 model support for reduced memory usage

- **Face Enhancement**:
  - Optional GFPGAN face quality enhancement (2x upscaling)
  - Real-time FPS monitoring and display

- **Cross-Platform Support**:
  - GPU acceleration detection (ONNX Runtime + PyTorch)
  - Headless environment support (graceful fallback)
  - Windows console compatibility

## Requirements

- Python 3.8+
- OpenCV (cv2)
- InsightFace >= 0.7
- PyTorch
- GFPGAN
- ONNX Runtime

## Installation

### Basic Installation

```bash
pip install opencv-python insightface torch gfpgan onnxruntime
```

### With GPU Support (CUDA)

```bash
# ONNX Runtime with GPU
pip install onnxruntime-gpu

# PyTorch with CUDA (adjust cu version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Full Setup

```bash
pip install -r requirements.txt  # If available
# or manually:
pip install opencv-python insightface>=0.7 torch gfpgan onnxruntime-gpu
```

## Usage

### Live Camera Face Swapping (CLI)

```bash
python test_insight.py -s source_image.jpg -m
```

With GFPGAN enhancement:
```bash
python test_insight.py -s source_image.jpg -m -e
```

### GUI Version (Tkinter)

```bash
python test_insight_ui.py
```

Features:
- Source image preview
- Camera ID selection
- Mirror and enhancement toggles
- Real-time FPS display
- Start/Stop controls

### Simple Camera Viewer

```bash
python test_camera.py -m
```

### Image-to-Image Face Swapping

```bash
python test_insight_face.py -s source_image.jpg -t target_image.jpg
```

## Command Line Arguments

### test_insight.py (Live Camera)

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--source` | `-s` | string | required | Path to source image (face to swap from) |
| `--camera` | `-c` | int | 0 | Camera ID |
| `--mirror` | `-m` | flag | False | Mirror the camera feed |
| `--enhance` | `-e` | flag | False | Enable GFPGAN face enhancement |
| `--width` | `-w` | int | 640 | Frame width (pixels) |
| `--height` | `-H` | int | 480 | Frame height (pixels) |
| `--fps` | `-f` | int | 30 | Target FPS |

### test_camera.py (Simple Camera Viewer)

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--camera` | `-c` | int | 0 | Camera ID |
| `--mirror` | `-m` | flag | False | Mirror the camera feed |
| `--width` | `-w` | int | 640 | Frame width (pixels) |
| `--height` | `-H` | int | 480 | Frame height (pixels) |
| `--smoothing` | | float | 0.7 | FPS smoothing factor |
| `--print-interval` | | int | 30 | Console FPS print interval (frames) |

## Models

The project uses the following pre-trained models:

- **Face Detection & Swapping**: `inswapper_128_fp16.onnx`
  - Automatically downloaded on first run
  - Float16 precision for reduced memory usage
  - Located in `Deep-Live-Cam/models/`

- **Face Enhancement (Optional)**: `GFPGANv1.4.pth`
  - 2x upscaling with artifact removal
  - Located in `Deep-Live-Cam/models/`

## Performance

### CPU Mode
- ~5-10 FPS with frame skipping enabled
- ~2-5 FPS without frame skipping

### GPU Mode (CUDA)
- ~15-30 FPS with frame skipping enabled
- ~10-20 FPS without frame skipping

*Performance varies based on GPU model and frame resolution*

## How It Works

1. **Source Face Loading**: Extracts face embedding from the source image using InsightFace
2. **Real-time Detection**: Detects faces in each camera frame
3. **Face Swapping**: Applies the source face to all detected faces using the inswapper model
4. **Optional Enhancement**: Applies GFPGAN to enhance face quality (2x upscaling)
5. **Frame Display**: Shows the processed frame with FPS overlay

### Frame Skipping Optimization

The application processes every 2nd frame to improve performance while maintaining smooth playback by reusing the last processed frame for intermediate display.

## GPU Acceleration

The project automatically detects available GPU:

- **ONNX Runtime**: Checks for `CUDAExecutionProvider`
- **PyTorch**: Checks `torch.cuda.is_available()`
- **Fallback**: Uses CPU if no GPU is available

GPU status is printed at startup:
```
FaceAnalysis initialized with ctx_id=0 (use_cuda=True)
```

## Troubleshooting

### Issue: "No faces found in source image"
- Ensure the source image contains a clear, front-facing face
- Try a different source image

### Issue: Display not available
- The application gracefully falls back to background processing
- Frames are still processed even without display output

### Issue: Out of memory (OOM)
- Reduce frame resolution with `-w` and `-H` arguments
- Use float16 model (already configured)
- Process fewer frames with frame skipping

### Issue: Data type mismatch error
- Ensure you're using the fp16 model: `inswapper_128_fp16.onnx`
- The code handles float16 conversion automatically

## Project Structure

```
deep-fake/
├── README.md
├── test_insight.py           # Live camera face swapping (CLI)
├── test_insight_ui.py        # GUI version with Tkinter
├── test_camera.py            # Simple camera viewer
├── test_insight_face.py      # Image-to-image swapping
├── inswapper_128.onnx        # Face swap model
├── Deep-Live-Cam/            # Deep-Live-Cam project files
│   ├── models/
│   │   ├── GFPGANv1.4.pth   # Face enhancement model
│   │   ├── inswapper_128.onnx
│   │   └── inswapper_128_fp16.onnx
│   └── ...
└── requirements.txt          # Python dependencies
```

## Limitations

- Works best with clear, front-facing faces
- Background quality depends on GPU resources
- Real-time processing on CPU is significantly slower than GPU

## Credits

- **InsightFace**: Face detection and swapping models
- **GFPGAN**: Face enhancement and restoration
- **Deep-Live-Cam**: Project inspiration and model resources
- **OpenCV**: Computer vision operations

## License

This project is based on open-source components. Please refer to individual project licenses:
- InsightFace: MIT License
- GFPGAN: Apache License 2.0
- Deep-Live-Cam: Check their LICENSE file

## Contributing

Contributions are welcome! Feel free to:
- Report bugs and issues
- Suggest improvements
- Submit pull requests
- Share performance optimization tips

## Contact & Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

**Note**: This project is for educational and research purposes. Ensure you have proper consent when using face swapping on others' images or videos.
