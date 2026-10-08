# YOLO-3D

Real-time 3D object detection combining **YOLOv11** (detection + tracking) with **Depth Anything V2**
(depth estimation) for pseudo-3D bounding boxes, Bird's Eye View visualization, and per-object
distance logging in **meters**.

## Versions

- `v2` (current): realtime pass — parallel detect+depth, depth every 3rd frame,
  small-model defaults. Full changelog in the
  [v2 release](https://github.com/Shanmukh-Villuri/YOLO-3D/releases/tag/v2).

## Features

- Object detection + tracking with YOLOv11 (`nano`–`extra`)
- Metric depth estimation in **meters** (Depth Anything V2 metric checkpoints)
- Pseudo-3D bounding boxes with `Distance: XX.XXm` labels
- Bird's Eye View (BEV) visualization scaled in true meters
- Per-frame distance log (`distances_log.csv`)
- Video file or webcam input, annotated video output
- CUDA with automatic CPU fallback

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Install the pinned deps:

```bash
pip install -r requirements.txt
```

Two imports are **not** in `requirements.txt`, install them if needed:

```bash
pip install transformers huggingface_hub  # required by depth_model.py
pip install moviepy                       # only needed for cut.py
```

## Model weights (auto-downloaded, not in repo)

Nothing to download manually. On first run:

- YOLO weights auto-download via `YOLO(model_name)` (`detection_model.py`) — `yolo11n/s/m/l/x`.
- Depth weights auto-download from Hugging Face via `transformers.pipeline()`
  (`depth_model.py`) — e.g. `depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf`.

## Usage

1. Put your video in `input/` (empty placeholder dirs are committed via `.gitkeep`).
2. Edit the config block at the top of `main()` in `run.py` (paths are machine-specific):

```python
source = "input/your_video.mp4"   # video file, or 0 for webcam
output_path = "output/result.mp4"

yolo_model_size = "large"   # "nano", "small", "medium", "large", "extra"
depth_model_size = "large"  # "small", "base", "large"
depth_metric = True         # True = meters, False = relative 0-1
depth_scene = "outdoor"     # "indoor" or "outdoor" — must match your scene (metric only)

device = 'cuda'             # falls back to CPU on failure

conf_threshold = 0.25
iou_threshold = 0.45
classes = None              # e.g. [0, 1, 2] to keep only some classes

enable_tracking = True
enable_bev = True
enable_pseudo_3d = True
```

3. Run:

```bash
python run.py
```

4. Press `q` or `Esc` to stop. Outputs:
   - Annotated video at `output_path`
   - `distances_log.csv` in the working directory: `frame_idx,object_id,class_name,x1,y1,x2,y2,predicted_distance_m`

## Helper scripts

- `frames_vid.py` — stitch an image folder (e.g. KITTI frames) into a video. Edit
  `input_folder`, `output_video`, `fps` inside the file first.
- `cut.py` — trim a video with moviepy. Edit `input_video`, `output_video`,
  `start_time`, `end_time` inside the file first. Writes `timestamps.json`.

Both contain absolute paths from the original machine — update them before use.

## Project structure

```
YOLO-3D/
├── run.py                  # Main script (edit config here)
├── detection_model.py      # YOLOv11 detection + tracking
├── depth_model.py          # Depth Anything V2 (metric or relative)
├── bbox3d_utils.py         # Pseudo-3D drawing, BEV, Kalman/3D-box utils
├── load_camera_params.py   # Optional camera calibration utilities
├── requirements.txt
├── input/                  # Your input videos (empty, .gitkeep)
└── output/                 # Generated videos (empty, .gitkeep)
```

Generated/ignored (never committed): `*.pt` weights, `input/*`, `output/*`,
`distances_log.csv`, `timestamps.json`, `__pycache__/`.

## How it works

1. **Detection**: YOLOv11 finds 2D boxes and track IDs per frame.
2. **Depth**: Depth Anything V2 predicts a dense depth map (meters when
   `depth_metric=True`, no per-frame renormalization).
3. **Association**: center-point depth for `person/cat/dog`, median box depth otherwise.
4. **Visualization**: pseudo-3D boxes (`Distance: XX.XXm`), BEV plot in meters, depth-map inset.

## Notes / caveats

- Set `depth_scene` to match your footage. Wrong scene (`indoor` vs `outdoor`) gives wrong scale.
- Metric depth is still monocular estimation — treat distances as approximate.
- `device='cuda'` needs a CUDA PyTorch build; both models fall back to CPU on init failure.
- Depth legend: overlay text shows the sampling method (`center` or `median`).

## Acknowledgments

- YOLOv11 by Ultralytics
- Depth Anything V2
