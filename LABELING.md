# SAM3 Segmentation Pipeline

This pipeline implements a **segmentation pipeline using SAM3** to segment two objects:

```
scripts/1_segmentation_pipeline/__segmentation_pipeline__.py
```

The pipeline performs:

1. Interactive first-frame labeling in the browser  
2. Video fragmentation on the server  
3. SAM3 segmentation across overlapping segments  
4. Automatic prompt propagation between segments  
5. Optional absent-object review and manual relabeling  
6. Mask export and overlay visualization  

The workflow now runs **entirely on the Ubuntu GPU server**.

| Environment | Purpose |
|-------------|---------|
| **Ubuntu GPU Server** | Labeling, fragmentation, SAM3 segmentation, output generation |
| **Browser on Local Machine** | Remote annotation UI via SSH port forwarding |
---

## Overview of the Workflow

```
Local Machine
    │ 
    │ 1. Upload full video to server
    |
    ▼
Ubuntu GPU Server
    │
    │ 2. Label first frame in browser
    │ 3. Fragment video into overlapping segments
    │ 4. Run SAM3 segmentation pipeline
    │
    ▼
Outputs
   masks_raw/
   masks_viz/
   overlay.mp4
   manual_review/   (optional, if absent objects reappear)
```

### Pipeline Command Summary

The config.json acts as the main control file, where all the parameters are defined. Created one per video you want to segment.

**Make sure to adapt the path of the --config configs/dd_m_yyyy_name/config_videoxxx.json.**

#### Label

```bash
python scripts/1_segmentation_pipeline/__segmentation_pipeline__.py \
--config configs/test_video/config_video001.json label
```

#### Fragment

```bash
python scripts/1_segmentation_pipeline/__segmentation_pipeline__.py \
--config configs/test_video/config_video001.json fragment
```

#### Segment

```bash
python scripts/1_segmentation_pipeline/__segmentation_pipeline__.py \
--config configs/test_video/config_video001.json segment
```
---

# Detailed description of the pipeline

## 1. Install Required Packages

Check the requirements_sam3_server.txt file. In this file all the required packages and their version are saved.

---

## 2. Create the Pipeline Configuration

Create a configuration file, for example:

**Make sure to follow the predefined folder structure to keep a clean working environment**

```
configs/dd_mm_yyyy_name/config_videoxxx.json
```

Example configuration:

```json
{
  "video_id": "video_001",

  "server": {
    "input_video": "data/raw/test_run/video_001/video_001.mp4",
    "segments_dir": "data/raw/test_run/video_001/segments",
    "prompts_json": "data/prompts/test_run/video_001_points.json",
    "frame0_image": "data/prompts/test_run/video_001_frame0.jpg",
    "outputs_root": "data/sam3_outputs/test_run/video_001"
  },

  "fragment": {
    "segment_seconds": 15,
    "overlap_frames": 1
  },

  "sam3": {
    "auto_pos": 8,
    "auto_neg": 8,
    "seed": 123
  },

  "label": {
    "host": "127.0.0.1",
    "port": 5000,
    "object_ids": [1, 2],
    "open_browser": false
  }
}
```

---

### Meaning of the SAM3 Parameters

```
"sam3": {
  "auto_pos": 8,
  "auto_neg": 8,
  "seed": 123
}
```

These parameters control **how the next segment is initialized from the previous segment’s last good mask**.

| Parameter | Meaning |
|-----------|---------|
| `auto_pos` | Number of positive points sampled inside the previous mask |
| `auto_neg` | Number of negative points sampled outside the previous mask |
| `seed` | Random seed for reproducible prompt sampling |

This is needed because SAM3 does **not automatically remember objects across separate video files**.

Your pipeline bridges segments using:

```
last good mask → sample points → prompts for next segment
```

---

## 3. Upload the Full Video to the Server

Upload the full input video to the server.

Example from Windows PowerShell:

```bash
scp "C:\\path\\to\\video_001.mp4" \
<user>@<gpu-server>:~/projects/master_thesis_social_behaviour/data/raw/test_run/
```
To log into the server, check the README.md file.

## 4. Label the First Frame (Browser-Based, Server-Hosted)

Run the labeling stage on the server with the following command:

```bash
python scripts/1_segmentation_pipeline/__segmentation_pipeline__.py \
    --config configs/test_video/config_video001.json \
    label
```

This starts a small Flask server for interactive annotation. 
**Make sure to adapt the path of the --config configs/dd_m_yyyy_name/config_videoxxx.json.**

If you are connecting from your local machine, create SSH port forwarding:

```bash
ssh -L 5000:localhost:5000 <user>@<gpu-server>
```

Then open this URL in your browser (ctrl + right click):

```
http://localhost:5000
```

The first frame of the full uploaded video will be shown.

---

### Labeling Controls

| Action | Control |
|------|------|
| Positive prompt | Left mouse click |
| Negative prompt | Right mouse click |
| Undo | Undo button |
| Next object | Next object button |
| Save | Save button |
| Zoom | Mouse wheel |
| Pan | Drag image |

After saving, the prompt file is written to:

```
data/prompts/test_run/video_001_points.json
```

And frame 0 is stored as:

```
data/prompts/test_run/video_001_frame0.jpg
```

---

## 5. Prompt JSON Format

Example JSON:

```json
{
  "objects": {
    "1": {
      "points_xy": [[960, 600], [980, 610], [930, 580]],
      "labels": [1, 1, 0]
    },
    "2": {
      "points_xy": [[300, 500], [320, 520]],
      "labels": [1, 0]
    }
  }
}
```

| Field | Meaning |
|------|------|
| `points_xy` | Pixel coordinates `[x, y]` |
| `labels` | `1 = positive`, `0 = negative` |

There is **no limit on the number of prompt points**.

---

## 6. Fragment the Video

Split the full video into overlapping segments:

```bash
python scripts/1_segmentation_pipeline/__segmentation_pipeline__.py \
    --config configs/test_video/config_video001.json \
    fragment
```

Output directory:

```
data/raw/test_run/video_001/segments/
```

Example output:

```
segment_001.mp4
segment_002.mp4
segment_003.mp4
...
```

Each segment overlaps with the previous one by **one frame**.

---

## 7. Run the SAM3 Segmentation Pipeline

```bash
python scripts/1_segmentation_pipeline/__segmentation_pipeline__.py \
    --config configs/test_video/config_video001.json \
    segment
```

This runs SAM3 sequentially across all segments.

---

### What the Segmentation Pipeline Does

The pipeline:

1. Loads `segment_001`
2. Applies manual prompts from JSON
3. Runs SAM3 propagation
4. Extracts final mask of the segment
5. Samples new prompts from the mask
6. Uses them to initialize the next segment
7. Repeats until the video is finished
8. Writes masks and overlay frames

---

## 8. Absent Object Handling

If an object disappears:

```
[frame=245 obj=1] ABSENT: object disappeared
```

When a candidate mask appears again:

```
Manual review required
```

You will be asked:

```
Your choice [y/r/s]
```

| Key | Action |
|----|----|
| y | accept candidate mask |
| r | relabel object |
| s | keep object absent |

Manual review images are saved to:

```
data/sam3_outputs/video_001/full/manual_review/
```

**Recommend Workflow:** 
1. Compare the file genereated in the manual_review folder with the last few generated masks in the folder masks_viz
2. Check if an object is absent or if there was an error while labelling (both objects got interpreted as one)
3. Mainly use r or s

---

## 9. Output Files

Outputs are written to:

```
data/sam3_outputs/video_001/full/
```

Structure:

```
overlay.avi
overlay.mp4

masks_raw/
masks_viz/
manual_review/
```

---
