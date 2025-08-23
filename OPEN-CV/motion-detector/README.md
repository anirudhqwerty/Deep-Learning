# Motion Detector with OpenCV

A simple real-time motion detection system using **OpenCV**. It uses the webcam feed, detects motion by comparing frames, and highlights moving objects with bounding boxes.

***

## Features

- Uses your webcam to monitor for motion.
- Draws rectangles and shows "Motion Detected" when movement is captured.
- Easy to run, modify, and integrate.

***

## Requirements

- Python 3.x
- OpenCV (`cv2`)

Install OpenCV via pip:

```bash
pip install opencv-python
```

***

## How to Run

1. Save the code below as `motion_detector.py`
2. Run the script:

```bash
python motion_detector.py
```

## Notes

- Press `Esc` key (27) to exit.
- To adjust sensitivity, change the value in `cv2.contourArea(contour) < 500`.
- For smoother motion capture, modify the value in `cv2.waitKey(40)`.

***