# Orange_Detection_YOLOv4
Custom trained YOLOv4 in python for orange detection and yield estimation.

## Inference using GUI 
1. Create virtual environment (recommneded)
2. Clone the repo or download the repo's .zip file.
3. Install necessary packages by running "pip install -r requirements.txt" on command line
4. Download custom (trained on custom orange dataset) yolo weights from https://drive.google.com/file/d/1Mo6yHlriemyFqWsulWO2az3QINwsTX6i/view?usp=sharing.
5. run "python3 main.py" for starting GUI
6. Browse the images to detect oranges
7. Provide path to base folder (which is yolo-orange) containing .cfg, .name  and .weights files.
8. Adjust Confidence and Threshold for YOLOv4 detector
9. Press OK to get detectors result.
