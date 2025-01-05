# Face Mask Detection Project

This repository contains the implementation of two deep learning models, YOLO and SSD, for detecting face masks in images. Both models are trained and evaluated to classify faces into three categories: 
- **With Mask**
- **Without Mask**
- **Mask Worn Incorrectly**

## Features
- **YOLO Model**
  - Optimized for real-time applications.
  - Utilizes EfficientNetB0 as the backbone.
- **SSD Model**
  - Simplified implementation with MobileNetV2 backbone.
  - Suitable for resource-constrained environments.
- **Preprocessing and Postprocessing**
  - Image resizing, normalization, and bounding box adjustments.
  - Scaled predictions to original image dimensions.
- **Evaluation Metrics**
  - Precision, Recall, F1-Score, and mAP.

## Requirements

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

### Major Dependencies:
- TensorFlow
- OpenCV
- NumPy
- Pickle

## File Structure

```
.
├── cmp3011.py          # Preprocessing script for images
├── ssd.py              # Implementation of the SSD model
├── test.py             # Testing both YOLO and SSD models
├── yolo.py             # Implementation of the YOLO model
├── README.md           # Project documentation
└── requirements.txt    # Dependencies
```

## How to Run

### 1. Preprocess Images
Use `cmp3011.py` to resize and normalize the images:

```bash
python cmp3011.py
```

### 2. Train Models
- For YOLO:

```bash
python yolo.py
```

- For SSD:

```bash
python ssd.py
```

### 3. Test Models
Evaluate the models on test images using `test.py`:

```bash
python test.py
```

## Results
- YOLO achieves higher accuracy and faster inference times.
- SSD is simpler and effective for smaller devices.

Refer to the comprehensive report for detailed metrics and observations.

## Future Work
- Expand the dataset to include more diverse scenarios.
- Optimize models for deployment on edge devices.

## License
This project is licensed under the MIT License.
