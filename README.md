# Chest X-Ray Pneumonia Detection

A Streamlit web application for classifying chest X-ray images as **Normal** or **Pneumonia** using deep learning models.

## Features

- Upload chest X-ray images (JPG, JPEG, PNG, BMP)
- Choose between two classification models:
  - **YOLOv8-cls** - Higher accuracy model
  - **ResNet-50** - Alternative classification model
- Real-time predictions with confidence scores
- Visual comparison of model performance metrics

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| YOLOv8-cls | 94.87% | 93.23% | 98.97% | 96.02% |
| ResNet-50 | 86.22% | 84.70% | 95.13% | 89.61% |

## Installation

1. Clone the repository and navigate to the streamlit directory:
   ```bash
   cd streamlit
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install model-specific dependencies:
   ```bash
   pip install keras tensorflow ultralytics
   ```

## Required Model Files

Ensure the following model files are in the project directory:

- `yolov8-cls-pneumonia-classification-model.pt` - YOLOv8 classification model
- `resnet-50-pneumonia-classification-model.keras` - ResNet-50 Keras model

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### Steps to Use

1. **Select Model** - Choose either YOLOv8-cls or ResNet-50
2. **Upload Image** - Upload a chest X-ray image
3. **View Results** - See the diagnosis and confidence score

## Project Structure

```
streamlit/
├── app.py                                      # Main Streamlit application
├── requirements.txt                            # Python dependencies
├── README.md                                   # This file
├── yolov8-cls-pneumonia-classification-model.pt    # YOLOv8 model weights
└── resnet-50-pneumonia-classification-model.keras  # ResNet-50 model weights
```

## Technical Details

- **Image Size**: 224x224 pixels
- **Classification Threshold**: 0.3
- **Classes**: Normal, Pneumonia
- **Supported Formats**: JPG, JPEG, PNG, BMP

## Disclaimer

This tool is for **educational purposes only** and should not be used for medical diagnosis. Always consult a qualified healthcare professional for medical advice.

## License

This project is for educational and research purposes.
