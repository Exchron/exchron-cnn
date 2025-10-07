# Lightkurve Exoplanet Detection API - Input Specifications

## Overview

This document describes the expected input types and formats for making API calls to the Lightkurve Exoplanet Detection system. The system uses a trained CNN model to predict exoplanet dispositions from Kepler telescope light curve data.

## Core Components

### 1. LightkurvePredictor Class

The main prediction class that handles model loading and predictions.

#### Initialization Parameters

```python
LightkurvePredictor(
    model_path='final_lightkurve_cnn_model.keras',
    metadata_path='lightkurve_model_metadata.json', 
    lightkurve_data_path='lightkurve_data/'
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `str` | `'final_lightkurve_cnn_model.keras'` | Path to the trained Keras model file |
| `metadata_path` | `str` | `'lightkurve_model_metadata.json'` | Path to model metadata JSON file |
| `lightkurve_data_path` | `str` | `'lightkurve_data/'` | Directory containing lightkurve CSV files |

## API Methods and Input Specifications

### 1. Single Prediction

#### Method: `predict_single(kepler_id)`

**Input:**
- `kepler_id` (int): The Kepler ID to predict

**Expected File Format:**
- File naming convention: `kepler_{kepler_id}_lightkurve.csv`
- Location: `{lightkurve_data_path}/kepler_{kepler_id}_lightkurve.csv`

**Output:**
```python
{
    'kepler_id': int,
    'predicted_class': str,
    'confidence': float,
    'class_probabilities': dict,
    'file_path': str
}
```

### 2. Prediction with Ground Truth Comparison

#### Method: `predict_with_metadata(kepler_id, metadata_file)`

**Input:**
- `kepler_id` (int): The Kepler ID to predict
- `metadata_file` (str, optional): Path to metadata CSV file (default: `'lightkurve_test_metadata.csv'`)

**Output:**
```python
{
    'kepler_id': int,
    'predicted_class': str,
    'confidence': float,
    'class_probabilities': dict,
    'file_path': str,
    'ground_truth': str,
    'correct_prediction': bool
}
```

### 3. Batch Operations

#### Method: `list_available_test_ids(metadata_file)`

**Input:**
- `metadata_file` (str, optional): Path to metadata CSV file (default: `'lightkurve_test_metadata.csv'`)

**Output:**
- List of available Kepler IDs (list of int)

## Required Data Formats

### 1. Lightkurve CSV Files

**Location:** `lightkurve_data/kepler_{kepler_id}_lightkurve.csv`

**Required Columns:**
- `pdcsap_flux` (primary) OR `flux` (fallback): Main flux measurements
- Additional columns present but not required for prediction:
  - `flux_err`, `quality`, `timecorr`, `centroid_col`, `centroid_row`, etc.

**Data Preprocessing:**
- Automatic outlier removal (3-sigma threshold)
- Normalization (zero mean, unit variance)
- Time series length: Fixed at 3000 points (padded/truncated)
- Missing values: Automatically handled (dropna)

**Example CSV Structure:**
```csv
flux,flux_err,quality,timecorr,centroid_col,centroid_row,cadenceno,sap_flux,sap_flux_err,sap_bkg,sap_bkg_err,pdcsap_flux,pdcsap_flux_err,sap_quality,psf_centr1,psf_centr1_err,psf_centr2,psf_centr2_err,mom_centr1,mom_centr1_err,mom_centr2,mom_centr2_err,pos_corr1,pos_corr2,kepler_id
425586.3,96.70743,0,0.0028010276,230.3775326051085,369.78817115636906,168250,415918.75,89.9525,1857.2335,0.7526129,425586.3,96.70743,0,,,,,230.3775326051085,0.00020998163,369.78817115636906,0.00016315946,0.041989032,0.005541398,10000490
...
```

### 2. Test Metadata CSV

**Location:** `lightkurve_test_metadata.csv`

**Required Columns:**
- `kepid` (int): Kepler ID
- `koi_disposition` (str): Ground truth classification

**Possible Classification Values:**
- `"CANDIDATE"`: Potential exoplanet candidate
- `"FALSE POSITIVE"`: False positive detection

**Example Structure:**
```csv
kepid,koi_disposition
10904857,CANDIDATE
9652632,FALSE POSITIVE
6781535,FALSE POSITIVE
6362874,CANDIDATE
...
```

### 3. Model Metadata JSON

**Location:** `lightkurve_model_metadata.json`

**Required Fields:**
```json
{
    "class_names": ["FALSE POSITIVE", "CANDIDATE"],
    "input_shape": [3000, 1],
    "test_accuracy": 0.xxxx,
    "training_info": {...}
}
```

## Error Handling

### Common Error Responses

1. **File Not Found:**
```python
{
    'error': 'Lightkurve file not found: {file_path}',
    'kepler_id': int
}
```

2. **Data Loading Failed:**
```python
{
    'error': 'Failed to load data from: {file_path}',
    'kepler_id': int
}
```

3. **Prediction Failed:**
```python
{
    'error': 'Prediction failed: {error_message}',
    'kepler_id': int
}
```

## Model Input Requirements

### Tensor Shape
- **Input Shape:** `(batch_size, 3000, 1)`
- **Data Type:** `float32`
- **Preprocessing:** Normalized time series data

### Data Processing Pipeline
1. Load CSV file
2. Extract flux column (`pdcsap_flux` or `flux`)
3. Remove NaN values
4. Remove outliers (3-sigma clipping)
5. Normalize (zero mean, unit variance)
6. Pad or truncate to 3000 points
7. Reshape to `(1, 3000, 1)` for single prediction

## Command Line Interface

### Usage
```bash
python lightkurve_predictor.py <kepler_id> [options]
```

### Arguments
- `kepler_id` (required): Kepler ID to predict
- `--model`: Path to model file (default: `final_lightkurve_cnn_model.keras`)
- `--metadata`: Path to metadata file (default: `lightkurve_model_metadata.json`)
- `--data-path`: Path to data directory (default: `lightkurve_data/`)
- `--test-metadata`: Path to test metadata (default: `lightkurve_test_metadata.csv`)
- `--list-ids`: List available test IDs

### Example Commands
```bash
# Single prediction
python lightkurve_predictor.py 10904857

# List available IDs
python lightkurve_predictor.py 0 --list-ids

# Custom paths
python lightkurve_predictor.py 10904857 --model custom_model.keras --data-path /path/to/data/
```

## Integration Examples

### Python API Usage
```python
# Initialize predictor
predictor = LightkurvePredictor()

# Single prediction
result = predictor.predict_single(10904857)
print(f"Prediction: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.3f}")

# Prediction with ground truth
result = predictor.predict_with_metadata(10904857)
print(f"Correct: {result['correct_prediction']}")
print(f"Ground truth: {result['ground_truth']}")

# List available test IDs
test_ids = predictor.list_available_test_ids()
print(f"Available IDs: {len(test_ids)}")
```

### REST API Integration (Conceptual)
```python
# Expected POST request format
{
    "kepler_id": 10904857,
    "include_metadata": true,
    "metadata_file": "lightkurve_test_metadata.csv"
}

# Expected response format
{
    "kepler_id": 10904857,
    "predicted_class": "CANDIDATE",
    "confidence": 0.8456,
    "class_probabilities": {
        "FALSE POSITIVE": 0.1544,
        "CANDIDATE": 0.8456
    },
    "ground_truth": "CANDIDATE",
    "correct_prediction": true,
    "file_path": "lightkurve_data/kepler_10904857_lightkurve.csv"
}
```

## Performance Considerations

- **Time Series Length:** Fixed at 3000 points for optimal model performance
- **Memory Usage:** Each prediction requires ~12KB for input data
- **Processing Time:** Typically <1 second per prediction on modern hardware
- **Batch Processing:** Consider batching multiple predictions for efficiency

## Dependencies

### Required Python Packages
- `tensorflow` (>= 2.x)
- `pandas`
- `numpy` 
- `os`, `json`, `argparse` (standard library)

### System Requirements
- Python 3.7+
- Sufficient memory for model loading (~100MB typical)
- Storage for lightkurve CSV files (varies by dataset size)

---

*This document provides comprehensive specifications for integrating with the Lightkurve Exoplanet Detection system. For additional support or questions, refer to the example usage in `example_usage.py`.*