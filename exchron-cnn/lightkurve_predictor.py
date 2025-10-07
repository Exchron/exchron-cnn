"""
Lightkurve Exoplanet Detection Predictor

This script loads a trained CNN model and makes predictions for individual Kepler objects
using their lightkurve data. It includes functions to retrieve data from metadata files
and process lightkurve CSV files for prediction.
"""

import pandas as pd
import numpy as np
import os
import json
import tensorflow as tf
from tensorflow import keras
import argparse
import warnings
warnings.filterwarnings('ignore')


class LightkurvePredictor:
    def __init__(self, model_path='final_lightkurve_cnn_model.keras', 
                 metadata_path='lightkurve_model_metadata.json',
                 lightkurve_data_path='lightkurve_data/'):
        """
        Initialize the predictor with model and metadata paths.
        
        Args:
            model_path (str): Path to the trained Keras model
            metadata_path (str): Path to the model metadata JSON file
            lightkurve_data_path (str): Path to the directory containing lightkurve CSV files
        """
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.lightkurve_data_path = lightkurve_data_path
        
        # Load model and metadata
        self.model = None
        self.metadata = None
        self.class_names = None
        self.input_shape = None
        
        self._load_model()
        self._load_metadata()
    
    def _load_model(self):
        """Load the trained CNN model."""
        try:
            self.model = keras.models.load_model(self.model_path)
            print(f"Model loaded successfully from: {self.model_path}")
        except Exception as e:
            raise Exception(f"Error loading model from {self.model_path}: {e}")
    
    def _load_metadata(self):
        """Load model metadata from JSON file."""
        try:
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            self.class_names = self.metadata['class_names']
            self.input_shape = tuple(self.metadata['input_shape'])
            
            print(f"Metadata loaded successfully from: {self.metadata_path}")
            print(f"Classes: {self.class_names}")
            print(f"Input shape: {self.input_shape}")
            
        except Exception as e:
            raise Exception(f"Error loading metadata from {self.metadata_path}: {e}")
    
    def load_lightkurve_data(self, file_path, max_length=3000):
        """
        Load and preprocess a single lightkurve CSV file.
        Returns normalized flux data as a time series.
        
        Args:
            file_path (str): Path to the lightkurve CSV file
            max_length (int): Maximum length of the time series (default: 3000)
            
        Returns:
            numpy.ndarray: Preprocessed flux data, or None if loading fails
        """
        try:
            df = pd.read_csv(file_path)
            
            # Use the main flux column (either 'flux' or 'pdcsap_flux')
            if 'pdcsap_flux' in df.columns:
                flux_data = df['pdcsap_flux'].dropna().values
            elif 'flux' in df.columns:
                flux_data = df['flux'].dropna().values
            else:
                print(f"Warning: No flux column found in {file_path}")
                return None
            
            # Remove outliers (simple method: remove values beyond 3 sigma)
            mean_flux = np.mean(flux_data)
            std_flux = np.std(flux_data)
            flux_data = flux_data[np.abs(flux_data - mean_flux) < 3 * std_flux]
            
            # Normalize the flux data
            if len(flux_data) > 0:
                flux_data = (flux_data - np.mean(flux_data)) / np.std(flux_data)
            
            # Pad or truncate to fixed length
            if len(flux_data) > max_length:
                flux_data = flux_data[:max_length]
            elif len(flux_data) < max_length:
                flux_data = np.pad(flux_data, (0, max_length - len(flux_data)), 
                                 mode='constant', constant_values=0)
            
            return flux_data
        
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def get_data_from_metadata(self, kepler_id, metadata_file='lightkurve_test_metadata.csv'):
        """
        Retrieve disposition information for a given Kepler ID from metadata file.
        
        Args:
            kepler_id (int): The Kepler ID to search for
            metadata_file (str): Path to the metadata CSV file
            
        Returns:
            dict: Dictionary containing kepid and koi_disposition, or None if not found
        """
        try:
            metadata_df = pd.read_csv(metadata_file)
            
            # Search for the Kepler ID
            result = metadata_df[metadata_df['kepid'] == kepler_id]
            
            if len(result) == 0:
                print(f"Kepler ID {kepler_id} not found in {metadata_file}")
                return None
            
            record = result.iloc[0]
            return {
                'kepid': int(record['kepid']),
                'koi_disposition': record['koi_disposition']
            }
            
        except Exception as e:
            print(f"Error reading metadata from {metadata_file}: {e}")
            return None
    
    def predict_single(self, kepler_id):
        """
        Make a prediction for a single Kepler object.
        
        Args:
            kepler_id (int): The Kepler ID to predict
            
        Returns:
            dict: Dictionary containing prediction results
        """
        # Construct the file path
        filename = f"kepler_{kepler_id}_lightkurve.csv"
        file_path = os.path.join(self.lightkurve_data_path, filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            return {
                'error': f"Lightkurve file not found: {file_path}",
                'kepler_id': kepler_id
            }
        
        # Load and preprocess the data
        flux_data = self.load_lightkurve_data(file_path)
        
        if flux_data is None:
            return {
                'error': f"Failed to load data from: {file_path}",
                'kepler_id': kepler_id
            }
        
        # Reshape for model input (add batch and channel dimensions)
        input_data = flux_data.reshape(1, self.input_shape[0], self.input_shape[1])
        
        # Make prediction
        try:
            predictions = self.model.predict(input_data, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = self.class_names[predicted_class_idx]
            confidence = float(predictions[0][predicted_class_idx])
            
            # Get probabilities for all classes
            class_probabilities = {}
            for i, class_name in enumerate(self.class_names):
                class_probabilities[class_name] = float(predictions[0][i])
            
            return {
                'kepler_id': kepler_id,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'class_probabilities': class_probabilities,
                'file_path': file_path
            }
            
        except Exception as e:
            return {
                'error': f"Prediction failed: {e}",
                'kepler_id': kepler_id
            }
    
    def predict_with_metadata(self, kepler_id, metadata_file='lightkurve_test_metadata.csv'):
        """
        Make a prediction and compare with ground truth from metadata.
        
        Args:
            kepler_id (int): The Kepler ID to predict
            metadata_file (str): Path to the metadata CSV file
            
        Returns:
            dict: Dictionary containing prediction results and ground truth
        """
        # Get prediction
        prediction_result = self.predict_single(kepler_id)
        
        # Get ground truth from metadata
        metadata_info = self.get_data_from_metadata(kepler_id, metadata_file)
        
        # Combine results
        result = prediction_result.copy()
        if metadata_info:
            result['ground_truth'] = metadata_info['koi_disposition']
            result['correct_prediction'] = (
                result.get('predicted_class') == metadata_info['koi_disposition']
            )
        else:
            result['ground_truth'] = 'Unknown'
            result['correct_prediction'] = None
        
        return result
    
    def list_available_test_ids(self, metadata_file='lightkurve_test_metadata.csv'):
        """
        List all available Kepler IDs in the test metadata file.
        
        Args:
            metadata_file (str): Path to the metadata CSV file
            
        Returns:
            list: List of available Kepler IDs
        """
        try:
            metadata_df = pd.read_csv(metadata_file)
            return metadata_df['kepid'].tolist()
        except Exception as e:
            print(f"Error reading metadata file: {e}")
            return []
    
    def get_model_info(self):
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information
        """
        return {
            'model_path': self.model_path,
            'metadata': self.metadata,
            'input_shape': self.input_shape,
            'class_names': self.class_names
        }


def main():
    """Command line interface for the predictor."""
    parser = argparse.ArgumentParser(description='Predict exoplanet disposition using trained CNN model')
    parser.add_argument('kepler_id', type=int, help='Kepler ID to predict')
    parser.add_argument('--model', type=str, default='final_lightkurve_cnn_model.keras',
                       help='Path to the trained model file')
    parser.add_argument('--metadata', type=str, default='lightkurve_model_metadata.json',
                       help='Path to the model metadata file')
    parser.add_argument('--data-path', type=str, default='lightkurve_data/',
                       help='Path to the lightkurve data directory')
    parser.add_argument('--test-metadata', type=str, default='lightkurve_test_metadata.csv',
                       help='Path to the test metadata file')
    parser.add_argument('--list-ids', action='store_true',
                       help='List all available test Kepler IDs')
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = LightkurvePredictor(
            model_path=args.model,
            metadata_path=args.metadata,
            lightkurve_data_path=args.data_path
        )
        
        if args.list_ids:
            print("\nAvailable Kepler IDs in test metadata:")
            ids = predictor.list_available_test_ids(args.test_metadata)
            for i, kid in enumerate(ids[:20]):  # Show first 20
                print(f"  {kid}")
            if len(ids) > 20:
                print(f"  ... and {len(ids) - 20} more")
            print(f"\nTotal: {len(ids)} test IDs available")
            return
        
        # Make prediction
        print(f"\nMaking prediction for Kepler ID: {args.kepler_id}")
        print("=" * 50)
        
        result = predictor.predict_with_metadata(args.kepler_id, args.test_metadata)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            return
        
        # Display results
        print(f"Kepler ID: {result['kepler_id']}")
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Ground Truth: {result['ground_truth']}")
        
        if result['correct_prediction'] is not None:
            status = "✓ CORRECT" if result['correct_prediction'] else "✗ INCORRECT"
            print(f"Prediction Status: {status}")
        
        print("\nClass Probabilities:")
        for class_name, prob in result['class_probabilities'].items():
            print(f"  {class_name}: {prob:.4f}")
        
        print(f"\nData file: {result['file_path']}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()