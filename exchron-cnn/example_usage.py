"""
Example usage of the Lightkurve Predictor

This script demonstrates how to use the LightkurvePredictor class
to make predictions for individual Kepler objects.
"""

from lightkurve_predictor import LightkurvePredictor
import pandas as pd

def example_usage():
    """Demonstrate the usage of LightkurvePredictor."""
    
    print("Lightkurve Exoplanet Predictor - Example Usage")
    print("=" * 50)
    
    # Initialize the predictor
    try:
        predictor = LightkurvePredictor()
        print("✓ Predictor initialized successfully\n")
    except Exception as e:
        print(f"✗ Error initializing predictor: {e}")
        return
    
    # Show model information
    model_info = predictor.get_model_info()
    print("Model Information:")
    print(f"  Classes: {model_info['class_names']}")
    print(f"  Input shape: {model_info['input_shape']}")
    print(f"  Test accuracy: {model_info['metadata']['test_accuracy']:.4f}")
    print()
    
    # Get some test IDs
    test_ids = predictor.list_available_test_ids()
    if not test_ids:
        print("✗ No test IDs found in metadata file")
        return
    
    print(f"Found {len(test_ids)} test samples")
    print(f"First 10 test IDs: {test_ids[:10]}")
    print()
    
    # Make predictions for a few sample IDs
    sample_ids = test_ids[:5]  # Use first 5 IDs
    
    print("Making predictions for sample IDs:")
    print("-" * 40)
    
    correct_predictions = 0
    total_predictions = 0
    
    for kepler_id in sample_ids:
        print(f"\nKepler ID: {kepler_id}")
        
        # Make prediction with metadata comparison
        result = predictor.predict_with_metadata(kepler_id)
        
        if 'error' in result:
            print(f"  ✗ Error: {result['error']}")
            continue
        
        total_predictions += 1
        
        print(f"  Predicted: {result['predicted_class']} (confidence: {result['confidence']:.3f})")
        print(f"  Ground truth: {result['ground_truth']}")
        
        if result['correct_prediction']:
            print(f"  ✓ CORRECT")
            correct_predictions += 1
        else:
            print(f"  ✗ INCORRECT")
        
        # Show class probabilities
        print("  Probabilities:")
        for class_name, prob in result['class_probabilities'].items():
            print(f"    {class_name}: {prob:.3f}")
    
    # Summary
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print(f"\nSample Prediction Summary:")
        print(f"  Correct: {correct_predictions}/{total_predictions}")
        print(f"  Accuracy: {accuracy:.3f}")


def predict_specific_id(kepler_id):
    """Predict for a specific Kepler ID."""
    
    print(f"Predicting for Kepler ID: {kepler_id}")
    print("=" * 30)
    
    try:
        predictor = LightkurvePredictor()
        
        # Get metadata info
        metadata_info = predictor.get_data_from_metadata(kepler_id)
        if metadata_info:
            print(f"Found in metadata: {metadata_info['koi_disposition']}")
        else:
            print("Not found in test metadata (may be in training set)")
        
        # Make prediction
        result = predictor.predict_single(kepler_id)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            return
        
        print(f"\nPrediction Results:")
        print(f"  Predicted class: {result['predicted_class']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  File: {result['file_path']}")
        
        print(f"\nClass probabilities:")
        for class_name, prob in result['class_probabilities'].items():
            print(f"  {class_name}: {prob:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")


def analyze_test_set_performance():
    """Analyze performance on the entire test set."""
    
    print("Analyzing Test Set Performance")
    print("=" * 35)
    
    try:
        predictor = LightkurvePredictor()
        
        # Load test metadata
        test_metadata = pd.read_csv('lightkurve_test_metadata.csv')
        print(f"Test set size: {len(test_metadata)}")
        
        # Analyze distribution
        disposition_counts = test_metadata['koi_disposition'].value_counts()
        print("\nTest set distribution:")
        for disposition, count in disposition_counts.items():
            print(f"  {disposition}: {count} ({count/len(test_metadata)*100:.1f}%)")
        
        # Sample predictions (first 10 to avoid long runtime)
        sample_size = min(10, len(test_metadata))
        sample_ids = test_metadata['kepid'].iloc[:sample_size].tolist()
        
        print(f"\nTesting on {sample_size} samples:")
        print("-" * 30)
        
        results = []
        for kepler_id in sample_ids:
            result = predictor.predict_with_metadata(kepler_id)
            if 'error' not in result:
                results.append(result)
                status = "✓" if result['correct_prediction'] else "✗"
                print(f"{status} {kepler_id}: {result['predicted_class']} (gt: {result['ground_truth']})")
        
        # Calculate accuracy
        if results:
            correct = sum(1 for r in results if r['correct_prediction'])
            accuracy = correct / len(results)
            print(f"\nSample accuracy: {correct}/{len(results)} = {accuracy:.3f}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Run example usage
    example_usage()
    
    print("\n" + "="*60 + "\n")
    
    # Predict for a specific ID (you can change this)
    predict_specific_id(10904857)  # First ID from test metadata
    
    print("\n" + "="*60 + "\n")
    
    # Analyze test set performance
    analyze_test_set_performance()