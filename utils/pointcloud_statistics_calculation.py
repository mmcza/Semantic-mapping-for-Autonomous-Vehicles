import pandas as pd
import argparse
import os
import numpy as np

def calculate_pointcloud_statistics(file):
    df = pd.read_csv(file)
    filename = os.path.basename(file)
    
    # Filter out "No prediction" and "Unknown" for accuracy calculation
    valid_df = df[~df['predicted_class'].isin(['No prediction', 'Unknown'])]
    
    # Calculate overall accuracy
    correct_predictions = (valid_df['gt_class'] == valid_df['predicted_class']).sum()
    total_valid_points = len(valid_df)
    accuracy = correct_predictions / total_valid_points if total_valid_points > 0 else 0
    
    # Calculate accuracy for high-confidence predictions (ideal predictions)
    ideal_df = valid_df[valid_df['ideal_prediction'] == True]
    ideal_correct = (ideal_df['gt_class'] == ideal_df['predicted_class']).sum()
    total_ideal = len(ideal_df)
    ideal_accuracy = ideal_correct / total_ideal if total_ideal > 0 else 0
    
    # Per-class statistics
    class_stats = {}
    for class_name in df['gt_class'].unique():
        if class_name in ['No prediction', 'Unknown']:
            continue
            
        # Get points with this ground truth class
        class_df = valid_df[valid_df['gt_class'] == class_name]
        
        # Calculate metrics
        class_total = len(class_df)
        class_correct = (class_df['gt_class'] == class_df['predicted_class']).sum()
        class_accuracy = class_correct / class_total if class_total > 0 else 0
        
        # Precision and recall
        predicted_as_class = valid_df[valid_df['predicted_class'] == class_name]
        class_precision = class_correct / len(predicted_as_class) if len(predicted_as_class) > 0 else 0
        class_recall = class_correct / class_total if class_total > 0 else 0
        
        # F1 score
        class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0
        
        # IoU (Intersection over Union)
        # IoU = TP / (TP + FP + FN)
        true_positives = class_correct
        false_positives = len(predicted_as_class) - class_correct
        false_negatives = class_total - class_correct
        class_iou = true_positives / (true_positives + false_positives + false_negatives) if (true_positives + false_positives + false_negatives) > 0 else 0
        
        # Store stats
        class_stats[class_name] = {
            'accuracy': class_accuracy,
            'precision': class_precision,
            'recall': class_recall,
            'f1_score': class_f1,
            'iou': class_iou,
            'total_points': class_total
        }
    
    # Calculate mean IoU (mIoU) - a key metric for segmentation
    mean_iou = np.mean([stats['iou'] for stats in class_stats.values()])
    
    # Calculate weighted IoU (weighted by class frequency)
    total_points = sum([stats['total_points'] for stats in class_stats.values()])
    weighted_iou = sum([(stats['iou'] * stats['total_points']) / total_points for stats in class_stats.values()])
    
    # Calculate confusion matrix
    classes = sorted(list(set(valid_df['gt_class']) | set(valid_df['predicted_class'])))
    conf_matrix = pd.crosstab(valid_df['gt_class'], valid_df['predicted_class'], rownames=['Ground Truth'], colnames=['Predicted'])
    
    # No-prediction statistics
    no_pred_count = len(df[df['predicted_class'] == 'No prediction'])
    no_pred_rate = no_pred_count / len(df) if len(df) > 0 else 0
    
    # Unknown prediction statistics
    unknown_count = len(df[df['predicted_class'] == 'Unknown'])
    unknown_rate = unknown_count / len(df) if len(df) > 0 else 0
    
    # Create a dictionary with all the statistics
    stats = {
        'file': filename,
        'total_points': len(df),
        'valid_points': total_valid_points,
        'overall_accuracy': accuracy,
        'ideal_prediction_accuracy': ideal_accuracy,
        'mean_iou': mean_iou,
        'weighted_iou': weighted_iou,
        'no_prediction_rate': no_pred_rate,
        'unknown_prediction_rate': unknown_rate
    }
    
    # Add per-class statistics to the dictionary
    for class_name, class_data in class_stats.items():
        for metric, value in class_data.items():
            stats[f'{class_name}_{metric}'] = value
    
    # Save confusion matrix to a separate file
    conf_matrix.to_csv(os.path.join(args.output_dir, f"{os.path.splitext(filename)[0]}_confusion_matrix.csv"))
    
    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate statistics.")
    parser.add_argument("--results_dir", type=str, required=True, help="Path to directory with pointcloud results.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to directory to save the output.")
    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        print(f"Error: The file {args.results_dir} does not exist.")
        exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Define columns dynamically based on classes expected
    expected_classes = ["Building", "Car", "Grass", "Person", "Road", "Traffic_lights", "Tree"]
    metrics = ["accuracy", "precision", "recall", "f1_score", "iou", "total_points"]
    
    columns = ["file", "overall_accuracy", "ideal_prediction_accuracy", "mean_iou", "weighted_iou", 
               "no_prediction_rate", "unknown_prediction_rate"]
    
    # Add per-class metrics columns
    for class_name in expected_classes:
        for metric in metrics:
            columns.append(f"{class_name}_{metric}")
            
    results_df = pd.DataFrame(columns=columns)

    for file in os.listdir(args.results_dir):
        if file.endswith(".csv"):
            pointcloud_file = os.path.join(args.results_dir, file)
            print(f"Calculating statistics for {pointcloud_file}...")
            
            # Calculate statistics
            stats = calculate_pointcloud_statistics(pointcloud_file)
            
            # Add to results dataframe
            results_df = pd.concat([results_df, pd.DataFrame([stats])], ignore_index=True)
    
    # Save results to CSV
    results_path = os.path.join(args.output_dir, "pointcloud_statistics_summary.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Statistics saved to {results_path}")