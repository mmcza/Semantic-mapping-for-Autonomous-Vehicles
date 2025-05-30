import open3d as o3d
import argparse
import numpy as np
import pandas as pd
import os
import json

NEW_COLOR_MAP = {
    0: (0, 0, 0),        # Other
    1: (70, 130, 180),   # Sky
    2: (70, 70, 70),     # Building
    3: (152, 251, 152),  # Grass
    4: (128, 64, 128),   # Road
    5: (107, 142, 35),   # Tree
    6: (220, 220, 0),    # Traffic_lights
    7: (0, 0, 142),      # Car
    8: (255, 0, 0)       # Person
}

CLASS_NAMES = {
    0: "Other",
    1: "Sky",
    2: "Building",
    3: "Grass",
    4: "Road",
    5: "Tree",
    6: "Traffic_lights",
    7: "Car",
    8: "Person"
}

def main(ground_truth_dir, ground_truth_dataset_dir, predictions_dir, output_dir):
    predictions = [f for f in os.listdir(predictions_dir) if f.endswith('.pcd')]

    results = []

    for pred_file in predictions:
        print(f"Processing {pred_file} ...")
        pred_timestamp = pred_file.split('-')[1][:8]
        gt_filename = f"merged_pointcloud_{pred_timestamp}.pcd.json"
        gt_file_path = os.path.join(ground_truth_dataset_dir, "ann", gt_filename)
        if not os.path.exists(gt_file_path):
            print(f"Ground truth file {gt_file_path} does not exist. Skipping prediction {pred_file}.")
            continue

        pred_file_path = os.path.join(predictions_dir, pred_file)
        if not os.path.exists(pred_file_path):
            print(f"Prediction file {pred_file_path} does not exist. Skipping.")
            continue

        calculate_accuracy(pred_file_path, gt_file_path)

def calculate_accuracy(predicted_file, ground_truth_file):
    pred_pcd = o3d.io.read_point_cloud(predicted_file)

    gt_data = json.load(open(ground_truth_file, 'r'))
    cuboids = []
    for figure in gt_data['figures']:
        if figure['geometryType'] == 'cuboid_3d':
            class_name = None
            for obj in gt_data['objects']:
                if obj['key'] == figure['objectKey']:
                    class_name = obj['classTitle']
                    break
            if class_name is None:
                print(f"Warning: No class found for object key {figure['objectKey']} in ground truth data.")
                continue

            cuboid = {
                'geometry': figure['geometry'],
                'class': class_name
            }
            cuboids.append(cuboid)
    
    print(f"Found {len(cuboids)} cuboids in ground truth data.")

    if not cuboids:
        print("No cuboids found in ground truth data. Skipping accuracy calculation.")
        return

    df = pd.DataFrame(columns=['index', 'predicted_class', 'gt_class', 'ideal_prediction', 'distance_to_ideal'])

    for cuboid in cuboids:
        position = cuboid['geometry']['position']
        rotation = cuboid['geometry']['rotation']
        dimensions = cuboid['geometry']['dimensions']
        class_name = cuboid['class']

        position_np = np.array([position['x'], position['y'], position['z']])
        dimensions_np = np.array([dimensions['x'], dimensions['y'], dimensions['z']])
        if cuboid['class'] == "Traffic_lights" or cuboid['class'] == "Tree":
            dimensions_np[0] += 0.25
            dimensions_np[1] += 0.25
        R = o3d.geometry.get_rotation_matrix_from_xyz([
            rotation['x'],
            rotation['y'],
            rotation['z']
        ])

        obb = o3d.geometry.OrientedBoundingBox(
            center=position_np,
            R=R,
            extent=dimensions_np
        )

        points = np.asarray(pred_pcd.points)
        colors = np.asarray(pred_pcd.colors) if pred_pcd.has_colors() else None

        mask = obb.get_point_indices_within_bounding_box(pred_pcd.points)

        if colors is not None:
            colors_inside = colors[mask]
        points_inside = points[mask]

        if len(points_inside) == 0:
            print(f"No points found inside cuboid for class '{class_name}'.")
            continue

        print(f"Found {len(points_inside)} points inside the cuboid for class '{class_name}'.")

        for i, point_id in enumerate(mask):
            if point_id in df['index'].values:
                if df.loc[df['index'] == point_id, 'gt_class'].values[0] == class_name:
                    continue
                elif df.loc[df['index'] == point_id, 'predicted_class'].values[0] == df.loc[df['index'] == point_id, 'gt_class'].values[0]:
                    continue
            
            predicted_class = "Unknown"
            ideal_prediction = False
            if colors is not None:
                color = colors_inside[i]
                distance_to_ideal = np.sum(np.abs(color - np.array([1.0, 1.0, 1.0])))

                if color[0] > 0.99 and color[1] > 0.99 and color[2] > 0.99:
                    predicted_class = "No prediction"
                else:
                    min_diff = float('inf')
                    closest_class = None
                    
                    for class_idx, class_name_check in CLASS_NAMES.items():
                        class_color = np.array(NEW_COLOR_MAP[class_idx]) / 255.0
                        
                        color_diff = np.abs(color - class_color)
                        if np.sum(color_diff) < distance_to_ideal:
                            distance_to_ideal = np.sum(color_diff)
                        
                        if np.all(color_diff <= 0.02):
                            total_diff = np.sum(color_diff)
                            
                            if total_diff < min_diff:
                                min_diff = total_diff
                                closest_class = class_idx
                                
                                if (np.count_nonzero(color_diff > 0.004) <= 1 and 
                                    np.all(color_diff <= 0.004)):
                                    ideal_prediction = True
                    
                    if closest_class is not None:
                        predicted_class = CLASS_NAMES[closest_class]

            if point_id in df['index'].values:
                df.loc[df['index'] == point_id, 'predicted_class'] = predicted_class
                df.loc[df['index'] == point_id, 'gt_class'] = class_name
                df.loc[df['index'] == point_id, 'ideal_prediction'] = ideal_prediction
                df.loc[df['index'] == point_id, 'distance_to_ideal'] = distance_to_ideal
            else:
                new_row = pd.DataFrame({
                    'index': [point_id],
                    'gt_class': [class_name],
                    'predicted_class': [predicted_class],
                    'ideal_prediction': [ideal_prediction],
                    'distance_to_ideal': [distance_to_ideal]
                })
                df = pd.concat([df, new_row], ignore_index=True)

    # Save the DataFrame to a CSV file
    output_csv = os.path.join(output_dir, "results", f"{os.path.splitext(os.path.basename(predicted_file))[0]}_accuracy.csv")
    df.to_csv(output_csv, index=False)
    print(f"Accuracy results saved to {output_csv}")

    # Calculate and save summary statistics - confusion matrix 
    confusion_matrix = pd.crosstab(df['gt_class'], df['predicted_class'], rownames=['Ground Truth'], colnames=['Predicted'], margins=True)
    summary_csv = os.path.join(output_dir, "summary", f"{os.path.splitext(os.path.basename(predicted_file))[0]}_summary.csv")
    confusion_matrix.to_csv(summary_csv)
    print(f"Summary statistics saved to {summary_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate point cloud accuracy")
    parser.add_argument('--gt_base_dir', type=str, required=True, help='Path to ground truth directory')
    parser.add_argument('--gt_dataset', type=str, required=True, help='Dataset name for ground truth')
    parser.add_argument('--predictions_dir', type=str, required=True, help='Path to directory with predicted point clouds')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results')
    args = parser.parse_args()

    ground_truth_dir = args.gt_base_dir
    ground_truth_dataset_dir = os.path.join(ground_truth_dir, args.gt_dataset)
    predictions_dir = args.predictions_dir
    output_dir = args.output_dir
    if not os.path.exists(ground_truth_dir):
        raise FileNotFoundError(f"Ground truth directory {ground_truth_dir} does not exist.")
    if not os.path.exists(ground_truth_dataset_dir):
        raise FileNotFoundError(f"Ground truth dataset directory {ground_truth_dataset_dir} does not exist.")
    if not os.path.exists(predictions_dir):
        raise FileNotFoundError(f"Predictions directory {predictions_dir} does not exist.")
    os.makedirs(output_dir, exist_ok=True)
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    summary_dir = os.path.join(output_dir, 'summary')
    os.makedirs(summary_dir, exist_ok=True)

    main(ground_truth_dir, ground_truth_dataset_dir, predictions_dir, output_dir)