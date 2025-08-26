#!/usr/bin/env python3
import os
import json
import numpy as np
import argparse
import csv
from pathlib import Path
import open3d as o3d
from collections import Counter, defaultdict
import math

# Semantic class color map (RGB values 0-255)
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

# Class name map
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

class PointCloudProcessor:
    def __init__(self, base_dir, dataset_dir, pointcloud_name, second_pcd_path):
        """Initialize the processor with paths to pointclouds and annotations"""
        self.base_dir = Path(base_dir)
        self.dataset_dir = self.base_dir / dataset_dir
        self.pointcloud_name = pointcloud_name
        self.second_pcd_path = Path(second_pcd_path)
        
        # Set up paths
        self.meta_path = self.base_dir / "meta.json"
        self.pcd_dir = self.dataset_dir / "pointcloud"
        self.ann_dir = self.dataset_dir / "ann"
        
        # Load class metadata
        self.class_colors, self.class_names, self.class_ids_by_title = self.load_meta()
        
    def load_meta(self):
        """Load class metadata and create mappings for colors and titles"""
        with open(self.meta_path, 'r') as f:
            meta = json.load(f)
        
        class_colors = {}
        class_names = {}
        class_ids_by_title = {}
        
        for cls in meta.get('classes', []):
            if 'id' in cls and 'color' in cls and 'title' in cls:
                class_id = cls['id']
                class_title = cls['title']
                class_colors[class_id] = self.hex_to_rgb(cls['color'])
                class_names[class_id] = class_title
                class_ids_by_title[class_title] = class_id
                print(f"Loaded class: {class_title} (ID: {class_id}, Color: {cls['color']})")
        
        return class_colors, class_names, class_ids_by_title
    
    def hex_to_rgb(self, hex_color):
        """Convert hex color to RGB tuple (values 0-1)"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    
    def load_first_pointcloud(self):
        """Load the specified Supervisely point cloud and its annotations"""
        # Find the point cloud file
        pcd_path = self.pcd_dir / self.pointcloud_name
        if not pcd_path.exists() and not self.pointcloud_name.endswith('.pcd'):
            pcd_path = self.pcd_dir / f"{self.pointcloud_name}.pcd"
        
        # Make sure the point cloud exists
        if not pcd_path.exists():
            print(f"Error: Point cloud not found: {pcd_path}")
            return None, None
            
        # Find corresponding annotation file
        ann_path = self.ann_dir / f"{pcd_path.name}.json"
        if not ann_path.exists():
            print(f"Warning: No annotation file found for {pcd_path.name}")
            
        print(f"Loading point cloud: {pcd_path.name}")
        
        # Load point cloud
        try:
            pcd = o3d.io.read_point_cloud(str(pcd_path))
            if len(pcd.points) == 0:
                print(f"Warning: Point cloud is empty: {pcd_path}")
                return None, None
                
            print(f"Loaded point cloud with {len(pcd.points)} points")
        except Exception as e:
            print(f"Error loading point cloud: {e}")
            return None, None
        
        # Load annotations if available
        annotations = None
        if ann_path.exists():
            try:
                with open(ann_path, 'r') as f:
                    annotations = json.load(f)
                print(f"Loaded annotations")
            except Exception as e:
                print(f"Error loading annotations: {e}")
        
        return pcd, annotations
        
    def load_second_pointcloud(self):
        """Load the second point cloud"""
        if not self.second_pcd_path.exists():
            print(f"Error: Second point cloud not found: {self.second_pcd_path}")
            return None
            
        print(f"Loading second point cloud: {self.second_pcd_path}")
        
        try:
            pcd = o3d.io.read_point_cloud(str(self.second_pcd_path))
            if len(pcd.points) == 0:
                print(f"Warning: Second point cloud is empty: {self.second_pcd_path}")
                return None
                
            print(f"Loaded second point cloud with {len(pcd.points)} points")
            return pcd
        except Exception as e:
            print(f"Error loading second point cloud: {e}")
            return None
    
    def extract_point_colors(self, pcd):
        """Extract unique colors and count occurrences from the point cloud"""
        if not pcd.has_colors():
            print("Point cloud has no color information")
            return []
        
        # Get colors array (Nx3 array, RGB values from 0-1)
        colors = np.asarray(pcd.colors)
        
        # Round to 3 decimal places to reduce number of "unique" colors due to floating point precision
        colors_rounded = np.round(colors, 3)
        
        # Convert to tuples for counting
        color_tuples = [tuple(color) for color in colors_rounded]
        
        # Count occurrences of each color
        color_counts = Counter(color_tuples)
        
        # Convert to list of (color, count) tuples
        color_data = [(np.array(color), count) for color, count in color_counts.items()]
        
        return color_data
    
    def classify_points_by_semantic_class(self, pcd):
        """Classify each point by semantic class based on color with tolerance"""
        if not pcd.has_colors():
            print("Point cloud has no color information for classification")
            return None
        
        # Get colors array and convert to 0-255 range
        colors = np.asarray(pcd.colors) * 255
        
        # Initialize counters for each class
        class_counts = defaultdict(int)
        
        # Define tolerance
        tolerance = 10
        
        # Create a lookup array for all NEW_COLOR_MAP colors
        color_map_keys = list(NEW_COLOR_MAP.keys())
        color_map_values = np.array([NEW_COLOR_MAP[k] for k in color_map_keys])
        
        # Function to check if a color matches within tolerance
        def color_matches(color1, color2, tol):
            return (
                abs(color1[0] - color2[0]) <= tol and
                abs(color1[1] - color2[1]) <= tol and
                abs(color1[2] - color2[2]) <= tol
            )
        
        # Classify each point
        unclassified = 0
        for color in colors:
            # Initialize to unclassified
            classified = False
            
            # Check against each semantic class color
            for class_id, class_color in NEW_COLOR_MAP.items():
                if color_matches(color, class_color, tolerance):
                    class_counts[class_id] += 1
                    classified = True
                    break
            
            # Count unclassified points
            if not classified:
                unclassified += 1
        
        # Add unclassified count
        if unclassified > 0:
            class_counts[-1] = unclassified
        
        return class_counts
    
    def save_to_csv(self, color_data, output_path):
        """Save color data to CSV file"""
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['R', 'G', 'B', 'R (0-255)', 'G (0-255)', 'B (0-255)', 'Count', 'Percentage'])
            
            # Calculate total points
            total_points = sum(count for _, count in color_data)
            
            # Write data
            for color, count in color_data:
                percentage = (count / total_points) * 100
                rgb_8bit = tuple(int(c * 255) for c in color)
                writer.writerow([
                    f"{color[0]:.3f}", 
                    f"{color[1]:.3f}", 
                    f"{color[2]:.3f}",
                    rgb_8bit[0],
                    rgb_8bit[1],
                    rgb_8bit[2],
                    count,
                    f"{percentage:.2f}%"
                ])
                
        print(f"Color data saved to {output_path}")
    
    def save_semantic_counts_to_csv(self, class_counts, total_points, output_path):
        """Save semantic class counts to CSV file"""
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['Class ID', 'Class Name', 'R', 'G', 'B', 'Count', 'Percentage'])
            
            # Write data for each class
            for class_id, count in sorted(class_counts.items()):
                if class_id == -1:
                    # Handle unclassified points
                    class_name = "Unclassified"
                    color = (128, 128, 128)  # Gray for unclassified
                else:
                    class_name = CLASS_NAMES.get(class_id, f"Unknown-{class_id}")
                    color = NEW_COLOR_MAP.get(class_id, (0, 0, 0))
                
                percentage = (count / total_points) * 100
                
                writer.writerow([
                    class_id,
                    class_name,
                    color[0],
                    color[1],
                    color[2],
                    count,
                    f"{percentage:.2f}%"
                ])
                
        print(f"Semantic class counts saved to {output_path}")
        
    def euler_to_quaternion(self, roll, pitch, yaw):
        """Convert Euler angles to quaternion"""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return np.array([x, y, z, w])
    
    def is_point_inside_cuboid(self, point, position, dimensions, rotation):
        """
        Check if a point is inside a rotated cuboid
        
        Args:
            point: 3D point coordinates [x, y, z]
            position: Cuboid center position dict {'x', 'y', 'z'}
            dimensions: Cuboid dimensions dict {'x', 'y', 'z'}
            rotation: Cuboid rotation dict {'x', 'y', 'z'} (Euler angles)
        """
        # Convert position to numpy array
        cuboid_pos = np.array([
            position.get('x', 0), 
            position.get('y', 0), 
            position.get('z', 0)
        ])
        
        # Convert dimensions to numpy array
        cuboid_dim = np.array([
            dimensions.get('x', 1), 
            dimensions.get('y', 1), 
            dimensions.get('z', 1)
        ])
        
        # Convert Euler angles to quaternion
        rx = rotation.get('x', 0)
        ry = rotation.get('y', 0)
        rz = rotation.get('z', 0)
        q = self.euler_to_quaternion(ry, rx, rz)  # Note the order
        
        # Create rotation matrix from quaternion
        R = np.zeros((3, 3))
        R[0, 0] = 1 - 2 * (q[1]**2 + q[2]**2)
        R[0, 1] = 2 * (q[0] * q[1] - q[2] * q[3])
        R[0, 2] = 2 * (q[0] * q[2] + q[1] * q[3])
        R[1, 0] = 2 * (q[0] * q[1] + q[2] * q[3])
        R[1, 1] = 1 - 2 * (q[0]**2 + q[2]**2)
        R[1, 2] = 2 * (q[1] * q[2] - q[0] * q[3])
        R[2, 0] = 2 * (q[0] * q[2] - q[1] * q[3])
        R[2, 1] = 2 * (q[1] * q[2] + q[0] * q[3])
        R[2, 2] = 1 - 2 * (q[0]**2 + q[1]**2)
        
        # Translate point relative to cuboid center
        translated = point - cuboid_pos
        
        # Apply inverse rotation to bring point to cuboid's local frame
        local_point = np.dot(R.T, translated)
        
        # Check if point is inside cuboid bounds
        half_width = cuboid_dim[0] / 2
        half_length = cuboid_dim[1] / 2
        half_height = cuboid_dim[2] / 2
        
        return (
            abs(local_point[0]) <= half_width and
            abs(local_point[1]) <= half_length and
            abs(local_point[2]) <= half_height
        )
    
    def extract_cuboids_from_annotations(self, annotations, object_to_class):
        """Extract cuboid information from annotations"""
        cuboids = []
        
        for figure in annotations.get("figures", []):
            object_key = figure.get("objectKey")
            if not object_key or object_key not in object_to_class:
                continue
                
            class_title, class_id = object_to_class[object_key]
            
            # Check if this is a cuboid
            if figure.get("geometryType") != "cuboid_3d":
                continue
                
            # In Supervisely format, the cuboid data is directly in the figure's geometry field
            # NOT nested in a type field
            position = figure.get("geometry", {}).get("position", {})
            dimensions = figure.get("geometry", {}).get("dimensions", {})
            rotation = figure.get("geometry", {}).get("rotation", {})
            
            print(f"Found cuboid: {class_title} (ID: {class_id})")
            print(f"  Position: {position}")
            print(f"  Dimensions: {dimensions}")
            print(f"  Rotation: {rotation}")
            
            # Add to cuboids list
            cuboids.append({
                "position": position,
                "dimensions": dimensions,
                "rotation": rotation,
                "class_id": class_id,
                "class_title": class_title
            })
        
        print(f"Found {len(cuboids)} cuboids")
        return cuboids
    
    def analyze_point_cuboid_classes(self, pcd, cuboids):
        """Analyze points and cuboids to create a confusion matrix"""
        # Initialize confusion matrix
        points_inside_any_cuboid = 0
        confusion_matrix = defaultdict(lambda: defaultdict(int))
        
        # Get point coordinates and colors
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) * 255
        
        # Define tolerance for color matching
        tolerance = 10
        
        # Function to check if a color matches within tolerance
        def color_matches(color1, color2, tol):
            return (
                abs(color1[0] - color2[0]) <= tol and
                abs(color1[1] - color2[1]) <= tol and
                abs(color1[2] - color2[2]) <= tol
            )
        
        # Process each point
        skipped_points = 0
        processed_points = 0
        
        print(f"Processing {len(points)} points against {len(cuboids)} cuboids...")
        
        for i, (point, color) in enumerate(zip(points, colors)):
            if i % 10000 == 0:
                print(f"Processed {i}/{len(points)} points...")
                
            # Classify point by color
            point_class = -1  # Default to unclassified
            for class_id, class_color in NEW_COLOR_MAP.items():
                if color_matches(color, class_color, tolerance):
                    point_class = class_id
                    break
            
            # Check if point is inside any cuboid
            point_cuboids = []
            for cuboid in cuboids:
                if self.is_point_inside_cuboid(
                    point, 
                    cuboid["position"], 
                    cuboid["dimensions"], 
                    cuboid["rotation"]
                ):
                    point_cuboids.append(cuboid["class_id"])
            
            # Determine cuboid class based on rules
            if not point_cuboids:
                # No cuboid contains this point
                cuboid_class = -1  # Unclassified
                confusion_matrix[point_class][cuboid_class] += 1
                processed_points += 1
            else:
                # Check if any cuboid has matching class
                if point_class in point_cuboids:
                    points_inside_any_cuboid += 1
                    # At least one cuboid has matching class
                    cuboid_class = point_class
                    confusion_matrix[point_class][cuboid_class] += 1
                    processed_points += 1
                else:
                    # No cuboid has matching class - skip this point
                    skipped_points += 1
            print(f"Points inside any cuboid: {points_inside_any_cuboid} out of {len(points)}")
        
        print(f"Processed {processed_points} points, skipped {skipped_points} points")
        return confusion_matrix
    
    def save_confusion_matrix_to_csv(self, confusion_matrix, total_points, output_path):
        """Save confusion matrix to CSV file"""
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Get all unique class IDs
            point_classes = set(confusion_matrix.keys())
            cuboid_classes = set()
            for pc in confusion_matrix:
                cuboid_classes.update(confusion_matrix[pc].keys())
            
            all_classes = sorted(list(point_classes.union(cuboid_classes)))
            
            # Write header row
            header = ['Point Class \\ Cuboid Class']
            for cls in all_classes:
                header.append(CLASS_NAMES.get(cls, f"Unknown-{cls}"))
            header.append('Total')
            writer.writerow(header)
            
            # Write data rows
            for point_class in sorted(list(point_classes)):
                row = [CLASS_NAMES.get(point_class, f"Unknown-{point_class}")]
                    
                # Add counts for each cuboid class
                row_total = 0
                for cuboid_class in all_classes:
                    count = confusion_matrix[point_class].get(cuboid_class, 0)
                    row.append(count)
                    row_total += count
                
                # Add row total
                row.append(row_total)
                writer.writerow(row)
            
            # Write column totals
            col_totals = ['Total']
            total_sum = 0
            for cuboid_class in all_classes:
                col_total = sum(confusion_matrix[pc].get(cuboid_class, 0) for pc in point_classes)
                col_totals.append(col_total)
                total_sum += col_total
            col_totals.append(total_sum)
            writer.writerow(col_totals)
            
        print(f"Confusion matrix saved to {output_path}")
    
    def process(self):
        """Process both point clouds and analyze colors"""
        # Load first point cloud with annotations
        first_pcd, annotations = self.load_first_pointcloud()
        if first_pcd is None:
            print("Failed to load first point cloud, exiting")
            return False
            
        print(f"Successfully loaded first point cloud: {self.pointcloud_name}")
        
        # Create object to class mapping
        object_to_class = {}
        if annotations:
            # Count objects in annotations
            object_count = len(annotations.get("objects", []))
            figure_count = len(annotations.get("figures", []))
            print(f"Annotations contain {object_count} objects and {figure_count} figures")
            
            # Create a map from objectKey to classTitle and classId
            for obj in annotations.get("objects", []):
                object_key = obj.get("key")
                class_title = obj.get("classTitle")
                if object_key and class_title:
                    class_id = self.class_ids_by_title.get(class_title)
                    if class_id:
                        object_to_class[object_key] = (class_title, class_id)
            
            # Extract cuboids from annotations
            cuboids = self.extract_cuboids_from_annotations(annotations, object_to_class)
            print(f"Extracted {len(cuboids)} cuboids from annotations")
        else:
            print("No annotations available")
            cuboids = []
        
        # Load second point cloud
        second_pcd = self.load_second_pointcloud()
        if second_pcd is None:
            print("Failed to load second point cloud, exiting")
            return False
            
        # Extract colors and counts from second point cloud
        color_data = self.extract_point_colors(second_pcd)
        
        if not color_data:
            print("No color information found in the second point cloud")
            return True
            
        # Sort by count (ascending)
        color_data.sort(key=lambda x: x[1])
        
        # Generate output filename for color analysis
        output_csv = self.second_pcd_path.with_suffix('.colors.csv')
        
        # Save color data to CSV
        self.save_to_csv(color_data, output_csv)
        
        # Calculate total points
        total_points = len(second_pcd.points)
        
        # Print results
        print("\n--- Second Point Cloud Color Analysis ---")
        print(f"Total points: {total_points}")
        print(f"Number of unique colors: {len(color_data)}")
        
        print("\nTop 10 most common colors:")
        for color, count in reversed(color_data[-10:]):
            percentage = (count / total_points) * 100
            rgb_8bit = tuple(int(c * 255) for c in color)
            print(f"RGB{rgb_8bit} ({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f}): {count} points ({percentage:.2f}%)")
        
        print(f"\nFull color analysis saved to {output_csv}")
        
        # Perform semantic classification
        print("\n--- Semantic Class Analysis (±10 tolerance) ---")
        class_counts = self.classify_points_by_semantic_class(second_pcd)
        
        if class_counts:
            # Generate output filename for semantic analysis
            semantic_csv = self.second_pcd_path.with_suffix('.semantic.csv')
            
            # Save semantic counts to CSV
            self.save_semantic_counts_to_csv(class_counts, total_points, semantic_csv)
            
            # Print semantic class distribution
            print(f"Semantic class distribution (total: {total_points} points):")
            for class_id, count in sorted(class_counts.items()):
                if class_id == -1:
                    class_name = "Unclassified"
                else:
                    class_name = CLASS_NAMES.get(class_id, f"Unknown-{class_id}")
                    
                percentage = (count / total_points) * 100
                print(f"  {class_name} (ID: {class_id}): {count} points ({percentage:.2f}%)")
            
            print(f"\nSemantic class analysis saved to {semantic_csv}")
        else:
            print("Could not perform semantic classification")
        
        # Perform point-cuboid class analysis if we have cuboids
        if cuboids:
            print("\n--- Point-Cuboid Class Analysis ---")
            print("Analyzing which points are inside cuboids and comparing classes...")
            
            # Create confusion matrix of point class vs cuboid class
            confusion_matrix = self.analyze_point_cuboid_classes(second_pcd, cuboids)
            
            # Save confusion matrix to CSV
            cuboid_csv = self.second_pcd_path.with_suffix('.cuboid_classes.csv')
            self.save_confusion_matrix_to_csv(confusion_matrix, total_points, cuboid_csv)
            
            # Calculate some statistics
            total_analyzed = sum(sum(counts.values()) for counts in confusion_matrix.values())
            print(f"Analyzed {total_analyzed} points out of {total_points} total points")
            print(f"Points-in-cuboid analysis saved to {cuboid_csv}")
            
        return True

def main():
    parser = argparse.ArgumentParser(description='Process Supervisely point clouds and extract colors')
    parser.add_argument('--base_dir', type=str, default='.', 
                        help='Base directory containing meta.json')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Dataset directory name under base_dir')
    parser.add_argument('--pointcloud_name', type=str, required=True,
                        help='Name of the first point cloud with annotations')
    parser.add_argument('--second_pcd', type=str, required=True,
                        help='Path to the second point cloud file')
    
    args = parser.parse_args()
    
    processor = PointCloudProcessor(
        args.base_dir,
        args.dataset_dir,
        args.pointcloud_name,
        args.second_pcd
    )
    
    processor.process()

if __name__ == '__main__':
    main()