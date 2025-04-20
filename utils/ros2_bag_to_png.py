import rclpy
import rosbag2_py
import argparse
import os
import cv2
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
import numpy as np

def main(rosbag_path, topics, output_dir):
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(
        uri=rosbag_path,
        storage_id='mcap')
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader.open(storage_options, converter_options)

    # Get the list of topics from the bag file
    topic_types = reader.get_all_topics_and_types()
    if not topics:      
        print("Topics in the bag file:")
        for topic in topic_types:
            if topic.type == 'sensor_msgs/msg/Image':
                topics.append(topic.name)
                print(f"Added topic: {topic.name} to process")
    else:
        for topic in topics:
            if topic not in [t.name for t in topic_types]:
                print(f"Topic {topic} not found in the bag file. Removing from list.")
                topics.remove(topic)

    if not topics:
        print("No topics found in the bag file. Exiting.")
        return
    
    storage_filter = rosbag2_py.StorageFilter(topics=topics)
    reader.set_filter(storage_filter)

    # Check if the output directory exists, if not create it
    # Get name of the rosbag (last part of the path)
    bag_name = ''
    if rosbag_path.endswith('/'):
        rosbag_path = rosbag_path[:-1]
    if '/' in rosbag_path:
        bag_name = rosbag_path.split('/')[-1]
    if '.' in rosbag_path:
        bag_name = rosbag_path.split('.')[0]
    if rosbag_path.startswith('/'):
        bag_name = rosbag_path.split('/')[-1]
    if rosbag_path.endswith('.db3'):
        bag_name = rosbag_path.split('.db3')[0]
    if rosbag_path.endswith('.mcap'):
        bag_name = rosbag_path.split('.mcap')[0]
    print(f"Bag name: {bag_name}")

    # Create a directory for the png images
    output_path = os.path.join(output_dir, bag_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Initialize the CvBridge
    bridge = CvBridge()

    # Read messages from the bag file and convert to PNG
    print("Reading messages from the bag file...")

    # Define camera calibration parameters using the camera_info topic
    # K matrix (intrinsic camera matrix) from the 'k' field
    # [fx  0 cx]
    # [ 0 fy cy]
    # [ 0  0  1]
    fx = 533.0181026793728
    fy = 533.1950304735122
    cx = 484.917807487239
    cy = 309.95867583935154
    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0,  0,  1]], dtype=np.float32)

    # Distortion coefficients from the 'd' field [k1, k2, p1, p2, k3]
    k1 = -0.30854049349822915
    k2 = 0.08268565804376049
    p1 = 0.0005477275652276282
    p2 = -0.0003941952306063375
    k3 = 0.0
    dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

    while reader.has_next():
        topic, data, *_ = reader.read_next()
        # print(f"Received message on topic {topic}, data length: {len(data)}")
        if topic in topics:
            # Deserialize the message
            msg = deserialize_message(data, Image)
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding=msg.encoding)
            
            # Undistort the image to correct the "fish eye" effect
            undistorted_image = cv2.undistort(cv_image, camera_matrix, dist_coeffs)
            
            # Save the undistorted image
            image_name = os.path.join(
                output_path, 
                f"{topic.replace('/', '_')}_{msg.header.stamp.sec}_{msg.header.stamp.nanosec}.png"
            )

            cv2.imwrite(image_name, undistorted_image)
            print(f"Saved image to {image_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert ROS2 bag to PNG images.')
    parser.add_argument('--bagfile', type=str, help='Path to the ROS2 bag file', required=True)
    parser.add_argument('--topics', type=str, help='List of topics to process', default='')
    parser.add_argument('--output', type=str, help='Directory to save the PNG images - a subdir with name of the ROSbag will be created there', default='.')
    args = parser.parse_args()
    rosbag_path = args.bagfile
    topics = args.topics

    if topics:
        topics = [topic.strip() for topic in topics.split(',')]
    else:
        topics = []

    if not topics:
        print("No topics provided. Processing all topics.")
        topics = []
    else:
        print("Processing the following topics:")
        print(topics)
    output_dir = args.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    main(rosbag_path, topics, output_dir)

