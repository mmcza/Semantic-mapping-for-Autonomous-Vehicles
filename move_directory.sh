#!/bin/bash


src=~/Downloads/segmentation_results\ \(1\)\(1\)/segmentation_hf_b5

dst_base=~/Downloads/Semantic-mapping-for-Autonomous-Vehicles-main/annotations2


for file in "$src"/*.png; do
    filename=$(basename "$file")

    folder_name="${filename%%__*}"


    target_dir="$dst_base/$folder_name/visualizations"

    if [ -d "$target_dir" ]; then
        echo " Moving $filename → $target_dir"
        mv -f "$file" "$target_dir/"
    else
        echo "Folder not found: $target_dir"
    fi
done
