#! /bin/bash  # employ bash shell

OBJECT="bg_plate_new" # "bg_raw" "bg_tablecloth" "bg_colorful_mat" "bg_large_plate" "bg_mid_plate" "bg_small_plate"
SPARSE_PATH="/data/exp_obj_data/${OBJECT}/sparse"

mkdir "$SPARSE_PATH"

SCAN_PATH="/data/exp_obj_data/${OBJECT}/scan1.db"
IMAGE_PATH="/data/exp_obj_data/${OBJECT}/${OBJECT}_sharp"

colmap feature_extractor \
    --database_path "$SCAN_PATH" \
    --image_path "$IMAGE_PATH" \
    --ImageReader.camera_model PINHOLE


OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES=2 colmap exhaustive_matcher --database_path "$SCAN_PATH"


OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES=2 colmap mapper \
    --database_path "$SCAN_PATH" \
    --image_path "$IMAGE_PATH" \
    --output_path "$SPARSE_PATH"

INPUT_PATH="${SPARSE_PATH}/0"
OUTPUT_PLY_PATH="${SPARSE_PATH}/points3D.ply"

OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES=2 colmap model_converter \
    --input_path "$INPUT_PATH" \
    --output_path "$OUTPUT_PLY_PATH" \
    --output_type PLY

OUTPUT_DENSE_PATH="/data/exp_obj_data/${OBJECT}/dense"

colmap image_undistorter --image_path "$IMAGE_PATH" --input_path "$INPUT_PATH" --output_path "$OUTPUT_DENSE_PATH"