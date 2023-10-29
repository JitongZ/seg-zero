SOURCE="blue eyes"
TARGET="black eyes"
INPUT_DIR="assets/test_images/faces"
INPUT_IMAGE_NAME="teacher"
EXP_NAME="test_teacher"

# python src/generate_sentences.py \
#     --source ${SOURCE} \
#     --target ${TARGET} \
#     --num_sentences 1000 \
#     --output_folder "assets/sentences/"

# python src/make_edit_direction.py \
#     --file_source_sentences "assets/sentences/${SOURCE}.txt" \
#     --file_target_sentences "assets/sentences/${TARGET}.txt" \
#     --output_folder assets/embeddings_sd_1.4

python src/inversion.py  \
        --input_image "${INPUT_DIR}/${INPUT_IMAGE_NAME}.jpg" \
        --results_folder "output/${EXP_NAME}/"

python src/edit_real.py \
    --inversion "output/${EXP_NAME}/inversion/${INPUT_IMAGE_NAME}.pt" \
    --prompt "output/${EXP_NAME}/prompt/${INPUT_IMAGE_NAME}.txt" \
    --task_name "${SOURCE}2${TARGET}" \
    --results_folder "output/${EXP_NAME}/" 
