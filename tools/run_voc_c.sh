#!/usr/bin/env bash
# Run DESA-TTA on all 15 VOC-C corruptions. Each corruption is an
# independent stream that starts from the pretrained checkpoint.
#
# Usage: bash tools/run_voc_c.sh CHECKPOINT [WORK_DIR]
set -e

CHECKPOINT=${1:?"usage: bash tools/run_voc_c.sh CHECKPOINT [WORK_DIR]"}
WORK_DIR=${2:-work_dirs/voc_c}
CONFIG=configs/voc_c/desa_tta_yolo_world_s_voc_c.py

CORRUPTIONS="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur \
motion_blur zoom_blur snow frost fog brightness contrast elastic_transform \
pixelate jpeg_compression"

for C in ${CORRUPTIONS}; do
    echo "=== ${C} ==="
    PYTHONPATH="$(pwd):${PYTHONPATH}" python tools/test.py "${CONFIG}" "${CHECKPOINT}" \
        --work-dir "${WORK_DIR}/${C}" \
        --cfg-options test_dataloader.dataset.dataset.data_prefix.img="${C}/"
done

python tools/summarize_voc_c.py "${WORK_DIR}"
