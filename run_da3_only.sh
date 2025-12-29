#!/bin/bash
# DA3 only comparison script
# Usage: bash run_da3_only.sh /path/to/mipnerf360

DATASET_ROOT="${1:-.}"
SCENES="${SCENES:-garden counter room bonsai bicycle stump kitchen}"
MODEL_NAME="${MODEL_NAME:-da3-large}"

echo "Dataset root: $DATASET_ROOT"
echo "Scenes: $SCENES"
echo "Model: $MODEL_NAME"
echo ""

mkdir -p logs

for scene in $SCENES; do
    SCENE_DIR="$DATASET_ROOT/$scene"

    if [ ! -d "$SCENE_DIR/images" ]; then
        echo "Skipping $scene: no images directory"
        continue
    fi

    echo "========================================"
    echo "Processing scene: $scene"
    echo "========================================"

    # 1. DA3 (baseline)
    echo "[1/2] Running DA3..."
    python demo_colmap_da3.py \
        --scene_dir "$SCENE_DIR" \
        --post_fix "_da3" \
        --model_name $MODEL_NAME \
        2>&1 | tee -a logs/da3_${scene}.log

    # 2. DA3 + GA
    echo "[2/2] Running DA3 + GA..."
    python demo_colmap_da3.py \
        --scene_dir "$SCENE_DIR" \
        --post_fix "_da3_ga" \
        --model_name $MODEL_NAME \
        --use_ga \
        2>&1 | tee -a logs/da3_ga_${scene}.log

    echo ""
done

echo "Done!"
