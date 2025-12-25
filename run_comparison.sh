#!/bin/bash
# Comparison script: VGGT vs DA3 (with/without GA)
# Usage: bash run_comparison.sh /path/to/mipnerf360

DATASET_ROOT="${1:-.}"
SCENES="garden counter room bonsai bicycle stump kitchen"

echo "Dataset root: $DATASET_ROOT"
echo "Scenes: $SCENES"
echo "Full evaluation (all frames)"
echo ""

for scene in $SCENES; do
    SCENE_DIR="$DATASET_ROOT/$scene"

    if [ ! -d "$SCENE_DIR/images" ]; then
        echo "Skipping $scene: no images directory"
        continue
    fi

    echo "========================================"
    echo "Processing scene: $scene"
    echo "========================================"

    # 1. VGGT (baseline)
    echo "[1/4] Running VGGT..."
    python demo_colmap.py \
        --scene_dir "$SCENE_DIR" \
        --post_fix "_vggt" \
        2>&1 | tee -a logs/vggt_${scene}.log

    # 2. VGGT + GA
    echo "[2/4] Running VGGT + GA..."
    python demo_colmap.py \
        --scene_dir "$SCENE_DIR" \
        --post_fix "_vggt_ga" \
        --use_ga \
        2>&1 | tee -a logs/vggt_ga_${scene}.log

    # 3. DA3 (baseline)
    echo "[3/4] Running DA3..."
    python demo_colmap_da3.py \
        --scene_dir "$SCENE_DIR" \
        --post_fix "_da3" \
        --model_name da3metric-large \
        2>&1 | tee -a logs/da3_${scene}.log

    # 4. DA3 + GA
    echo "[4/4] Running DA3 + GA..."
    python demo_colmap_da3.py \
        --scene_dir "$SCENE_DIR" \
        --post_fix "_da3_ga" \
        --model_name da3metric-large \
        --use_ga \
        2>&1 | tee -a logs/da3_ga_${scene}.log

    echo ""
done

# Collect results
echo "========================================"
echo "Collecting results..."
echo "========================================"

echo ""
echo "=== Evaluation Results ==="
echo ""
printf "%-12s | %-20s | %-20s | %-20s | %-20s\n" "Scene" "VGGT" "VGGT+GA" "DA3" "DA3+GA"
echo "-------------|----------------------|----------------------|----------------------|----------------------"

for scene in $SCENES; do
    vggt_result=$(cat "${DATASET_ROOT}_vggt/$scene/results.txt" 2>/dev/null | grep -E "AUC|Acc" | head -1 || echo "N/A")
    vggt_ga_result=$(cat "${DATASET_ROOT}_vggt_ga/$scene/results.txt" 2>/dev/null | grep -E "AUC|Acc" | head -1 || echo "N/A")
    da3_result=$(cat "${DATASET_ROOT}_da3/$scene/results.txt" 2>/dev/null | grep -E "AUC|Acc" | head -1 || echo "N/A")
    da3_ga_result=$(cat "${DATASET_ROOT}_da3_ga/$scene/results.txt" 2>/dev/null | grep -E "AUC|Acc" | head -1 || echo "N/A")

    printf "%-12s | %-20s | %-20s | %-20s | %-20s\n" "$scene" "$vggt_result" "$vggt_ga_result" "$da3_result" "$da3_ga_result"
done

echo ""
echo "Done! Check logs/ directory for detailed outputs."
