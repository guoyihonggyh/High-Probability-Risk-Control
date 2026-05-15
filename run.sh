

# #!/bin/bash

# set -e

# SCRIPT="opo_medical.py"
# OUT_DIR=""
# BASELINE="none"
# MAX_JOBS=10
# NUM_RUNS=500

# mkdir -p "${OUT_DIR}"

# for ((seed=51; seed<NUM_RUNS; seed++)); do
#     echo "Launching seed ${seed}"
#     python "${SCRIPT}" --seed "${seed}" --out_dir "${OUT_DIR}" --ucb bootstrap --ucb_alpha 0.1  &
#     sleep 10
#     while [ "$(jobs -r | wc -l)" -ge "${MAX_JOBS}" ]; do
#         sleep 1
#     done
# done

# wait
# echo "All ${NUM_RUNS} runs finished."

