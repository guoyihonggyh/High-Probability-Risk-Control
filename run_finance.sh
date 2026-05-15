# #!/bin/bash

# set -e

# SCRIPT="opo_finance_baseline.py"
# OUT_DIR="/scratch/aliu74/yihong/conformal/risk_control_result/opo_finance_snips"

# BASELINE="snips"
# MAX_JOBS=20
# NUM_RUNS=500

# mkdir -p "${OUT_DIR}"

# for ((seed=0; seed<NUM_RUNS; seed++)); do
#     echo "Launching seed ${seed}"
#     python "${SCRIPT}" --seed "${seed}" --out_dir "${OUT_DIR}" --baseline "${BASELINE}"  &
#     sleep 10
#     while [ "$(jobs -r | wc -l)" -ge "${MAX_JOBS}" ]; do
#         sleep 2
#     done
# done

# wait
# echo "All ${NUM_RUNS} runs finished."


# #!/bin/bash

# set -e

# SCRIPT="opo_finance.py"
# OUT_DIR="/scratch/aliu74/yihong/conformal/risk_control_result/cpc_v2"
# BASELINE="none"
# MAX_JOBS=10
# NUM_RUNS=500
# ucb_method="tail_safe"
# ucb_alpha=0.1

# mkdir -p "${OUT_DIR}"

# for ((seed=0; seed<NUM_RUNS; seed++)); do
#     echo "Launching seed ${seed}"
#     python "${SCRIPT}" --seed "${seed}" --out_dir "${OUT_DIR}" --ucb "${ucb_method}" --ucb_alpha "${ucb_alpha}" &
#     sleep 10
#     while [ "$(jobs -r | wc -l)" -ge "${MAX_JOBS}" ]; do
#         sleep 1
#     done
# done

# wait
# echo "All ${NUM_RUNS} runs finished."



#!/bin/bash

set -e

SCRIPT="opo_finance_ltt.py"
OUT_DIR="/scratch/aliu74/yihong/conformal/risk_control_result/opo_finance_ltt_original_fixsequence_alpha0.2"
BASELINE="none"
MAX_JOBS=10
NUM_RUNS=500

ltt_pvalue="hb_original"
ucb_alpha=0.2
ltt_method="fixed_sequence"

mkdir -p "${OUT_DIR}"

for ((seed=0; seed<NUM_RUNS; seed++)); do
    echo "Launching seed ${seed}"
    python "${SCRIPT}" \
        --seed "${seed}" \
        --out_dir "${OUT_DIR}" \
        --ltt_pvalue "${ltt_pvalue}" \
        --ucb_alpha "${ucb_alpha}" \
        --ltt_method "${ltt_method}" &
    
    sleep 10

    while [ "$(jobs -r | wc -l)" -ge "${MAX_JOBS}" ]; do
        sleep 1
    done
done

wait
echo "All ${NUM_RUNS} runs finished."

# #!/bin/bash

# set -e

# SCRIPT="opo_finance_baseline_rolling_rc.py"
# OUT_DIR="/scratch/aliu74/yihong/conformal/risk_control_result/opo_finance_rolling_rc_observed_gamma10.0"
# BASELINE="rolling_rc"
# MAX_JOBS=10
# NUM_RUNS=500

# rolling_estimator="snips"
# rolling_gamma=10.0

# mkdir -p "${OUT_DIR}"

# for ((seed=0; seed<NUM_RUNS; seed++)); do
#     echo "Launching seed ${seed}"
#     python "${SCRIPT}" \
#         --seed "${seed}" \
#         --out_dir "${OUT_DIR}" \
#         --baseline "${BASELINE}" \
#         --rolling_gamma "${rolling_gamma}" &

#     sleep 10

#     while [ "$(jobs -r | wc -l)" -ge "${MAX_JOBS}" ]; do
#         sleep 1
#     done
# done

# wait
# echo "All ${NUM_RUNS} runs finished."