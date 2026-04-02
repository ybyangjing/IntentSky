# Train the RL model first
echo "=========================================================="
echo "Starting RL model training..."
echo "This will train the model once to avoid multiprocessing conflicts."
echo "=========================================================="

python ../run_simulator_sweep.py \
--dataset helios_dag \
--cluster_size 16 \
--sched_alg fifo \
--waiting_policy cardinal_query-1 \
--max_queue_length 1000 \
--long_job_thres 0.25 \
--preempt_cloud_ratio 3 \
--seed 2025 \
--log ~/logs/training_run.log \
--train_rl 1 \
--use_rl 1 \
--rl_workers 1 \
--processes 1 \
--total_jobs 2000 \
--loop 1

echo "=========================================================="
echo "Training completed."
echo "Now running the evaluation sweep (Fig 7) using the trained model."
echo "=========================================================="

# Evaluation: Philly Trace End2End.
# Explicitly set --train_rl 0 and --use_rl 1 to use the trained model without retraining.
# Using a loop to avoid OOM by restarting python process for each cluster_size.

SIZES=(16 20 24 28 32 36 40 44 48 52 56 60 64 68 72 76 80 84 88 92 96 100 104 108 112 116 120 124 132 136 140 144)
LOG_FILE=~/logs/helios_end2end404.log

# Clear the log file if it exists, as we will be appending to it (if the python script appends)
# However, run_simulator_sweep.py typically overwrites or writes a list of results.
# The pickle file format (list of logs) makes appending tricky if each run writes a separate list.
# But for now, let's solve the OOM first. The most robust way to combine results is to let them write to different files and then merge, OR assume the script handles append (which it likely doesn't for pickle).
# Let's write to separate files to be safe, then merge them.

# Actually, to minimize code changes and complexity for the user, let's check if we can just append.
# Pickle doesn't support simple appending.
# So we will write to separate logs: ~/logs/philly_end2end404_{size}.log

for size in "${SIZES[@]}"
do
    echo "Running simulation for cluster_size: $size"
    
    python ../run_simulator_sweep.py \
    --dataset helios_dag \
    --cluster_size $size \
    --sched_alg fifo \
    --waiting_policy zero-1 cardinal_query-1 heft-1 constant-1 linear_cost-0.076 linear_capacity-0.77 \
    --max_queue_length 30 1000000 \
    --long_job_thres -1 0.25 \
    --loop 0 1 \
    --preempt_cloud_ratio -1 3 \
    --seed 2025 \
    --filter_name helios_end2end \
    --log ~/logs/helios_end2end404_${size}.log \
    --train_rl 0 \
    --use_rl 1
    
    if [ $? -ne 0 ]; then
        echo "Error running size $size. Stopping."
        exit 1
    fi
done

echo "=========================================================="
echo "All simulations completed."
echo "Merging logs..."
echo "=========================================================="

# Create a simple python script to merge the pickle logs
python -c "
import pickle
import os
import glob
from pathlib import Path

log_dir = Path('~/logs').expanduser()
merged_logs = []
pattern = str(log_dir / 'helios_end2end404_*.log')
files = sorted(glob.glob(pattern), key=lambda x: int(x.split('_')[-1].split('.')[0]))

print(f'Found {len(files)} log files to merge.')

for f_path in files:
    try:
        with open(f_path, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, list):
                merged_logs.extend(data)
            else:
                merged_logs.append(data)
    except Exception as e:
        print(f'Error reading {f_path}: {e}')

output_path = log_dir / 'helios_end2end404.log'
with open(output_path, 'wb') as f:
    pickle.dump(merged_logs, f)

print(f'Successfully merged logs to {output_path}. Total records: {len(merged_logs)}')
"
