# Parallel_Workflow
Simple Parallel Workflow

To run:
chmod +x run_worker.py reduce_results.py
sbatch sweep.sbatch

When done running:
python3 reduce_results.py --indir results --write-csv best_results.csv