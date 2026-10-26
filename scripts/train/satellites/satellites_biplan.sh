export FD_EXEC_PATH=${FD_EXEC_PATH:-ext/downward}
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p logs/final/satellites

for seed in 0 1 2 3 4
do
    echo "Running Seed $seed --------------------------------------"
    # Record start time
    start_time=$(date +%s)
    # low-level sampling is very hard for this environment
    if python3 predicators/main.py --env satellites --approach ivntr \
        --seed $seed \
        --excluded_predicates "ViewClear,IsCalibrated,HasChemX,HasChemY,Sees" \
        --neupi_pred_config "predicators/config/satellites/pred.yaml" \
        --sesame_task_planner "astar" \
        --neupi_do_normalization False \
        --num_train_tasks 500 \
        --load_data \
        --neupi_selection_thresh 100000 \
        --neupi_equ_dataset 1.0 \
        --sesame_max_samples_per_step 30 \
        --timeout 5 \
        --approach_dir "saved_approaches/final/satellites/ivntr_$seed" \
        --neupi_save_path "saved_approaches/final/satellites/ivntr_$seed" \
        --log_file logs/final/satellites/ivntr_train_$seed.log; then
        echo "Seed $seed completed successfully."
    else
        echo "Seed $seed encountered an error."
    fi

    # Record end time
    end_time=$(date +%s)

    # Calculate the duration in seconds
    runtime=$((end_time - start_time))

    # Convert to hours, minutes, and seconds
    hours=$((runtime / 3600))
    minutes=$(( (runtime % 3600) / 60 ))
    seconds=$((runtime % 60))

    # Output the total runtime
    echo "Seed $seed completed in: ${hours}h ${minutes}m ${seconds}s"
done
