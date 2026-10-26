export FD_EXEC_PATH=${FD_EXEC_PATH:-ext/downward}
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p logs

for seed in 0 1 2 3 4
do
    echo "Running Seed $seed --------------------------------------"
    # Record start time
    start_time=$(date +%s)
    # low-level sampling is very hard for this environment
    if python3 predicators/main.py --env blocks_onclear --approach ivntr \
        --seed $seed \
        --excluded_predicates "On,OnTable,GripperOpen,Holding,Clear" \
        --neupi_pred_config "predicators/config/blocks_onclear/pred.yaml" \
        --sesame_task_planner "astar" \
        --num_train_tasks 500 \
        --timeout 5 \
        --neupi_entropy_w 0.0 \
        --neupi_loss_w 1.0 \
        --execution_monitor expected_atoms \
        --neupi_selection_thresh 6000000 \
        --approach_dir "saved_approaches/final/blocks_pos/ivntr_$seed" \
        --neupi_save_path "saved_approaches/final/blocks_pos/ivntr_$seed" \
        --log_file logs/ivntr_ood_$seed.log; then
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
