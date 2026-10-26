export FD_EXEC_PATH=${FD_EXEC_PATH:-ext/downward}
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p logs/final/satellites/sim

for seed in 0 1 2 3 4
do
    # low-level sampling is very hard for this environment
    if python3 predicators/main.py --env satellites --approach random_nsrt \
        --seed $seed \
        --excluded_predicates "ViewClear,IsCalibrated,HasChemX,HasChemY,Sees" \
        --num_train_tasks 500 \
        --in_domain_test True \
        --load_data \
        --timeout 5 \
        --ivntr_nsrt_path saved_approaches/final/satellites/ivntr_${seed}/satellites__ivntr__${seed}__ViewClear,IsCalibrated,HasChemX,HasChemY,Sees___aesuperv_False__.saved.neupi_info \
        --log_file logs/final/satellites/sim/random_nsrt_in_domain_$seed.log; then
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

    # Record start time
    start_time=$(date +%s)
    # low-level sampling is very hard for this environment
    if python3 predicators/main.py --env satellites --approach random_nsrt \
        --seed $seed \
        --excluded_predicates "ViewClear,IsCalibrated,HasChemX,HasChemY,Sees" \
        --num_train_tasks 500 \
        --load_data \
        --timeout 5 \
        --ivntr_nsrt_path saved_approaches/final/satellites/ivntr_${seed}/satellites__ivntr__${seed}__ViewClear,IsCalibrated,HasChemX,HasChemY,Sees___aesuperv_False__.saved.neupi_info \
        --log_file logs/final/satellites/sim/random_nsrt_ood_$seed.log; then
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

    # low-level sampling is very hard for this environment
    if python3 predicators/main.py --env satellites --approach random_options \
        --seed $seed \
        --excluded_predicates "ViewClear,IsCalibrated,HasChemX,HasChemY,Sees" \
        --num_train_tasks 500 \
        --in_domain_test True \
        --load_data \
        --timeout 5 \
        --log_file logs/final/satellites/sim/random_options_in_domain_$seed.log; then
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

    # Record start time
    start_time=$(date +%s)
    # low-level sampling is very hard for this environment
    if python3 predicators/main.py --env satellites --approach random_options \
        --seed $seed \
        --excluded_predicates "ViewClear,IsCalibrated,HasChemX,HasChemY,Sees" \
        --num_train_tasks 500 \
        --load_data \
        --timeout 5 \
        --log_file logs/final/satellites/sim/random_options_ood_$seed.log; then
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
