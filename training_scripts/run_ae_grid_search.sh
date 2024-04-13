#!/bin/bash

# Loop through all json files starting with "channel"
for file in ae_configs/channel*.json; do
    # Prompt the user for confirmation
    echo "Do you want to execute the file $file? (y/n)"
    read -p "" answer

    # Check if the answer is 'y' or 'Y'
    if [[ $answer = [Yy] ]]; then
        # Execute the Python command with the current file
        sbatch slurm_train_ae.sh "$file"
    else
        echo "Skipping $file"
    fi
done

