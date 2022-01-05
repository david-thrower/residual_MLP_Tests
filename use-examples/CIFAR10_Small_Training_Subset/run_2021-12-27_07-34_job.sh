echo "2021-12-27_07-34_lr0007_blocks1_final_layers3_flatten_dropout" >> 2021-12-27_07-34_lr0007_blocks1_final_layers3_flatten_dropout_python3_shell_log.txt
    echo "Test Blocks 1 Final layers 3 with dropout 25 and sidelayers [[5],[5]] with 25% dropout between them." >> 2021-12-27_07-34_lr0007_blocks1_final_layers3_flatten_dropout_python3_shell_log.txt
    pip3 install pandas >> 2021-12-27_07-34_lr0007_blocks1_final_layers3_flatten_dropout_python3_shell_log.txt
    pip3 install numpy >> 2021-12-27_07-34_lr0007_blocks1_final_layers3_flatten_dropout_python3_shell_log.txt
    python3 -u task.py --project_name tandem-EfficientNetB7-ResidualMLP\
                       --results_dir 2021-12-27_07-34_lr0007_blocks1_final_layers3_flatten_dropout_results\
                       --best_model_dir 2021-12-27_07-34_lr0007_blocks1_final_layers3_flatten_dropout_ExportedModel\
                       --flatten True\
                       --blocks 1\
                       --final_layers 3\
                       --b_norm_or_dropout_last_layers dropout\
                       --dropout_rate 0.25\
                       --epochs 120\
                       --patience 20\
                       --patience_min_delta 0.00001\
                       --training_set_size 5000\
                       --batch_size 50\
                       --eval_size 50\
                       --learning_rate 0.0007
    
    