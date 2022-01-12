echo "2022-01-12_04-46_lr0007_blocks2_final_layers3_flatten_bnorm" >> 2022-01-12_04-46_lr0007_blocks2_final_layers3_flatten_bnorm_python3_shell_log.txt
    echo "Test to make sure the new code in residualmlp/residual_mlp.py doesnt break anything." >> 2022-01-12_04-46_lr0007_blocks2_final_layers3_flatten_bnorm_python3_shell_log.txt
    pip3 install pandas >> 2022-01-12_04-46_lr0007_blocks2_final_layers3_flatten_bnorm_python3_shell_log.txt
    pip3 install numpy >> 2022-01-12_04-46_lr0007_blocks2_final_layers3_flatten_bnorm_python3_shell_log.txt
    python3 -u task.py --project_name tandem-EfficientNetB7-ResidualMLP\
                       --results_dir 2022-01-12_04-46_lr0007_blocks2_final_layers3_flatten_bnorm_results\
                       --best_model_dir 2022-01-12_04-46_lr0007_blocks2_final_layers3_flatten_bnorm_ExportedModel\
                       --flatten True\
                       --blocks 2\
                       --residual_bypass_dense_layers ""\
                       --b_norm_or_dropout_residual_bypass_layers bnorm\
                       --dropout_rate_for_bypass_layers 0.0\
                       --inter_block_layers_per_block [60]\
                       --final_layers 3\
                       --b_norm_or_dropout_last_layers bnorm\
                       --dropout_rate 0.00\
                       --epochs 25\
                       --patience 10\
                       --patience_min_delta 0.00001\
                       --training_set_size 5000\
                       --batch_size 50\
                       --eval_size 50\
                       --learning_rate 0.0007
    
    