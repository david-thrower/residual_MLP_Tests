echo "Original model but with image preprocessing... 200 epochs training" >> 2021-12-21_01_52_lr10e-3_bl5_7fl5_flaten_bnorm_python3_shell_log.txt
pip3 install pandas
pip3 install numpy
python3 -u task.py --project_name "tandem-EfficientNetB7-ResidualMLP" --results_dir "2021-12-21_01_52_lr10e-3_bl5_7fl5_flaten_bnorm_results" --best_model_dir "2021-12-21_01_52_exported_model_lr10e-3_bl5_7fl5_flaten_bnorm" --epochs 200 --patience 80 --patience_min_delta 0.00001 --training_set_size 5000 --batch_size 50 --eval_size 50 --learning_rate .0007 >> 2021-12-21_01_52_lr10e-3_bl5_7fl5_flaten_bnorm_python3_shell_log.txt
