import subprocess
preliminary_install_commands = ["pip3 install --upgrade pip",
                                "pip3 install pendulum"]
for cmd in preliminary_install_commands:
    subprocess.run(cmd,
                   shell=True,
                   check=True)
import pendulum

if __name__ == '__main__':
    time_stamp = pendulum.now().\
    	__str__().\
    	replace('T','_').\
    	replace(':','-')[:16]
    
    # Configure the run with this dict.
    # Always enter boolean and floats as strings.
    
    hparams = {
                'project_name':"tandem-EfficientNetB7-ResidualMLP",
                'flatten':'True',
                'blocks':2,
                'residual_bypass_dense_layers':'',
                'b_norm_or_dropout_residual_bypass_layers':'bnorm',
                'dropout_rate_for_bypass_layers':0.00,
                'final_layers':3,
                'b_norm_or_dropout_last_layers':'bnorm',
                'dropout_rate':'0.00',
                'epochs':25,
                'patience':10,
                'patience_min_delta':"0.00001",
                'training_set_size':5000,
                'batch_size':50,
                'eval_size': 50,
                'learning_rate':0.0007,
                'comments':"Test to make sure the new code in residualmlp/residual_mlp.py doesnt break anything."
                }
    if hparams['residual_bypass_dense_layers'] == '':
    	hparams['residual_bypass_dense_layers'] = '""'
    if hparams['flatten'] == "True":
        file_name_flatten = "flatten"
    elif hparams['flatten'] == 'False':
        file_name_flatten = 'NOTflattened'
    else:
        raise ValueError("The parameter 'flatten must be 'True' "
                         "| 'False' (str).")
     
    BASE_FILE_NAME = f"{time_stamp}"\
        + f"_lr{str(hparams['learning_rate']).split('.')[1]}"\
        + f"_blocks{hparams['blocks']}"\
        + f"_final_layers{hparams['final_layers']}"\
        + f"_{file_name_flatten}_{hparams['b_norm_or_dropout_last_layers']}"
    SHELL_LOGS_FILE_NAME = f"{BASE_FILE_NAME}_python3_shell_log.txt"
    RESULTS_DIR = f"{BASE_FILE_NAME}_results"
    BEST_MODEL_DIR = f"{BASE_FILE_NAME}_ExportedModel"
    SHELL_SCRIPT_NAME = f"run_{time_stamp}_job.sh"
    
    print(BASE_FILE_NAME)
    back_slash = "\\"
    shell_text = f"""echo "{BASE_FILE_NAME}" >> {SHELL_LOGS_FILE_NAME}
    echo "{hparams['comments']}" >> {SHELL_LOGS_FILE_NAME}
    pip3 install pandas >> {SHELL_LOGS_FILE_NAME}
    pip3 install numpy >> {SHELL_LOGS_FILE_NAME}
    python3 -u task.py --project_name {hparams['project_name']}{back_slash}
                       --results_dir {RESULTS_DIR}{back_slash}
                       --best_model_dir {BEST_MODEL_DIR}{back_slash}
                       --flatten {hparams['flatten']}{back_slash}
                       --blocks {hparams['blocks']}{back_slash}
                       --residual_bypass_dense_layers {hparams['residual_bypass_dense_layers']}{back_slash}
                       --b_norm_or_dropout_residual_bypass_layers {hparams['b_norm_or_dropout_residual_bypass_layers']}{back_slash}
                       --dropout_rate_for_bypass_layers {hparams['dropout_rate_for_bypass_layers']}{back_slash}
                       --final_layers {hparams['final_layers']}{back_slash}
                       --b_norm_or_dropout_last_layers {hparams['b_norm_or_dropout_last_layers']}{back_slash}
                       --dropout_rate {hparams['dropout_rate']}{back_slash}
                       --epochs {hparams['epochs']}{back_slash}
                       --patience {hparams['patience']}{back_slash}
                       --patience_min_delta {str(hparams['patience_min_delta'])}{back_slash}
                       --training_set_size {hparams['training_set_size']}{back_slash}
                       --batch_size {hparams['batch_size']}{back_slash}
                       --eval_size {hparams['eval_size']}{back_slash}
                       --learning_rate {hparams['learning_rate']}
    
    """
    
    with open(SHELL_SCRIPT_NAME,'w',encoding="utf8") as f:
        f.write(shell_text)
    
    command = f"sh {SHELL_SCRIPT_NAME} >> {SHELL_LOGS_FILE_NAME} &"
    subprocess.run(command,
                   shell=True,
                   check=True)
