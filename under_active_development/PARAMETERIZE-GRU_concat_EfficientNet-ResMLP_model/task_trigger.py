"""Runs task.py, is a more convenient way to set up the parameters than
manually writing a shell script."""

import subprocess
preliminary_install_commands = ["pip3 install --upgrade pip",
                                "pip3 install pendulum"]
for cmd in preliminary_install_commands:
    subprocess.run(cmd,
                   shell=True,
                   check=True)
import pendulum

# Configure the run with this dict.
# Always enter boolean and floats as strings.

TIME_STAMP = pendulum.now().\
                    	__str__().\
                    	replace('T','_').\
                    	replace(':','-')[:16]

HPARAMS = {
    'project_name':"2022-02-25_CIFAR10_GRU_Concat_EfficientNetB7-ResidualMLP_NAS_attempt",
    'problem_type':"classification",
    'number_of_classes':10,
    'training_set_size':50000,
    'patience':5,
    'patience_min_delta':0.00001,
    'batch_size':20,
    'max_epochs':150,
    'minimum_learning_rate':0.0001,
    'maximum_learning_rate':0.02,
    'number_of_learning_rates_to_try':7,
    'minimum_number_of_blocks':2,
    'maximum_number_of_blocks':5,
    'minimum_number_of_layers_per_block':2,
    'maximum_number_of_layers_per_block':4,
    'minimum_neurons_per_block_layer':30,
    'maximum_neurons_per_block_layer':140,
    'n_options_of_neurons_per_layer_to_try':7,
    'minimum_neurons_per_block_layer_decay':3,
    'maximum_neurons_per_block_layer_decay':50,
    'minimum_dropout_rate_for_bypass_layers':0.2,
    'maximim_dropout_rate_for_bypass_layers':.6,
    'n_options_dropout_rate_for_bypass_layers':3,
    'minimum_inter_block_layers_per_block':0,
    'maximum_inter_block_layers_per_block':85,
    'n_options_inter_block_layers_per_block':7,
    'minimum_dropout_rate':0.2,
    'maximum_dropout_rate':.6,
    'n_options_dropout_rate':5,
    'minimum_final_dense_layers':0,
    'maximum_final_dense_layers':175,
    'n_options_final_dense_layers':7,
    'min_efficient_net_head_layer_dense_units':5,
	'max_efficient_net_head_layer_dense_units':50,
	'n_options_efficient_net_head_layer_dense_units':5,
	'min_efficient_net_residual_block_layers':5,
	'max_efficient_net_residual_block_layers':50,
	'n_options_efficient_net_residual_block_layers':5,
	'min_head_gru_units':3,
	'max_head_gru_units':30,
	'n_options_head_gru_units':3,
	'min_second_gru_units':3,
	'max_second_gru_units':30,
	'n_options_second_gru_units':3,
	'min_gru_head_layer_dense_units':5,
	'max_gru_head_layer_dense_units':50,
	'n_options_gru_head_layer_dense_units':5,
	'min_gru_residual_block_layers':5,
	'max_gru_residual_block_layers':50,
	'n_options_gru_residual_block_layers':5
}


if __name__ == '__main__':
    
    BASE_FILE_NAME = f"{TIME_STAMP}_CIFAR10_EfficientNetB7-ResidualMLP_NAS_SECOND_PASS_SEARCH"
    SHELL_LOGS_FILE_NAME = f"{BASE_FILE_NAME}_python3_shell_log.txt"
    SHELL_SCRIPT_NAME = f"{BASE_FILE_NAME}_task_trigger.sh"
    

    back_slash = "\\"
    shell_script_content = f"python3 task.py {back_slash}" + "\n"
    
    for key, value in HPARAMS.items():
        shell_script_content += f"    --{key} '{value}'{back_slash}" + "\n"
    shell_script_content = shell_script_content[:-2] + "\n"
    print(shell_script_content)

    
    with open(SHELL_SCRIPT_NAME,'w',encoding="utf8") as f:
        f.write(shell_script_content)
    
    command = f"sh {SHELL_SCRIPT_NAME} >> {SHELL_LOGS_FILE_NAME} &"
    subprocess.run(command,
                   shell=True,
                   check=True)
