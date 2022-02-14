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
    'project_name':"CIFAR10_EfficientNetB7-ResidualMLP_NAS_THIRD_PASS_SEARCH",
    'problem_type':"classification",
    'number_of_classes':10,
    'training_set_size':50000,
    'patience':25,
    'patience_min_delta':0.00001,
    'batch_size':50,
    'max_epochs':150,
    'minimum_learning_rate':0.0001,
    'maximum_learning_rate':0.02,
    'number_of_learning_rates_to_try':7,
    'minimum_number_of_blocks':2,
    'maximum_number_of_blocks':3,
    'minimum_number_of_layers_per_block':2,
    'maximum_number_of_layers_per_block':4,
    'minimum_neurons_per_block_layer':70,
    'maximum_neurons_per_block_layer':140,
    'n_options_of_neurons_per_layer_to_try':7,
    'minimum_neurons_per_block_layer_decay':15,
    'maximum_neurons_per_block_layer_decay':35,
    'minimum_dropout_rate_for_bypass_layers':0.2,
    'maximim_dropout_rate_for_bypass_layers':.6,
    'n_options_dropout_rate_for_bypass_layers':3,
    'minimum_inter_block_layers_per_block':0,
    'maximum_inter_block_layers_per_block':85,
    'n_options_inter_block_layers_per_block':7,
    'minimum_dropout_rate':0.2,
    'maximum_dropout_rate':.6,
    'n_options_dropout_rate':5,
    'minimum_final_dense_layers':33,
    'maximum_final_dense_layers':300,
    'n_options_final_dense_layers':7
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