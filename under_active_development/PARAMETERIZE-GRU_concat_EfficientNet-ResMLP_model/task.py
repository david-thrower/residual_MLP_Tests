import os
import subprocess
import argparse
# Add parser for common params

try:
    import pendulum
except ModuleNotFoundError as not_found:
    print("We're missing a module:")
    print(not_found)
    print("No problem, we install it...")
    subprocess.run("pip3 install pendulum", 
    	           shell=True, 
    	           check=True)
    import pendulum

try:
    import pandas as pd
except ModuleNotFoundError as not_found:
    print("We're missing a module:")
    print(not_found)
    print("No problem, we install it...")
    subprocess.run("pip3 install pandas", 
    	           shell=True, 
    	           check=True)
    import pandas as pd

try:
    import keras_tuner as kt
except ModuleNotFoundError as not_found:
    print("We're missing a module:")
    print(not_found)
    print("No problem, we install it...")
    subprocess.run("pip3 install keras_tuner", 
    	           shell=True, 
    	           check=True)
    import keras_tuner as kt


try:
    import tensorflow as tf
except ModuleNotFoundError as not_found:
    print("We're missing a module:")
    print(not_found)
    print("No problem, we install it...")
    subprocess.run("pip3 install tensorflow", 
    	           shell=True, 
    	           check=True)
    import tensorflow as tf

from residualmlp.residual_mlp import ResidualMLP

parser = argparse.ArgumentParser()

parser.add_argument(
    "--project_name", type=str, 
    help="Name for this project.",
    default="CIFAR10_EfficientNetB7-ResidualMLP_NAS_AUGMENTED_SPACE")

parser.add_argument(
    "--number_of_classes", 
    type = int,
    help="Number of categorical classes. If a regression or "
    "logstic regression problem set this to 1.",
    default=10)

parser.add_argument(
    "--problem_type",
    type=str,
    help="'classification' | 'regression'. "
         "Default:'classification'",
    default="classification")

parser.add_argument(
    "--training_set_size", 
    type=int,
    help="Training set size (how many observations).",
    default=50000)

parser.add_argument(
    "--patience",
    type=int,
    help="How many epochs with no improved performance before the "
        "early stopping callback stops further training?",
    default=25)

parser.add_argument(
    "--patience_min_delta",
    type=float,
    help="How sensitive should the early stopping callback be"
        "  to change?",
    default=0.00001)

parser.add_argument(
    "--batch_size",
    type=int,
    help="How many observations to train with...",
    default=50)

parser.add_argument(
    "--max_epochs",
    type=int,
    help="max_epochs: Integer, the maximum number of epochs to train one "
         "model. It is recommended to set this to a value slightly higher "
         "than the expected epochs to convergence for your largest Model, "
         "and to use early stopping during training (for example, via .",
         default=50)

parser.add_argument(
        "--minimum_learning_rate",
        type=float,
        help="Lowest learning rate to try?",
        default = 0.00007)

parser.add_argument(
        "--maximum_learning_rate",
        type=float,
        help="Highest learning rate to try?",
        default = 0.7)

parser.add_argument(
        "--number_of_learning_rates_to_try",
        type=int,
        help="How many learning rate to try (maximum)?",
        default = 7)

parser.add_argument(
        "--minimum_number_of_blocks",
        type=int,
        help="Minimum number of ResidualMLP blocks in neural architectures "
             "to try?",
        default = 1)

parser.add_argument(
        "--maximum_number_of_blocks",
        type=int,
        help="Maximum number of ResidualMLP blocks in neural architectures "
              "to try?",
        default = 8)


parser.add_argument(
        "--minimum_number_of_layers_per_block",
        type=int,
        help="Minimum number of layers to try in each ResidualMLP block in "
             "the neural architectures to try?",
        default = 1)

parser.add_argument(
        "--maximum_number_of_layers_per_block",
        type=int,
        help="Maximum number of layers to try in each ResidualMLP block in "
             "the neural architectures to try?",
        default = 8)

parser.add_argument(
        "--minimum_neurons_per_block_layer",
        type=int,
        help="Minimum number of neurons to try in the Dense layers in "
             "each ResidualMLP block in the neural architectures to tried?",
        default = 30)

parser.add_argument(
        "--maximum_neurons_per_block_layer",
        type=int,
        help="Maximum number of neurons to try in the Dense layers in "
             "each ResidualMLP block in the neural architectures to tried?",
        default = 130)

parser.add_argument(
        "--n_options_of_neurons_per_layer_to_try",
        type=int,
        help="How many different numbers of neurons (at most) to try in "
             "the Dense layers try in each ResidualMLP block in the neural "
             "architectures to tried?",
        default = 7)

parser.add_argument(
        "--minimum_neurons_per_block_layer_decay",
        type=int,
        help="Lowest decay in number of neurons (n less neurons than the "
             "last layer) to try in the Dense layers try in each "
             "ResidualMLP block in the neural "
             "architectures to tried?",
        default = 0)

parser.add_argument(
        "--maximum_neurons_per_block_layer_decay",
        type=int,
        help="Highest decay in number of neurons (n less neurons than the "
             "last layer) to try in the Dense layers try in each "
             "ResidualMLP block in the neural "
             "architectures to tried?",
        default = 50)

parser.add_argument(
        "--minimum_dropout_rate_for_bypass_layers",
        type=float,
        help="Lowest dropout rate to try for dropout layers located in "
             "residual byass layers?",
        default = 0.01)

parser.add_argument(
        "--maximim_dropout_rate_for_bypass_layers",
        type=float,
        help="Highest dropout rate to try for dropout layers located in "
             "residual byass layers?",
        default = 0.7)

parser.add_argument(
        "--n_options_dropout_rate_for_bypass_layers",
        type=int,
        help="How many dropout rates to try for dropout layers located in "
             "residual byass layers?",
        default = 7)

parser.add_argument(
        "--minimum_inter_block_layers_per_block",
        type=int,
        help="Minimum number of neurons per Dense layer for Dense layers "
             "inserted between ResidualMLP blocks? 0 Will not insert a "
             "layer if selected.",
        default = 0)

parser.add_argument(
        "--maximum_inter_block_layers_per_block",
        type=int,
        help="Maximum number of neurons per Dense layer for Dense layers "
             "inserted between ResidualMLP blocks? 0 Will not insert a "
             "layer if selected.",
        default = 150)

parser.add_argument(
        "--n_options_inter_block_layers_per_block",
        type=int,
        help="How many different numbers of neurons per Dense layer for "
             "Dense layers inserted between ResidualMLP blocks will we try "
             "(at most)?",
        default = 7)

parser.add_argument(
        "--minimum_dropout_rate",
        type=float,
        help="Minimum dropout rate for final Dense layers?",
        default = 0.01)

parser.add_argument(
        "--maximum_dropout_rate",
        type=float,
        help="Maximum dropout rate for final Dense layers?",
        default = 0.7)

parser.add_argument(
        "--n_options_dropout_rate",
        type=int,
        help="How many dropout rates (at max) to try for final Dense layers?",
        default = 7)

parser.add_argument(
        "--minimum_final_dense_layers",
        type=int,
        help="Lowest number of neurons to try for final Dense layers, after "
             "the last ResidualMLP block, before the very last Dense layer "
             "returning an output? "
             "(0 doesn't create a layer')",
        default = 0)

parser.add_argument(
        "--maximum_final_dense_layers",
        type=int,
        help="Highest number of neurons to try for final Dense layers, after "
             "the last ResidualMLP block, before the very last Dense layer "
             "returning an output? "
             "(0 doesn't create a layer')",
        default = 150)

parser.add_argument(
        "--n_options_final_dense_layers",
        type=int,
        help="How many options for neurons to try for final Dense layers, "
             "after the last ResidualMLP block, before the very last "
             "Dense layer returning an output?",
        default = 7)

parser.add_argument(
    "--min_efficient_net_head_layer_dense_units", 
    type = int,
    help="Minimum dense units for the first MLP layer after the "
         "effifientnet submodel.",
    default=1)
	

	
parser.add_argument(
    "--max_efficient_net_head_layer_dense_units", 
    type = int,
    help="Maximum dense units for the first MLP layer after the effifientnet "
         "submodel.",
    default=50)
	
parser.add_argument(
    "--n_options_efficient_net_head_layer_dense_units", 
    type = int,
    help="How many options for  the first MLP layer after the effifientnet "
         "submodel to try.",
    default=3)
	

	
parser.add_argument(
    "--min_efficient_net_residual_block_layers", 
    type = int,
    help="Minimum dense units for the residual Dense layer after the "
         "effifientnet submodel.",
    default=1)
	

	
parser.add_argument(
    "--max_efficient_net_residual_block_layers", 
    type = int,
    help="Maximum dense units for the residual Dense layer after the "
         "effifientnet submodel.",
    default=50)
	

	
parser.add_argument(
    "--n_options_efficient_net_residual_block_layers", 
    type = int,
    help="How many options for residual Dense layer after the effifientnet "
         "submodel to try.",
    default=3)
	

	
parser.add_argument(
    "--min_head_gru_units", 
    type = int,
    help="Minimum GRU units for the first GRU layer to try.",
    default=2)
	

parser.add_argument(
    "--max_head_gru_units", 
    type = int,
    help="Maximum GRU units for the first GRU layer to try.",
    default=25)
	

	
parser.add_argument(
    "--n_options_head_gru_units", 
    type = int,
    help="How many options for GRU units for the first GRU layer to try.",
    default=3)
	

	
parser.add_argument(
    "--min_second_gru_units", 
    type = int,
    help="Minimum GRU units for the 2nd GRU layer to try.",
    default=2)
	

	
parser.add_argument(
    "--max_second_gru_units", 
    type = int,
    help="Maximum GRU units for the 2nd GRU layer to try.",
    default=25)
	

	
parser.add_argument(
    "--n_options_second_gru_units", 
    type = int,
    help="How many options for GRU units for the first GRU layer to try.",
    default=3)
	

	
parser.add_argument(
    "--min_gru_head_layer_dense_units", 
    type = int,
    help="Minimum dense units for the first MLP layer after the GRU "
         "submodel.",
    default=1)
	

	
parser.add_argument(
    "--max_gru_head_layer_dense_units", 
    type = int,
    help="Maximum dense units for the first MLP layer after the "
         "GRU submodel.",
    default=50)
	

	
parser.add_argument(
    "--n_options_gru_head_layer_dense_units", 
    type = int,
    help="How many options for  the first MLP layer after the GRU "
         "submodel to try.",
    default=3)
	

	
parser.add_argument(
    "--min_gru_residual_block_layers", 
    type = int,
    help="Minimum dense units for the residual Dense layer after the "
         "GRU submodel.",
    default=1)
	

	
parser.add_argument(
    "--max_gru_residual_block_layers", 
    type = int,
    help="Maximum dense units for the residual Dense layer after the GRU "
         "submodel.",
    default=50)
	

	
parser.add_argument(
    "--n_options_gru_residual_block_layers", 
    type = int,
    help="How many options for residual Dense layer after the GRU "
         "submodel to try.",
    default=3)


args, _ = parser.parse_known_args()
hparams = args.__dict__

# Boilerplate args

DATE = pendulum.now().__str__()[:16].replace("T","_").replace(":","_")
PROJECT_NAME = hparams["project_name"]
TRAINING_SET_SIZE = hparams["training_set_size"]
NUMBER_OF_CLASSES = hparams["number_of_classes"]
PROBLEM_TYPE = hparams["problem_type"]

# Keras tuner search & fit args

PATIENCE = hparams["patience"]
PATIENCE_MIN_DELTA = hparams["patience_min_delta"]
BATCH_SIZE = hparams["batch_size"]
MAX_EPOCHS = hparams["max_epochs"]
RESULTS_DIR_FOR_SEARCH =\
    f'{DATE}_{PROJECT_NAME}_SEARCH_RUN'


# Base model args

BASE_MODEL_INPUT_SHAPE = (32,32,3)

# ResidualMLP model args

INPUT_SHAPE = (32,32,3)
MINIMUM_LEARNING_RATE = hparams["minimum_learning_rate"]
MAXIMUM_LEARNING_RATE = hparams["maximum_learning_rate"]
NUMBER_OF_LEARNING_RATES_TO_TRY = hparams["number_of_learning_rates_to_try"]
MINIMUM_NUMBER_OF_BLOCKS = hparams["minimum_number_of_blocks"]
MAXIMUM_NUMBER_OF_BLOCKS = hparams["maximum_number_of_blocks"]
MINIMUM_NUMBER_OF_LAYERS_PER_BLOCK =\
    hparams["minimum_number_of_layers_per_block"]
MAXIMUM_NUMBER_OF_LAYERS_PER_BLOCK =\
    hparams["maximum_number_of_layers_per_block"]
MINIMUM_NEURONS_PER_BLOCK_LAYER = hparams["minimum_neurons_per_block_layer"]
MAXIMUM_NEURONS_PER_BLOCK_LAYER = hparams["maximum_neurons_per_block_layer"]
N_OPTIONS_OF_NEURONS_PER_LAYER_TO_TRY =\
    hparams["n_options_of_neurons_per_layer_to_try"]
MINIMUM_NEURONS_PER_BLOCK_LAYER_DECAY =\
    hparams["minimum_neurons_per_block_layer_decay"]
MAXIMUM_NEURONS_PER_BLOCK_LAYER_DECAY =\
    hparams["maximum_neurons_per_block_layer_decay"]
MINIMUM_DROPOUT_RATE_FOR_BYPASS_LAYERS =\
    hparams["minimum_dropout_rate_for_bypass_layers"]
MAXIMIM_DROPOUT_RATE_FOR_BYPASS_LAYERS =\
    hparams["maximim_dropout_rate_for_bypass_layers"]
N_OPTIONS_DROPOUT_RATE_FOR_BYPASS_LAYERS =\
    hparams["n_options_dropout_rate_for_bypass_layers"]
MINIMUM_INTER_BLOCK_LAYERS_PER_BLOCK =\
    hparams["minimum_inter_block_layers_per_block"]
MAXIMUM_INTER_BLOCK_LAYERS_PER_BLOCK =\
    hparams["maximum_inter_block_layers_per_block"]
N_OPTIONS_INTER_BLOCK_LAYERS_PER_BLOCK =\
    hparams["n_options_inter_block_layers_per_block"]
MINIMUM_DROPOUT_RATE = hparams["minimum_dropout_rate"]
MAXIMUM_DROPOUT_RATE = hparams["maximum_dropout_rate"]
N_OPTIONS_DROPOUT_RATE = hparams["n_options_dropout_rate"]
MINIMUM_FINAL_DENSE_LAYERS = hparams["minimum_final_dense_layers"]
MAXIMUM_FINAL_DENSE_LAYERS = hparams["maximum_final_dense_layers"]
N_OPTIONS_FINAL_DENSE_LAYERS = hparams["n_options_final_dense_layers"]

# GRU-Concat-EfficientNetB7 base model kwargs:

MIN_EFFICIENT_NET_HEAD_LAYER_DENSE_UNITS =\
    hparams['min_efficient_net_head_layer_dense_units']
MAX_EFFICIENT_NET_HEAD_LAYER_DENSE_UNITS =\
    hparams['max_efficient_net_head_layer_dense_units']
N_OPTIONS_EFFICIENT_NET_HEAD_LAYER_DENSE_UNITS =\
    hparams['n_options_efficient_net_head_layer_dense_units']
MIN_EFFICIENT_NET_RESIDUAL_BLOCK_LAYERS =\
    hparams['min_efficient_net_residual_block_layers']
MAX_EFFICIENT_NET_RESIDUAL_BLOCK_LAYERS =\
    hparams['max_efficient_net_residual_block_layers']
N_OPTIONS_EFFICIENT_NET_RESIDUAL_BLOCK_LAYERS =\
    hparams['n_options_efficient_net_residual_block_layers']
MIN_HEAD_GRU_UNITS =\
    hparams['min_head_gru_units']
MAX_HEAD_GRU_UNITS =\
    hparams['max_head_gru_units']
N_OPTIONS_HEAD_GRU_UNITS =\
    hparams['n_options_head_gru_units']
MIN_SECOND_GRU_UNITS =\
    hparams['min_second_gru_units']
MAX_SECOND_GRU_UNITS =\
    hparams['max_second_gru_units']
N_OPTIONS_SECOND_GRU_UNITS =\
    hparams['n_options_second_gru_units']
MIN_GRU_HEAD_LAYER_DENSE_UNITS =\
    hparams['min_gru_head_layer_dense_units']
MAX_GRU_HEAD_LAYER_DENSE_UNITS =\
    hparams['max_gru_head_layer_dense_units']
N_OPTIONS_GRU_HEAD_LAYER_DENSE_UNITS =\
    hparams['n_options_gru_head_layer_dense_units']
MIN_GRU_RESIDUAL_BLOCK_LAYERS =\
    hparams['min_gru_residual_block_layers']
MAX_GRU_RESIDUAL_BLOCK_LAYERS =\
    hparams['max_gru_residual_block_layers']
N_OPTIONS_GRU_RESIDUAL_BLOCK_LAYERS =\
    hparams['n_options_gru_residual_block_layers']

# CIFAR10_EfficientNetB7-ResidualMLP_NAS

RESULTS_DIR_FOR_FINAL_MODEL =\
    f'{DATE}_{PROJECT_NAME}_FINAL_MODEL'

FINAL_MODEL_PATH =\
    f"{DATE}_{PROJECT_NAME}_FINAL_EXPORTED_MODEL"

# Print header info to shell script logs:

print("# To reproduce the environment the NAS task was run in, copy and paste "
      "the following information into the evaluation script environent "
      "(excluding this line). PLEASE NOTE!!: If you do not reporduce these "
      "variables, the hyperband will not reproduce the correct model!")    
print(f"Date = {DATE}")
print(f"PROBLEM_TYPE = {PROBLEM_TYPE}")
print(f"NUMBER_OF_CLASSES = {NUMBER_OF_CLASSES}")
print(f"INPUT_SHAPE = {INPUT_SHAPE}")
print(f"BASE_MODEL_INPUT_SHAPE = {BASE_MODEL_INPUT_SHAPE}")
print(f"PROJECT_NAME = {PROJECT_NAME}")
print(f"TRAINING_SET_SIZE = {TRAINING_SET_SIZE}")
print(f"PATIENCE = {PATIENCE}")
print(f"PATIENCE_MIN_DELTA = {PATIENCE_MIN_DELTA}")
print(f"BATCH_SIZE = {BATCH_SIZE}")
print(f"MAX_EPOCHS = {MAX_EPOCHS}")
print(f"RESULTS_DIR_FOR_SEARCH = {RESULTS_DIR_FOR_SEARCH}")
print(f"MINIMUM_LEARNING_RATE = {MINIMUM_LEARNING_RATE}")
print(f"MAXIMUM_LEARNING_RATE = {MAXIMUM_LEARNING_RATE}")
print(f"NUMBER_OF_LEARNING_RATES_TO_TRY = {NUMBER_OF_LEARNING_RATES_TO_TRY}")
print(f"MINIMUM_NUMBER_OF_BLOCKS = {MINIMUM_NUMBER_OF_BLOCKS}")
print(f"MAXIMUM_NUMBER_OF_BLOCKS = {MAXIMUM_NUMBER_OF_BLOCKS}")
print("MINIMUM_NUMBER_OF_LAYERS_PER_BLOCK = "
      f"{MINIMUM_NUMBER_OF_LAYERS_PER_BLOCK}")
print("MAXIMUM_NUMBER_OF_LAYERS_PER_BLOCK = "
      f"{MAXIMUM_NUMBER_OF_LAYERS_PER_BLOCK}")
print(f"MINIMUM_NEURONS_PER_BLOCK_LAYER = {MINIMUM_NEURONS_PER_BLOCK_LAYER}")
print(f"MAXIMUM_NEURONS_PER_BLOCK_LAYER = {MAXIMUM_NEURONS_PER_BLOCK_LAYER}")
print("N_OPTIONS_OF_NEURONS_PER_LAYER_TO_TRY = "
      f"{N_OPTIONS_OF_NEURONS_PER_LAYER_TO_TRY}")
print("MINIMUM_NEURONS_PER_BLOCK_LAYER_DECAY = "
      f"{MINIMUM_NEURONS_PER_BLOCK_LAYER_DECAY}")
print("MAXIMUM_NEURONS_PER_BLOCK_LAYER_DECAY = "
      f"{MAXIMUM_NEURONS_PER_BLOCK_LAYER_DECAY}")
print("MINIMUM_DROPOUT_RATE_FOR_BYPASS_LAYERS = "
      f"{MINIMUM_DROPOUT_RATE_FOR_BYPASS_LAYERS}")
print("MAXIMIM_DROPOUT_RATE_FOR_BYPASS_LAYERS = "
      f"{MAXIMIM_DROPOUT_RATE_FOR_BYPASS_LAYERS}")
print("N_OPTIONS_DROPOUT_RATE_FOR_BYPASS_LAYERS = "
      f"{N_OPTIONS_DROPOUT_RATE_FOR_BYPASS_LAYERS}")
print("MINIMUM_INTER_BLOCK_LAYERS_PER_BLOCK = "
      f"{MINIMUM_INTER_BLOCK_LAYERS_PER_BLOCK}")
print("MAXIMUM_INTER_BLOCK_LAYERS_PER_BLOCK = "
      f"{MAXIMUM_INTER_BLOCK_LAYERS_PER_BLOCK}")
print("N_OPTIONS_INTER_BLOCK_LAYERS_PER_BLOCK = "
      f"{N_OPTIONS_INTER_BLOCK_LAYERS_PER_BLOCK}")
print("MINIMUM_DROPOUT_RATE = "
	  f"{MINIMUM_DROPOUT_RATE}")
print("MAXIMUM_DROPOUT_RATE = "
      f"{MAXIMUM_DROPOUT_RATE}")
print("N_OPTIONS_DROPOUT_RATE = "
      f"{N_OPTIONS_DROPOUT_RATE}")
print("MINIMUM_FINAL_DENSE_LAYERS = "
	  f"{MINIMUM_FINAL_DENSE_LAYERS}")
print("MAXIMUM_FINAL_DENSE_LAYERS = "
      f"{MAXIMUM_FINAL_DENSE_LAYERS}")
print("N_OPTIONS_FINAL_DENSE_LAYERS = "
      f"{N_OPTIONS_FINAL_DENSE_LAYERS}")
print("MIN_EFFICIENT_NET_HEAD_LAYER_DENSE_UNITS = "
      f"{MIN_EFFICIENT_NET_HEAD_LAYER_DENSE_UNITS}")
print("MAX_EFFICIENT_NET_HEAD_LAYER_DENSE_UNITS = "
      f"{MAX_EFFICIENT_NET_HEAD_LAYER_DENSE_UNITS}")
print("N_OPTIONS_EFFICIENT_NET_HEAD_LAYER_DENSE_UNITS = "
      f"{N_OPTIONS_EFFICIENT_NET_HEAD_LAYER_DENSE_UNITS}")
print("MIN_EFFICIENT_NET_RESIDUAL_BLOCK_LAYERS = "
      f"{MIN_EFFICIENT_NET_RESIDUAL_BLOCK_LAYERS}")
print("MAX_EFFICIENT_NET_RESIDUAL_BLOCK_LAYERS = "
      f"{MAX_EFFICIENT_NET_RESIDUAL_BLOCK_LAYERS}")
print("N_OPTIONS_EFFICIENT_NET_RESIDUAL_BLOCK_LAYERS = "
      f"{N_OPTIONS_EFFICIENT_NET_RESIDUAL_BLOCK_LAYERS}")
print("MIN_HEAD_GRU_UNITS = "
      f"{MIN_HEAD_GRU_UNITS}")
print("MAX_HEAD_GRU_UNITS = "
      f"{MAX_HEAD_GRU_UNITS}")
print("N_OPTIONS_HEAD_GRU_UNITS = "
      f"{N_OPTIONS_HEAD_GRU_UNITS}")
print("MIN_SECOND_GRU_UNITS = "
      f"{MIN_SECOND_GRU_UNITS}")
print("MAX_SECOND_GRU_UNITS = "
      f"{MAX_SECOND_GRU_UNITS}")
print("N_OPTIONS_SECOND_GRU_UNITS = "
      f"{N_OPTIONS_SECOND_GRU_UNITS}")
print("MIN_GRU_HEAD_LAYER_DENSE_UNITS = "
      f"{MIN_GRU_HEAD_LAYER_DENSE_UNITS}")
print("MAX_GRU_HEAD_LAYER_DENSE_UNITS = "
      f"{MAX_GRU_HEAD_LAYER_DENSE_UNITS}")
print("N_OPTIONS_GRU_HEAD_LAYER_DENSE_UNITS = "
      f"{N_OPTIONS_GRU_HEAD_LAYER_DENSE_UNITS}")
print("MIN_GRU_RESIDUAL_BLOCK_LAYERS = "
      f"{MIN_GRU_RESIDUAL_BLOCK_LAYERS}")
print("MAX_GRU_RESIDUAL_BLOCK_LAYERS = "
      f"{MAX_GRU_RESIDUAL_BLOCK_LAYERS}")
print("N_OPTIONS_GRU_RESIDUAL_BLOCK_LAYERS = "
      "{N_OPTIONS_GRU_RESIDUAL_BLOCK_LAYERS}")

print("""

cifar = tf.keras.datasets.cifar10.load_data()
(x_train, y_train), (x_test, y_test) = cifar

y_train_ohe = tf.one_hot([i[0] for i in  y_train],10)
indexes_for_rows = tf.range(0,y_train.shape[0])
shuffled_indexes = tf.random.shuffle(indexes_for_rows)
selected_indexes = shuffled_indexes[:TRAINING_SET_SIZE]
selected_x_train = x_train[selected_indexes,:,:,:]
selected_y_train_ohe = y_train_ohe.numpy()[selected_indexes,:]


#mod_with_fc_raw = tf.keras.applications.efficientnet.EfficientNetB7(
#    include_top=True, weights='imagenet', input_tensor=None,
#    input_shape = BASE_MODEL_INPUT_SHAPE, pooling='max', classes=1000
#)

# Make the deepest conv2d layer trainable, leave everything else
# as not trainable
#for layer in mod_with_fc_raw.layers:
    layer.trainable = False
# Last conv2d layer. This we want to train .
#mod_with_fc_raw.layers[-6].trainable = True

# Create the final base model
# (remove the final Dense and BatchNormalization layers ...) 
#efficient_net_b_7_transferable_base_model =\
#    tf.keras.Model(inputs=mod_with_fc_raw.layers[0].input, 
#                    outputs=mod_with_fc_raw.layers[-3].output)

model_builder = ResidualMLP(
                    problem_type = PROBLEM_TYPE,
                    minimum_learning_rate = MINIMUM_LEARNING_RATE, 
                    maximum_learning_rate = MAXIMUM_LEARNING_RATE, 
                    number_of_learning_rates_to_try =
                        NUMBER_OF_LEARNING_RATES_TO_TRY, 
                    input_shape = INPUT_SHAPE, 
                    bw_images = False, 
                    base_model = '', 
                    base_model_input_shape = BASE_MODEL_INPUT_SHAPE, 
                    flatten_after_base_model = False, 
                    minimum_number_of_blocks = MINIMUM_NUMBER_OF_BLOCKS, 
                    maximum_number_of_blocks = MAXIMUM_NUMBER_OF_BLOCKS, 
                    minimum_number_of_layers_per_block =
                        MINIMUM_NUMBER_OF_LAYERS_PER_BLOCK, 
                    maximum_number_of_layers_per_block =
                        MAXIMUM_NUMBER_OF_LAYERS_PER_BLOCK,
                    minimum_neurons_per_block_layer =
                        MINIMUM_NEURONS_PER_BLOCK_LAYER, 
                    maximum_neurons_per_block_layer =
                        MAXIMUM_NEURONS_PER_BLOCK_LAYER, 
                    n_options_of_neurons_per_layer_to_try =
                        N_OPTIONS_OF_NEURONS_PER_LAYER_TO_TRY, 
                    minimum_neurons_per_block_layer_decay =
                        MINIMUM_NEURONS_PER_BLOCK_LAYER_DECAY, 
                    maximum_neurons_per_block_layer_decay = 
                        MAXIMUM_NEURONS_PER_BLOCK_LAYER_DECAY, 
                    minimum_dropout_rate_for_bypass_layers =
                        MINIMUM_DROPOUT_RATE_FOR_BYPASS_LAYERS, 
                    maximim_dropout_rate_for_bypass_layers =
                        MAXIMIM_DROPOUT_RATE_FOR_BYPASS_LAYERS, 
                    n_options_dropout_rate_for_bypass_layers =
                        N_OPTIONS_DROPOUT_RATE_FOR_BYPASS_LAYERS,
                    minimum_inter_block_layers_per_block =
                        MINIMUM_INTER_BLOCK_LAYERS_PER_BLOCK, 
                    maximum_inter_block_layers_per_block =
                        MAXIMUM_INTER_BLOCK_LAYERS_PER_BLOCK,
                    n_options_inter_block_layers_per_block =\
                        N_OPTIONS_INTER_BLOCK_LAYERS_PER_BLOCK,
                    minimum_dropout_rate = MINIMUM_DROPOUT_RATE, 
                    maximum_dropout_rate = MAXIMUM_DROPOUT_RATE,
                    n_options_dropout_rate = N_OPTIONS_DROPOUT_RATE, 
                    minimum_final_dense_layers =
                        MINIMUM_FINAL_DENSE_LAYERS,
                    maximum_final_dense_layers =
                        MAXIMUM_FINAL_DENSE_LAYERS, 
                    n_options_final_dense_layers =
                        N_OPTIONS_FINAL_DENSE_LAYERS, 
                    number_of_classes = NUMBER_OF_CLASSES,
                    final_activation = tf.keras.activations.softmax,
                    min_efficient_net_head_layer_dense_units =\
                    	MIN_EFFICIENT_NET_HEAD_LAYER_DENSE_UNITS,
					max_efficient_net_head_layer_dense_units =\
					    MAX_EFFICIENT_NET_HEAD_LAYER_DENSE_UNITS,
					n_options_efficient_net_head_layer_dense_units =\
					    N_OPTIONS_EFFICIENT_NET_HEAD_LAYER_DENSE_UNITS,
					min_efficient_net_residual_block_layers =\
					    MIN_EFFICIENT_NET_RESIDUAL_BLOCK_LAYERS,
					max_efficient_net_residual_block_layers =\
					    MAX_EFFICIENT_NET_RESIDUAL_BLOCK_LAYERS,
					n_options_efficient_net_residual_block_layers =\
					    N_OPTIONS_EFFICIENT_NET_RESIDUAL_BLOCK_LAYERS,
					min_head_gru_units = MIN_HEAD_GRU_UNITS,
					max_head_gru_units = MAX_HEAD_GRU_UNITS,
					n_options_head_gru_units =\
					    N_OPTIONS_HEAD_GRU_UNITS,
					min_second_gru_units =\
					    MIN_SECOND_GRU_UNITS,
					max_second_gru_units =\
					    MAX_SECOND_GRU_UNITS,
					n_options_second_gru_units =\
					    N_OPTIONS_SECOND_GRU_UNITS,
					min_gru_head_layer_dense_units =\
					    MIN_GRU_HEAD_LAYER_DENSE_UNITS,
					max_gru_head_layer_dense_units =\
					    MAX_GRU_HEAD_LAYER_DENSE_UNITS,
					n_options_gru_head_layer_dense_units =\
					    N_OPTIONS_GRU_HEAD_LAYER_DENSE_UNITS,
					min_gru_residual_block_layers =\
					    MIN_GRU_RESIDUAL_BLOCK_LAYERS,
					max_gru_residual_block_layers =\
					    MAX_GRU_RESIDUAL_BLOCK_LAYERS,
					n_options_gru_residual_block_layers =\
					    N_OPTIONS_GRU_RESIDUAL_BLOCK_LAYERS)


logdir_for_search = os.path.join("logs", RESULTS_DIR_FOR_SEARCH + "_TB")
tensorboard_callback_search =\
    tf.keras.callbacks.TensorBoard(logdir_for_search, histogram_freq=1)

tuner = kt.Hyperband(
    model_builder.make_gru_concat_efficientnetb7_residualmlp_model,
    objective='val_loss',
    project_name = PROJECT_NAME,
    max_epochs = MAX_EPOCHS,
    hyperband_iterations = 2)

""")


if __name__ == "__main__":
    cifar = tf.keras.datasets.cifar10.load_data()
    (x_train, y_train), (x_test, y_test) = cifar
    
    y_train_ohe = tf.one_hot([i[0] for i in  y_train],10)
    indexes_for_rows = tf.range(0,y_train.shape[0])
    shuffled_indexes = tf.random.shuffle(indexes_for_rows)
    selected_indexes = shuffled_indexes[:TRAINING_SET_SIZE]
    selected_x_train = x_train[selected_indexes,:,:,:]
    selected_y_train_ohe = y_train_ohe.numpy()[selected_indexes,:]
    
    
    
    
    #mod_with_fc_raw = tf.keras.applications.efficientnet.EfficientNetB7(
    #    include_top=True, weights='imagenet', input_tensor=None,
    #    input_shape = BASE_MODEL_INPUT_SHAPE, pooling='max', classes=1000
    #)
    
    # Make the deepest conv2d layer trainable, leave everything else
    # as not trainable
    #for layer in mod_with_fc_raw.layers:
    #    layer.trainable = False
    # Last conv2d layer. This we want to train .
    #mod_with_fc_raw.layers[-6].trainable = True
    
    # Create the final base model
    # (remove the final Dense and BatchNormalization layers ...) 
    #efficient_net_b_7_transferable_base_model =\
    #    tf.keras.Model(inputs=mod_with_fc_raw.layers[0].input, 
    #                    outputs=mod_with_fc_raw.layers[-3].output)
    

    
    
    logdir_for_search = os.path.join("logs", RESULTS_DIR_FOR_SEARCH + "_TB")
    tensorboard_callback_search =\
        tf.keras.callbacks.TensorBoard(logdir_for_search, histogram_freq=1)
    
    tuner = kt.Hyperband(
        hypermodel = ResidualMLP(
                          problem_type = PROBLEM_TYPE,
                          minimum_learning_rate = MINIMUM_LEARNING_RATE, 
                          maximum_learning_rate = MAXIMUM_LEARNING_RATE, 
                          number_of_learning_rates_to_try =
                              NUMBER_OF_LEARNING_RATES_TO_TRY, 
                          input_shape = INPUT_SHAPE, 
                          bw_images = False,
                          base_model = '', 
                          base_model_input_shape = BASE_MODEL_INPUT_SHAPE,
                          flatten_after_base_model = False, 
                          minimum_number_of_blocks = MINIMUM_NUMBER_OF_BLOCKS, 
                          maximum_number_of_blocks = MAXIMUM_NUMBER_OF_BLOCKS, 
                          minimum_number_of_layers_per_block =
                              MINIMUM_NUMBER_OF_LAYERS_PER_BLOCK, 
                          maximum_number_of_layers_per_block =
                              MAXIMUM_NUMBER_OF_LAYERS_PER_BLOCK,
                          minimum_neurons_per_block_layer =
                              MINIMUM_NEURONS_PER_BLOCK_LAYER, 
                          maximum_neurons_per_block_layer =
                              MAXIMUM_NEURONS_PER_BLOCK_LAYER, 
                          n_options_of_neurons_per_layer_to_try =
                              N_OPTIONS_OF_NEURONS_PER_LAYER_TO_TRY, 
                          minimum_neurons_per_block_layer_decay =
                              MINIMUM_NEURONS_PER_BLOCK_LAYER_DECAY, 
                          maximum_neurons_per_block_layer_decay = 
                              MAXIMUM_NEURONS_PER_BLOCK_LAYER_DECAY, 
                          minimum_dropout_rate_for_bypass_layers =
                              MINIMUM_DROPOUT_RATE_FOR_BYPASS_LAYERS, 
                          maximim_dropout_rate_for_bypass_layers =
                              MAXIMIM_DROPOUT_RATE_FOR_BYPASS_LAYERS, 
                          n_options_dropout_rate_for_bypass_layers =
                              N_OPTIONS_DROPOUT_RATE_FOR_BYPASS_LAYERS,
                          minimum_inter_block_layers_per_block =
                              MINIMUM_INTER_BLOCK_LAYERS_PER_BLOCK, 
                          maximum_inter_block_layers_per_block =
                              MAXIMUM_INTER_BLOCK_LAYERS_PER_BLOCK,
                          n_options_inter_block_layers_per_block =\
                              N_OPTIONS_INTER_BLOCK_LAYERS_PER_BLOCK,
                          minimum_dropout_rate = MINIMUM_DROPOUT_RATE, 
                          maximum_dropout_rate = MAXIMUM_DROPOUT_RATE,
                          n_options_dropout_rate = N_OPTIONS_DROPOUT_RATE, 
                          minimum_final_dense_layers =
                              MINIMUM_FINAL_DENSE_LAYERS,
                          maximum_final_dense_layers =
                              MAXIMUM_FINAL_DENSE_LAYERS, 
                          n_options_final_dense_layers =
                              N_OPTIONS_FINAL_DENSE_LAYERS, 
                          number_of_classes = NUMBER_OF_CLASSES,
                          final_activation = tf.keras.activations.softmax,
  		                min_efficient_net_head_layer_dense_units =\
  		                    MIN_EFFICIENT_NET_HEAD_LAYER_DENSE_UNITS,
  						max_efficient_net_head_layer_dense_units =\
  						    MAX_EFFICIENT_NET_HEAD_LAYER_DENSE_UNITS,
  						n_options_efficient_net_head_layer_dense_units =\
  						    N_OPTIONS_EFFICIENT_NET_HEAD_LAYER_DENSE_UNITS,
  						min_efficient_net_residual_block_layers =\
  						    MIN_EFFICIENT_NET_RESIDUAL_BLOCK_LAYERS,
  						max_efficient_net_residual_block_layers =\
  						    MAX_EFFICIENT_NET_RESIDUAL_BLOCK_LAYERS,
  						n_options_efficient_net_residual_block_layers =\
  						    N_OPTIONS_EFFICIENT_NET_RESIDUAL_BLOCK_LAYERS,
  						min_head_gru_units = MIN_HEAD_GRU_UNITS,
  						max_head_gru_units = MAX_HEAD_GRU_UNITS,
  						n_options_head_gru_units =\
  						    N_OPTIONS_HEAD_GRU_UNITS,
  						min_second_gru_units = MIN_SECOND_GRU_UNITS,
  						max_second_gru_units = MAX_SECOND_GRU_UNITS,
  						n_options_second_gru_units =\
  						    N_OPTIONS_SECOND_GRU_UNITS,
  						min_gru_head_layer_dense_units =\
  						    MIN_GRU_HEAD_LAYER_DENSE_UNITS,
  						max_gru_head_layer_dense_units =\
  						    MAX_GRU_HEAD_LAYER_DENSE_UNITS,
  						n_options_gru_head_layer_dense_units =\
  						    N_OPTIONS_GRU_HEAD_LAYER_DENSE_UNITS,
  						min_gru_residual_block_layers =\
  						    MIN_GRU_RESIDUAL_BLOCK_LAYERS,
  						max_gru_residual_block_layers =\
  						    MAX_GRU_RESIDUAL_BLOCK_LAYERS,
  						n_options_gru_residual_block_layers =\
  						    N_OPTIONS_GRU_RESIDUAL_BLOCK_LAYERS).\
            make_gru_concat_efficientnetb7_residualmlp_model,
        objective='val_loss',
        project_name = PROJECT_NAME,
        max_epochs = MAX_EPOCHS,
        hyperband_iterations = 3)
    
    strategy = tf.distribute.MirroredStrategy(
                                          devices=None,
                                          cross_device_ops=None)
    tuner.search(x=selected_x_train,  
                 y=selected_y_train_ohe,
                 epochs=MAX_EPOCHS,
                 batch_size=BATCH_SIZE, 
                 callbacks=[
                        tf.keras.callbacks.EarlyStopping(
                            monitor="val_loss",
                            patience=PATIENCE,
                            min_delta=PATIENCE_MIN_DELTA,
                            restore_best_weights=True,
                        ),
                        tensorboard_callback_search,
                    ],
                 validation_split=0.3)
    
    print("These are the best params and results:")
    tuner.results_summary(num_trials=10)
    
    # final_model = tuner.get_best_models(num_models=1)[0]
    
    best_hp = tuner.get_best_hyperparameters()[0]
    final_model = tuner.hypermodel.build(best_hp)

    
    
    logdir_final_model = os.path.join("logs",
                                      RESULTS_DIR_FOR_FINAL_MODEL + "_TB")
    tensorboard_callback_final =\
        tf.keras.callbacks.TensorBoard(logdir_final_model, histogram_freq=1)
    
    history = final_model.fit(x=selected_x_train,  
                        y=selected_y_train_ohe, 
                        batch_size=BATCH_SIZE, 
                        epochs=150,      
                        verbose='auto', 
                        callbacks=[tf.keras.callbacks.\
                                   EarlyStopping(monitor='val_loss',
                                                 patience=PATIENCE,
                                                 min_delta=PATIENCE_MIN_DELTA,
                                                 restore_best_weights=True),
                                tensorboard_callback_final], 
                        validation_split=0.3,
                        validation_data=None,
                        shuffle=True,
                        class_weight=None, 
                        sample_weight=None, 
                        initial_epoch=0, 
                        steps_per_epoch=None, 
                        validation_steps=None, 
                        validation_batch_size=10, 
                        validation_freq=1, 
                        max_queue_size=10, 
                        workers=5, 
                        use_multiprocessing=True)
    
    
    
    hy = pd.DataFrame(history.history)
    hy.to_csv(f'{DATE}_test_history.csv')
    hy.to_json(f'{DATE}_test_history.json')
    
    final_model.save(FINAL_MODEL_PATH)
    print("Successful Run!")

