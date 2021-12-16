import argparse
import os
import json
import pandas as pd
import tensorflow as tf
from residualmlp.residual_mlp import make_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_name",
        type=str,
        help = "Name for this project."
    )
    # parser.add_argument(
    #     "--hyperband_iterations",
    #     type=int,
    #     help = "Run hyperband algo n times."
    # )
    # parser.add_argument(
    #    "--max_epochs",
    #    type=int,
    #    help = "Value for max epochs.",
    #    default = 50  
    # )
    parser.add_argument(
        "--results_dir",
        type=str,
        help = "directory for results",
        default = 'results'
    )
    parser.add_argument(
        "--best_model_dir",
        type=str,
        help = "Directory for model.save to save the best model?",
        default = 'best_model'
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help = "How many training epochs?",
        default = 50
    )
    parser.add_argument(
        "--patience",
        type=int,
        help = "How many epochs with no improved performance before the "\
            + "early stopping callback gives up?",
        default = 150
    )
    parser.add_argument(
        "--patience_min_delta",
        type=float,
        help = "How sensitive should the early stopping callback be"\
            + "  to change?",
        default = .00001
    )        
    parser.add_argument(
        "--training_set_size",
        type=int,
        help = "How many observations to train with...",
        default = 5000
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help = "How many observations to train with...",
        default = 50
    )
    parser.add_argument(
        "--eval_size",
        type=int,
        help = "How many observations per training batch...",
        default = 50
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help = "Learning_rate...",
        default = .0007
    )

    # parser.add_argument(
    #     "--learning_rate_decay_factor",
    #     type=int,
    #     help = "Learning_rate...",
    #     default = .00001
    # )    
    args, _  = parser.parse_known_args()
    hparams = args.__dict__
    print("Hyperparameters and metadata:")
    print(hparams)
    # Parsed params:
    #BATCH_SIZE = hparams['batch_size'] *
    #BW_IMAGES = hparams['bw_images'] == "True" 
    #LEARNING_RATE = hparams['learning_rate'] *
    


    LEARNING_RATE = hparams['learning_rate']
    PATIENCE_MIN_DELTA = hparams['patience_min_delta']
    BATCH_SIZE = hparams['batch_size']
    TRAINING_SET_SIZE = hparams['training_set_size']
    EVAL_BATCH_SIZE = hparams['eval_size']
    # MAX_TRIALS = 20 # change to at least 1000 for the scales up test
    PROJECT_NAME = hparams['project_name']
    # HYPERBAND_ITERATIONS = hparams['hyperband_iterations']
    #MAX_EPOCHS = hparams['max_epochs']
    RESULTS_DIR = hparams['results_dir']
    BEST_MODEL_DIR = hparams['best_model_dir']
    EPOCHS = hparams['epochs']
    PATIENCE = hparams['patience']

    # #########
    
    
    # Things to parameterize
    # PROJECT_NAME = 'small-test'
    # RESULTS_DIR = 'small-test-results'
    #BEST_MODEL_DIR = 'small-test-best-model'
    #EPOCHS = 1000
    #PATIENCE = 7
    BLOCKS_SELECTION = 5
    FINAL_DENSE_LAYERS_SELECTION = 7
    
    BATCH_SIZE = 50 #[5,10,20,50]
    BLOCKS = []
    BLOCKS.append([[5,100,10],[7,50,5]])
    BLOCKS.append([[5,100,10],[7,75,8]])
    BLOCKS.append([[5,75,10],[5,75,10],[5,75,10],[5,75,10],[5,75,10]])
    BLOCKS.append([[5,150,10]])
    BLOCKS.append([[6,200,25]])
    BLOCKS.append([[5,150,10],[5,75,10],[5,75,10],
                   [5,75,10],[5,75,10],[5,75,10]])
    
    
    
    B_NORM_OR_DROPOUT_LAST_LAYERS = ['dropout','bnorm']
    
    
    
    FINAL_DENSE_LAYERS_PERMUTATIONS = []
    FINAL_DENSE_LAYERS_PERMUTATIONS.append([75,35])
    FINAL_DENSE_LAYERS_PERMUTATIONS.append([100,35])
    FINAL_DENSE_LAYERS_PERMUTATIONS.append([50,50])
    FINAL_DENSE_LAYERS_PERMUTATIONS.append([35,35])
    FINAL_DENSE_LAYERS_PERMUTATIONS.append([25 ,10])
    FINAL_DENSE_LAYERS_PERMUTATIONS.append([25,25,25])
    FINAL_DENSE_LAYERS_PERMUTATIONS.append([75,25,25])
    FINAL_DENSE_LAYERS_PERMUTATIONS.append([100,35,20])
    
    # hard coded variables
    HEIGHT = 32
    WIDTH = 32
    CHANNELS = 3
    INPUT_SHAPE = (HEIGHT,WIDTH,CHANNELS)
    BASE_MODEL_INPUT_SHAPE = (600,600,3)
    
   
    tf.keras.backend.clear_session()
    
    #tf_config = {
    #    'cluster': {
    #        'worker': ['localhost:12345', 'localhost:23456','localhost:23457']
    #    },
    #    'task': {'type': 'worker', 'index': 0}
    #} 
    #os.environ["TF_CONFIG"] = json.dumps(tf_config)
    strategy = tf.distribute.MirroredStrategy(
                                              devices=None,
                                              cross_device_ops=None)
 
    # tf.distribute.experimental.MultiWorkerMirroredStrategy()
    


    cifar = tf.keras.datasets.cifar10.load_data()
    (x_train, y_train), (x_test, y_test) = cifar

    y_train_ohe = tf.one_hot([i[0] for i in  y_train],10)
    indexes_for_rows = tf.range(0,y_train.shape[0])
    shuffled_indexes = tf.random.shuffle(indexes_for_rows)
    selected_indexes = shuffled_indexes[:TRAINING_SET_SIZE]
    selected_x_train = x_train[selected_indexes,:,:,:]
    selected_y_train_ohe = y_train_ohe.numpy()[selected_indexes,:]
    
    # It looks weird that we are using a base model with 1000 classes.
    # To pull the model that is pre-trained onn imagenet, we have to do this
    # or we get a model with uninitilized weights. Not recommended for
    # transfer learning...
    with strategy.scope():
        # Base moel for transfer learning: (needs a few modifications...)
        mod_with_fc_raw = tf.keras.applications.efficientnet.EfficientNetB7(
            include_top=True, weights='imagenet', input_tensor=None,
            input_shape = BASE_MODEL_INPUT_SHAPE, pooling='max', classes=1000
        )

        # Make the deepest conv2d layer trainable, leave everything else
        # as not trainable
        for layer in mod_with_fc_raw.layers:
            layer.trainable = False
        # Last conv2d layer. This we want to train .
        mod_with_fc_raw.layers[-6].trainable = True

    
        efficient_net_b_7_transferable_base_model =\
            tf.keras.Model(inputs=mod_with_fc_raw.layers[0].input, 
                            outputs=mod_with_fc_raw.layers[-3].output)


        # build a tandem EfficientNetB7-ResidualMLP model using my ResidualMLP
        # package
        model=\
            make_model(learning_rate=LEARNING_RATE,
                        input_shape = INPUT_SHAPE,  
                        base_model= efficient_net_b_7_transferable_base_model,
                        base_model_input_shape = BASE_MODEL_INPUT_SHAPE,
                        flatten_after_base_model = True,
                        blocks = BLOCKS[BLOCKS_SELECTION],
                        b_norm_or_dropout_last_layers = 'bnorm',
                        dropout_rate=.2,
                        final_dense_layers =\
                            FINAL_DENSE_LAYERS_PERMUTATIONS[
                                FINAL_DENSE_LAYERS_SELECTION]
            )
    
    
    
            
    logdir = os.path.join("logs", RESULTS_DIR + '_TB')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, 
                                                          histogram_freq=1)
    

    history = model.fit(x=selected_x_train,  
                        y=selected_y_train_ohe, 
                        batch_size=BATCH_SIZE, 
                        epochs=EPOCHS,      
                        verbose='auto', 
                        callbacks=[tf.keras.callbacks.\
                                   EarlyStopping(monitor='val_loss',
                                                 patience=PATIENCE,
                                                 min_delta=PATIENCE_MIN_DELTA,
                                                 restore_best_weights=True),
                                tensorboard_callback], 
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


 
    hist_df = pd.DataFrame(history.history) 
    
    # save to json:  
    hist_json_file = f'{BEST_MODEL_DIR}-history.json' 
    with open(hist_json_file, mode='w',encoding='utf8') as f:
        hist_df.to_json(f)
    

    hist_csv_file = f'{BEST_MODEL_DIR}-history.csv'
    hist_df.to_csv(f)

    model.save(hist_csv_file)
    
   
