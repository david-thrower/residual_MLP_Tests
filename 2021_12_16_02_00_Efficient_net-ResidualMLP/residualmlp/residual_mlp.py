import tensorflow as tf
import numpy as np

# Becomes a layer to convert BW images to RGB


def grayscale_to_rgb(images, channel_axis=-1):
    images= tf.expand_dims(images, axis=channel_axis)
    tiling = [1] * 4    # 4 dimensions: B, H, W, C
    tiling[channel_axis] *= 3
    images= tf.tile(images, tiling)
    im = tf.keras.preprocessing.image.smart_resize(images,(224,224))
    return im


# builds and compiles a model given these params:
def make_model(learning_rate = .0007,
               input_shape = (32,32,3),
               bw_images = False,
               base_model = '',
               base_model_input_shape = (600,600,3),
               flatten_after_base_model = True,
               blocks = [[5,400,50]], # 2D ARRAY: Each row: [number_of_layers,first_layer_neurons, decay_coefficient]
               b_norm_or_dropout_last_layers = 'dropout', # alternative 'bnorm'
               dropout_rate = .2,
               activation = tf.keras.activations.relu,
               final_dense_layers = [75,35],
               number_of_classes = 10, # 1 if a regression problem
               final_activation = tf.keras.activations.softmax,
               loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)):
    
    precision = tf.keras.metrics.Precision(), 
    recall = tf.keras.metrics.Recall()
    accuracy = tf.keras.metrics.Accuracy()
    metrics = [precision,recall,accuracy]
            # use number_of_classes = 1 for a regression problem
    inp = tf.keras.layers.Input(shape = input_shape) # Start with input layer that fits 
                                                 # fucntional API will blow up if
                                                 # there is not an input layer 
                                                 # that coerces a specific size ...
                                                 # quite annoying if you ask me, but
                                                 # obviously Google didn't so here 
                                                 # we are ...
    if bw_images:
        x = grayscale_to_rgb(inp) # In (28,28) out (None,224, 224, 3)
    else:
        x = inp
    if base_model != '':
        x = tf.keras.layers.Resizing(base_model_input_shape[0],
                                     base_model_input_shape[1])(x)
        x = base_model(x)         # In (None,224, 224, 3) out (None, 2048)  (Flattened)
    if flatten_after_base_model:
        tf.keras.layers.Flatten()(x)
    initializer = tf.keras.initializers.GlorotNormal()
    for block in blocks:
        x = tf.keras.layers.Dense(block[1],
                                  activation,
                                  kernel_initializer=initializer)(x) 
        x = tf.keras.layers.BatchNormalization()(x) 
        # x proceeds sequentially to the 
        # next Dense layer.
        y = tf.keras.layers.BatchNormalization()(x) 
        # y does NOT proceed sequentially
        # to the next layer. This bupasses 
        # several layers and give a memory 
        # that atenuates some of the 
        # deleterious effects of a deeper 
        # network and lets us capture more 
        # complex interactions before 
        # overfitting becomes an issue than 
        # the textbook sequential multi - 
        # layer perceptron ...
        for j in np.arange(block[0]): # Parameterize this as a hyperparameter later
            x = tf.keras.layers.Dense(block[1] - block[2] * j,
                                      activation,
                                      kernel_initializer=initializer)(x) 
                            # In (None,500) out (None,300 - 50 * i)
            x = tf.keras.layers.BatchNormalization()(x)
                            # In (None,200 - 50 * i) out (None,200 - 50 * i)

        # New, test out keras layer for concat
        x = tf.keras.layers.Concatenate(axis=1)([x, y])
        # x = tf.concat([x,y],axis = 1) # In: [(None,100),(None,700)]  out: (None,800)
        x = tf.keras.layers.Dense(block[1] - block[2] * block[0],
                                  activation,
                                  kernel_initializer=initializer)(x) # In: (None,800)
        x = tf.keras.layers.BatchNormalization()(x)
        
    for i in final_dense_layers:
        x = tf.keras.layers.Dense(i,
                                  activation,
                                  kernel_initializer=initializer)(x) 
        if b_norm_or_dropout_last_layers == 'bnorm':
            x = tf.keras.layers.BatchNormalization()(x)
        elif b_norm_or_dropout_last_layers == 'dropout':
            x = tf.keras.layers.Dropout(dropout_rate)(x)
        else:
            raise ValueError("For b_norm_or_dropout_last_layers, must pick either 'dropout' or 'bnorm'")
    out = tf.keras.layers.Dense(number_of_classes,
                                final_activation,
                                kernel_initializer=initializer)(x)# In: (None,35) out: (None,10)
                                                            # One hot encoded
# Declare the graph for our model ...
    modelo_final = tf.keras.Model(inputs=inp,outputs = out)
    
    modelo_final.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = learning_rate, clipnorm=1.0),
                     loss=loss, 
                     metrics=metrics)
    return modelo_final
