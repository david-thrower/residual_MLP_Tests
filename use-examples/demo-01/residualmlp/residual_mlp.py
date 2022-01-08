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
               residual_bypass_dense_layers = list(),
               b_norm_or_dropout_residual_bypass_layers = 'dropout',
               dropout_rate_for_bypass_layers = .35,
               b_norm_or_dropout_last_layers = 'dropout', # alternative 'bnorm'
               dropout_rate = .2,
               activation = tf.keras.activations.relu,
               final_dense_layers = [75,35],
               number_of_classes = 10, # 1 if a regression problem
               final_activation = tf.keras.activations.softmax,
               loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)):
    # Screen for nonsense combinations of the parameters 'blocks' and 
    # 'residual_bypass_dense_layers'
    if not isinstance(residual_bypass_dense_layers,list):
        raise ValueError("The parameter residual_bypass_dense_layers should "
                         "be one of these 2: 1. a 2d list, one 1d list of "
                         "integers for each list in blocks, or 2. an empty "
                         "list.")
    if len(residual_bypass_dense_layers) != 0:
        if len(blocks) != len(residual_bypass_dense_layers):
            raise ValueError("The parameter 'blocks' and "
                             "'residual_bypass_dense_layers are 2d arrays' "
                             "must have the same number of 1d arrays "
                             "nested within them OR be aan empty list or "
                             "left default. To fix this error do one "
                             "of the following: Fix 1: Make them the same "
                             "number of 1d arrays, for example "
                             "blocks = [[5,20,2],[5,20,2],[5,20,2]] ..."
                             " (3 nested 1d arrays) "
                             "residual_bypass_dense_layers = "
                             "[[10,7],[],[10]] also 3. "
                             "the ith array nested in "
                             "residual_bypass_dense_layers will be "
                             "associated with the i_th block in blocks. "
                             " each j_th item in the i_th nested 1d array "
                             "will make one dense layer in the tensor that "
                             "bypasses the main block of dense layers in the "
                             "ith block in the residual MLP. This is "
                             "probably a bit confusing to read. Please "
                             "refer to the tutorials and documentation. Note "
                             "the order I referred to the tutorials and "
                             "documentation in. Fix 2: leave "
                             "residual_bypass_dense_layers default / set it "
                             "to an empty list.")
    else:
        residual_bypass_dense_layers = [[] for block in blocks]
    
    #precision = tf.keras.metrics.Precision(), 
    #recall = tf.keras.metrics.Recall()
    #accuracy = tf.keras.metrics.Accuracy()
    #top_5 = tf.keras.metrics.TopKCategoricalAccuracy(k=5, 
    #                             name='top_5_categorical_accuracy', 
    #                             dtype=None)
    #top_4 = tf.keras.metrics.TopKCategoricalAccuracy(k=4, 
    #                                                 name='top_4_categorical_accuracy', 
    #                                                 dtype=None)
    #top_3 = tf.keras.metrics.TopKCategoricalAccuracy(k=3, 
    #                                                 name='top_3_categorical_accuracy', 
    #                                                 dtype=None)
    #top_2 = tf.keras.metrics.TopKCategoricalAccuracy(k=2, 
    #                             name='top_2_categorical_accuracy', 
    #                             dtype=None)
    #top_1 = tf.keras.metrics.TopKCategoricalAccuracy(k=1, 
    #                             name='top_1_categorical_accuracy', 
    #                             dtype=None)   
    rmse = tf.keras.metrics.RootMeanSquaredError() 
    metrics = [#precision,
               #recall,
               #accuracy,
               #top_5,
               #top_4,
               #top_3,
               #top_2,
               #top_1,
               rmse]
    # Use number_of_classes = 1 for a regression problem still. I paln to apply
    # automation to parameterize and selectively include all commonly used the metrics
    # that are valid for the problem type and number of classes ... 
    inp = tf.keras.layers.Input(shape = input_shape) 
    # Start with input layer that fits. 
    # The keras fucntional API will blow up appearently if
    # there is not an explicit input layer 
    # that coerces inputs as a specific size
    # quite annoying if you ask me, but
    # obviously Google didn't, so here 
    # we are ...
    if bw_images:
        x = grayscale_to_rgb(inp)
    else:
        x = inp
    if base_model != '':
        x = tf.keras.layers.Resizing(base_model_input_shape[0],
                                     base_model_input_shape[1])(x)
        x = base_model(x)
    if flatten_after_base_model:
        tf.keras.layers.Flatten()(x)
    initializer = tf.keras.initializers.GlorotNormal()
    for block, bypass_block in zip(blocks,residual_bypass_dense_layers):
        x = tf.keras.layers.Dense(block[1],
                                  activation,
                                  kernel_initializer=initializer)(x) 
        x = tf.keras.layers.BatchNormalization()(x) 
        # x proceeds sequentially to the 
        # next Dense layer.
        if b_norm_or_dropout_residual_bypass_layers == 'dropout':
            y = tf.keras.layers.Dropout(dropout_rate_for_bypass_layers)(x)
        elif b_norm_or_dropout_residual_bypass_layers == 'bnorm':
            y = tf.keras.layers.BatchNormalization()(x)
        else:
            raise ValueError("The parameter: "
                             "'b_norm_or_dropout_residual_bypass_layers'"
                             " must be left default '', or be "
                             "'dropout' or may be 'bnorm'.")
        for bypass_layer in bypass_block:
            y = tf.keras.layers.Dense(bypass_layer,
                                      activation,
                                      kernel_initializer=initializer)(y)
            if b_norm_or_dropout_residual_bypass_layers == 'dropout':
                y = tf.keras.layers.Dropout(dropout_rate_for_bypass_layers)(y)
            elif b_norm_or_dropout_residual_bypass_layers == 'bnorm':
                y = tf.keras.layers.BatchNormalization()(y)
            else:
                raise ValueError("The parameter: "
                                 "'b_norm_or_dropout_residual_bypass_layers'"
                                 " must be left default '', or be "
                                 "'dropout' or may be 'bnorm'.")
        # y does NOT proceed sequentially
        # to the next layer. This bupasses 
        # several layers and give a memory 
        # that attenuates some of the 
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
