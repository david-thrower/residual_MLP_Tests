import tensorflow as tf
import numpy as np

# Becomes a layer to convert BW images to RGB

class ResidualMLP:
    
    def __init__(self, problem_type = 'classification', #
                      learning_rate = .0007, #
                      input_shape = (32,32,3), #
                      bw_images = False, #
                      base_model = '', #
                      base_model_input_shape = (600,600,3), #
                      flatten_after_base_model = True, #
                          # 2D ARRAY: Each row: 
                          # [number_of_layers,
                          #  first_layer_neurons, 
                          #  decay_of_n_Dense_units_per_layer]
                      blocks = [[5,400,50]], #
                      residual_bypass_dense_layers = list(), #
                      b_norm_or_dropout_residual_bypass_layers = 'dropout', #
                      dropout_rate_for_bypass_layers = .35, #
                      inter_block_layers_per_block = list(),
                      b_norm_or_dropout_last_layers = 'dropout', # | 'bnorm'
                      dropout_rate = .2, #
                      activation = tf.keras.activations.relu, #
                      final_dense_layers = [75,35], #
                      number_of_classes = 10, # 1 if a regression problem
                      final_activation = tf.keras.activations.softmax, #
                      loss = tf.keras.losses.CategoricalCrossentropy(
                          from_logits=False)): #
        self.problem_type = problem_type
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.bw_images = bw_images
        self.base_model = base_model
        self.base_model_input_shape = base_model_input_shape
        self.flatten_after_base_model = flatten_after_base_model
        self.blocks = blocks
        # residual_bypass_dense_layers
        # Screen for nonsense combinations of the parameters 'blocks' and 
        # 'residual_bypass_dense_layers'
        if not isinstance(residual_bypass_dense_layers,list):
            raise ValueError("The parameter residual_bypass_dense_layers "
                             "should be one of these 2: 1. a 2d list, one 1d "
                             "list of positive    integers for each list in "
                             "blocks, or 2. an empty list.")
        if len(residual_bypass_dense_layers) != 0:
            if len(blocks) != len(residual_bypass_dense_layers):
                raise ValueError("The parameter 'blocks' and "
                                 "'residual_bypass_dense_layers are 2d "
                                 "arrays' must have the same number of 1d "
                                 "arrays nested within them OR be aan empty "
                                 "list or left default. To fix this error "
                                 "do one of the following: Fix 1: Make them " 
                                 "the same number of 1d arrays, for example "
                                 "blocks = [[5,20,2],[5,20,2],[5,20,2]] ..."
                                 " (3 nested 1d arrays) "
                                 "residual_bypass_dense_layers = "
                                 "[[10,7],[],[10]] also 3. "
                                 "the ith array nested in "
                                 "residual_bypass_dense_layers will be "
                                 "associated with the i_th block in blocks. "
                                 " each j_th item in the i_th nested 1d "
                                 "array will make one dense layer in the "
                                 "tensor that bypasses the main block of "
                                 "dense layers in the ith block in the "
                                 "residual MLP. This is probably a bit "
                                 "confusing to read. Please refer to the "
                                 "tutorials and documentation. Note "
                                 "the order I referred to the tutorials and "
                                 "documentation in. Fix 2: leave "
                                 "residual_bypass_dense_layers default / "
                                 "set it to an empty list.")
            else:
                self.residual_bypass_dense_layers =\
                    residual_bypass_dense_layers
        else:
            self.residual_bypass_dense_layers = [[] for block in blocks]
        self.b_norm_or_dropout_residual_bypass_layers =\
            b_norm_or_dropout_residual_bypass_layers
        self.dropout_rate_for_bypass_layers = dropout_rate_for_bypass_layers
        self.inter_block_layers_per_block = inter_block_layers_per_block
        self.b_norm_or_dropout_last_layers = b_norm_or_dropout_last_layers
        self.dropout_rate  = dropout_rate
        self.activation = activation
        self.final_dense_layers = final_dense_layers
        self.number_of_classes = number_of_classes
        self.final_activation = final_activation
        self.loss = loss
        
    
    def grayscale_to_rgb(images, channel_axis=-1):
        images= tf.expand_dims(images, axis=channel_axis)
        tiling = [1] * 4    # 4 dimensions: B, H, W, C
        tiling[channel_axis] *= 3
        images= tf.tile(images, tiling)
        im = tf.keras.preprocessing.image.smart_resize(images,(224,224))
        return im


    # builds and compiles a tandem model given these params and
    # selected base model:
    def make_tandem_model(self):
        if self.problem_type == 'classification':
            precision = tf.keras.metrics.Precision(), 
            recall = tf.keras.metrics.Recall()
            accuracy = tf.keras.metrics.Accuracy()
        if self.problem_type == 'classification' and\
                self.number_of_classes > 1:
            metrics = [tf.keras.metrics.TopKCategoricalAccuracy(
                k = k,
                name=f'top_{k}_'
                'categorical_'
                'accuracy',
                dtype=None)
                           for k in np.arange(1,self.number_of_classes)\
                               if k < 10]
            metrics.append(precision)
            metrics.append(recall)
            metrics.append(accuracy)
        elif self.problem_type == 'classification' and\
                self.number_of_classes == 1:    
            metrics = [precision, recall, accuracy]
        else:
            rmse = tf.keras.metrics.RootMeanSquaredError()
            mae = tf.keras.metrics.MeanAbsoluteError()
            metrics = [rmse, mae]
    
        inp = tf.keras.layers.Input(shape = self.input_shape) 
        # Start with input layer that fits. 
        # The keras fucntional API will blow up appearently if
        # there is not an explicit input layer 
        # that coerces inputs as a specific size
        # quite annoying if you ask me, but
        # obviously Google didn't, so here 
        # we are ...
        if self.bw_images:
            x = self.grayscale_to_rgb(inp)
        else:
            x = inp
        if self.base_model != '':
            x = tf.keras.layers.Resizing(self.base_model_input_shape[0],
                                         self.base_model_input_shape[1])(x)
            x = self.base_model(x)
        if self.flatten_after_base_model:
            tf.keras.layers.Flatten()(x)
        initializer = tf.keras.initializers.GlorotNormal()
        for bl in np.arange(len(self.blocks)):
            block = self.blocks[bl]
            bypass_block = self.residual_bypass_dense_layers[bl]
            
            
            x = tf.keras.layers.Dense(block[1],
                                      self.activation,
                                      kernel_initializer=initializer)(x)
            y = x
            x = tf.keras.layers.BatchNormalization()(x)
            # x proceeds sequentially to the 
            # next Dense layer.
            
            if self.b_norm_or_dropout_residual_bypass_layers == 'dropout':
                y = tf.keras.layers\
                    .Dropout(self.dropout_rate_for_bypass_layers)(y)
            elif self.b_norm_or_dropout_residual_bypass_layers == 'bnorm':
                y = tf.keras.layers.BatchNormalization()(y)
            else:
                raise ValueError("The parameter: "
                                 "'b_norm_or_dropout_residual_bypass_"
                                 "layers'"
                                 " must be left default '', or be "
                                 "'dropout' or may be 'bnorm'.")
            for bypass_layer in bypass_block:
                y = tf.keras.layers.Dense(bypass_layer,
                                          self.activation,
                                          kernel_initializer=initializer)(y)
                if self.b_norm_or_dropout_residual_bypass_layers == 'dropout':
                    y = tf.keras.layers\
                        .Dropout(self.dropout_rate_for_bypass_layers)(y)
                elif self.b_norm_or_dropout_residual_bypass_layers == 'bnorm':
                    y = tf.keras.layers.BatchNormalization()(y)
                else:
                    raise ValueError("The parameter: "
                                     "'b_norm_or_dropout_residual_bypass_"
                                     "layers' must be left default '', or be "
                                     "'dropout' or may be 'bnorm'.")
            # y does NOT proceed sequentially
            # to the next layer. This bypasses 
            # several layers and give a memory 
            # that attenuates some of the 
            # deleterious effects of a deeper 
            # network and lets us capture more 
            # complex interactions before 
            # overfitting becomes an issue than 
            # the textbook sequential multi - 
            # layer perceptron ...
            for j in np.arange(block[0]): 
                x = tf.keras.layers.Dense(block[1] - block[2] * j,
                                          self.activation,
                                          kernel_initializer=initializer)(x) 
                x = tf.keras.layers.BatchNormalization()(x)
    
            x = tf.keras.layers.Concatenate(axis=1)([x, y])
            
            if block != np.arange(len(self.blocks)).max():
                for inter_block_layer in self.inter_block_layers:
                    x = tf.keras.layers.Dense(inter_block_layer,
                                          self.activation,
                                          kernel_initializer=initializer)(x)
                    x = tf.keras.layers.BatchNormalization()(x)
    
        for i in self.final_dense_layers:
            x = tf.keras.layers.Dense(i,
                                      self.activation,
                                      kernel_initializer=initializer)(x) 
            if self.b_norm_or_dropout_last_layers == 'bnorm':
                x = tf.keras.layers.BatchNormalization()(x)
            elif self.b_norm_or_dropout_last_layers == 'dropout':
                x = tf.keras.layers.Dropout(self.dropout_rate)(x)
            else:
                raise ValueError("For b_norm_or_dropout_last_layers, " 
                                 "you must pick either 'dropout' or 'bnorm'")
        out = tf.keras.layers.Dense(self.number_of_classes,
                                    self.final_activation,
                                    kernel_initializer=initializer)(x)
    
        # Declare the graph for our model ...
        modelo_final = tf.keras.Model(inputs=inp,outputs = out)
        
        modelo_final\
            .compile(optimizer=\
                     tf.keras.optimizers.Adam(
                         learning_rate=self.learning_rate, 
                         clipnorm=1.0),
                         loss=self.loss, 
                         metrics=metrics)
        return modelo_final

