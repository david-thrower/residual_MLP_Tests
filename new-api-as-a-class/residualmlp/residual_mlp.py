try:
    import keras_tuner as kt
except Exception as exc:
    print("Importing Keras tuner appears to be unsuccesful. "
          "keras tuner may need to be installed $ pip install -q -U "
          "keras-tuner. The auto-ml features are disabled until this is "
           "fixed, but ResidualMLP will work. A more detailed error is: "
           f"{exc}")
import tensorflow as tf
import numpy as np
import pandas as pd
import pendulum

# Becomes a layer to convert BW images to RGB

class ResidualMLP:
    
    def __init__(self, problem_type = 'classification',
                      learning_rate = .0007,
                      minimum_learning_rate = 0.00007,
                      maximum_learning_rate = 0.7,
                      number_of_learning_rates_to_try = 5,
                      input_shape = (32,32,3),
                      bw_images = False,
                      base_model = '',
                      base_model_input_shape = (600,600,3),
                      base_model_hyperparameters = {},
                      flatten_after_base_model = True,
                          # 2D ARRAY: Each row: 
                          # [number_of_layers,
                          #  first_layer_neurons, 
                          #  decay_of_n_Dense_units_per_layer]
                      blocks = [[5,400,50]],
                      minimum_number_of_blocks = 1,
                      maximum_number_of_blocks = 7,
                      minimum_number_of_layers_per_block = 1,
                      maximum_number_of_layers_per_block = 7,
                      minimum_neurons_per_block_layer = 3,
                      maximum_neurons_per_block_layer = 30,
                      n_options_of_neurons_per_layer_to_try = 7,
                      minimum_neurons_per_block_layer_decay = 1,
                      maximum_neurons_per_block_layer_decay = 7,
                      residual_bypass_dense_layers = list(),
                      b_norm_or_dropout_residual_bypass_layers = 'dropout',
                      dropout_rate_for_bypass_layers = .35,
                      minimum_dropout_rate_for_bypass_layers = 0.01,
                      maximim_dropout_rate_for_bypass_layers = 0.7,
                      n_options_dropout_rate_for_bypass_layers = 7,
                      inter_block_layers_per_block = list(),
                      minimum_inter_block_layers_per_block = 3,
                      maximum_inter_block_layers_per_block = 30,
                      n_options_inter_block_layers_per_block = 7,
                      b_norm_or_dropout_last_layers = 'dropout', # | 'bnorm'
                      dropout_rate = .2, #
                      minimum_dropout_rate = 0.01,
                      maximum_dropout_rate = 0.7,
                      n_options_dropout_rate = 7,
                      activation = tf.keras.activations.relu,
                      final_dense_layers = [75,35],
                      minimum_final_dense_layers = 0,
                      maximum_final_dense_layers = 30,
                      n_options_final_dense_layers = 2,
                      number_of_classes = 10, # 1 if a regression problem
                      final_activation = tf.keras.activations.softmax,
                      loss = tf.keras.losses.CategoricalCrossentropy(
                          from_logits=False)):
        self.problem_type = problem_type
        self.learning_rate = learning_rate
        self.minimum_learning_rate = minimum_learning_rate
        self.maximum_learning_rate = maximum_learning_rate
        self.number_of_learning_rates_to_try = number_of_learning_rates_to_try
        self.input_shape = input_shape
        self.bw_images = bw_images
        self.base_model = base_model
        self.base_model_input_shape = base_model_input_shape
        self.flatten_after_base_model = flatten_after_base_model
        self.blocks = blocks
        self.minimum_number_of_blocks = minimum_number_of_blocks
        self.maximum_number_of_blocks = maximum_number_of_blocks
        self.minimum_number_of_layers_per_block =\
            minimum_number_of_layers_per_block
        self.maximum_number_of_layers_per_block =\
            maximum_number_of_layers_per_block
        self.minimum_neurons_per_block_layer = minimum_neurons_per_block_layer
        self.maximum_neurons_per_block_layer = maximum_neurons_per_block_layer
        self.n_options_of_neurons_per_layer_to_try =\
            n_options_of_neurons_per_layer_to_try
        self.minimum_neurons_per_block_layer_decay =\
            minimum_neurons_per_block_layer_decay
        self.maximum_neurons_per_block_layer_decay =\
            maximum_neurons_per_block_layer_decay
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
        self.minimum_dropout_rate_for_bypass_layers =\
            minimum_dropout_rate_for_bypass_layers
        self.maximim_dropout_rate_for_bypass_layers =\
            maximim_dropout_rate_for_bypass_layers
        self.n_options_dropout_rate_for_bypass_layers =\
            n_options_dropout_rate_for_bypass_layers
        self.inter_block_layers_per_block = inter_block_layers_per_block
        self.minimum_inter_block_layers_per_block =\
            minimum_inter_block_layers_per_block
        self.maximum_inter_block_layers_per_block =\
            maximum_inter_block_layers_per_block
        self.n_options_inter_block_layers_per_block =\
            n_options_inter_block_layers_per_block
        self.b_norm_or_dropout_last_layers = b_norm_or_dropout_last_layers
        self.dropout_rate  = dropout_rate
        self.minimum_dropout_rate = minimum_dropout_rate
        self.maximum_dropout_rate = maximum_dropout_rate
        self.n_options_dropout_rate = n_options_dropout_rate
        self.activation = activation
        self.final_dense_layers = final_dense_layers
        self.minimum_final_dense_layers = minimum_final_dense_layers
        self.maximum_final_dense_layers = maximum_final_dense_layers
        self.n_options_final_dense_layers = n_options_final_dense_layers
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
        # The keras fucntional API requires an explicit input layer
        if self.bw_images:
            x = self.grayscale_to_rgb(inp)
        else:
            x = inp
        if self.base_model != '':
            x = tf.keras.layers.Resizing(self.base_model_input_shape[0],
                                         self.base_model_input_shape[1])(x)
            x = self.base_model(x)
        if self.flatten_after_base_model:
            x = tf.keras.layers.Flatten()(x)
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
            
            if bl != np.arange(len(self.blocks)).max():
                for inter_block_layer in self.inter_block_layers_per_block:
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
    
    
    def parse_block(self,number_of_blocks,
                        layers_per_block,
                        neurons_per_block_layer,
                        neurons_per_block_layer_decay):
        blocks_0 = []
        for i in np.arange(number_of_blocks):
            block_0 = [layers_per_block,
                       neurons_per_block_layer,
                       neurons_per_block_layer_decay]
            blocks_0.append(block_0)
        return blocks_0
    
    def build_auto_residual_mlp(self,hp):
        
        self.learning_rate = hp.Float(name='learning_rate',
                                      min_value=self.minimum_learning_rate, 
                                      max_value=self.maximum_learning_rate,
                                      sampling='log')

        permutations_of_blocks_array = np.array([[i,j,k,l]
         for i in np.arange(self.minimum_number_of_blocks,
                            self.maximum_number_of_blocks + 1)
         for j in np.arange(self.minimum_number_of_layers_per_block,
                            self.maximum_number_of_layers_per_block + 1)
         for k in np.linspace(self.minimum_neurons_per_block_layer,
                              self.maximum_neurons_per_block_layer, 
                              self.n_options_of_neurons_per_layer_to_try,
                              dtype=int)
         for l in np.arange(self.minimum_neurons_per_block_layer_decay,
                            self.maximum_neurons_per_block_layer_decay + 1)])
        
        permutations_of_blocks_df = pd.DataFrame(permutations_of_blocks_array)
        permutations_of_blocks_df.columns = ['number_of_blocks',
                                             'layers_per_block',
                                             'neurons_per_block_layer',
                                             'neurons_per_block_layer_decay']
        print("All permutations:")
        print(permutations_of_blocks_df)

        # Filter out any invalid permutations that would try creating Dense
        # layers with 0 units or a negative number of units.
        permutations_of_blocks_df['valid_block'] =\
            permutations_of_blocks_df['neurons_per_block_layer'] >\
            permutations_of_blocks_df["layers_per_block"] *\
            permutations_of_blocks_df["neurons_per_block_layer_decay"]
        valid_permutations_df =\
            permutations_of_blocks_df.query("valid_block == True")\
            .reset_index(drop=True)
        print("Valid permutations")
        print(valid_permutations_df)
        
        list_of_blocks_args = list()
        for i in np.arange(valid_permutations_df.shape[0]):
            blocks_arg = self.parse_block(
                valid_permutations_df.loc[i]['number_of_blocks'],
                valid_permutations_df.loc[i]['layers_per_block'],
                valid_permutations_df.loc[i]['neurons_per_block_layer'],
                valid_permutations_df.loc[i]['neurons_per_block_layer_decay'])
            list_of_blocks_args.append(blocks_arg)
        
        valid_permutations_df['blocks'] = list_of_blocks_args
        
        print("Valid permutations with blocks column")
        print(valid_permutations_df)
 
        valid_permutations_df.sort_values(['layers_per_block',
                                           'neurons_per_block_layer'],
                                            ascending=True)\
            .reset_index(drop=True)
        
        # for reeference, the list of block options is saved as a csv
        date = pendulum.now().__str__()[:16].replace("T","_").replace(":","_")
        valid_permutations_df.to_csv(f'{date}_blocks_permutations.csv')
        
        list_to_choose_blocks_option_from =\
            [int(i) for i in np.arange(valid_permutations_df.shape[0])]
        blocks_index_chosen = hp.Choice(
                    name='blocks',
                    values=list_to_choose_blocks_option_from,
                    ordered=True)
        
        
        self.blocks = valid_permutations_df.loc[blocks_index_chosen]['blocks']
        print(self.blocks)
        
        bypass_layers_units = hp.Choice(name='bypass_layers_units',
                        values=[int(i) 
                                for i in np.linspace(
                                self.minimum_neurons_per_block_layer ,
                                self.maximum_neurons_per_block_layer,
                                self.n_options_of_neurons_per_layer_to_try,
                                dtype=int)],
                        ordered=True)
        
        if bypass_layers_units == 0:
            self.residual_bypass_dense_layers =\
                [list() for _ in np.arange(len(self.blocks))]
        else:
            self.residual_bypass_dense_layers =\
                [[bypass_layers_units] for _ in np.arange(len(self.blocks))]
        
        inter_block_layers_per_block_options =\
            np.linspace(self.minimum_inter_block_layers_per_block,
                        self.maximum_inter_block_layers_per_block,
                        self.n_options_inter_block_layers_per_block,
                        dtype=int) 
        inter_block_layers_per_block_choice =\
            hp.Choice(name='inter_block_layers',
                      values=inter_block_layers_per_block_options,
                      ordered=True)
        if inter_block_layers_per_block_choice == 0:
            self.inter_block_layers_per_block = list()
        else:
            self.inter_block_layers_per_block =\
                [inter_block_layers_per_block_choice] #Add for i in range max interblock layers...

        final_dense_layers_options = \
            np.linspace(self.minimum_final_dense_layers,
                        self.maximum_final_dense_layers,
                        self.n_options_final_dense_layers,
                        dtype=int)
        final_dense_layers_choice = hp.Choice(
                                            name='final_dense_layers',
                                            values=final_dense_layers_options,
                                            ordered=True)
        if final_dense_layers_choice == 0:
            self.final_dense_layers = []
        else:
            self.final_dense_layers = [final_dense_layers_choice]

        self.b_norm_or_dropout_residual_bypass_layers =\
            hp.Choice(name="b_norm_or_dropout_residual_bypass_layers",
                      values=['dropout','bnorm'],
                      ordered=False)
        dropout_rate_for_bypass_layers_choices =\
            np.linspace(self.minimum_dropout_rate_for_bypass_layers,
                        self.maximim_dropout_rate_for_bypass_layers,
                        self.n_options_dropout_rate_for_bypass_layers,
                        dtype=float)
        self.dropout_rate_for_bypass_layers =\
            hp.Choice(name='dropout_rate_for_bypass_layers',
                      values=dropout_rate_for_bypass_layers_choices,
                      ordered=True)

        self.b_norm_or_dropout_last_layers =\
                        hp.Choice(name='b_norm_or_dropout_last_layers',
                                  values=['dropout','bnorm'],
                                  ordered=False)
        
        dropout_rate_options = np.linspace(self.minimum_dropout_rate, 
                                                self.maximum_dropout_rate,
                                                self.n_options_dropout_rate,
                                                dtype=float)
        self.dropout_rate = hp.Choice(name='dropout_rate',
                                      values=dropout_rate_options,
                                      ordered=True)

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
        # The keras fucntional API requires an explicit input layer
        if self.bw_images:
            x = self.grayscale_to_rgb(inp)
        else:
            x = inp
        if self.base_model != '':
            x = tf.keras.layers.Resizing(self.base_model_input_shape[0],
                                         self.base_model_input_shape[1])(x)
            x = self.base_model(x)
        if self.flatten_after_base_model:
            x = tf.keras.layers.Flatten()(x)
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
            
            if bl != np.arange(len(self.blocks)).max():
                for inter_block_layer in self.inter_block_layers_per_block:
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
