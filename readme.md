#  Residual Multi Layer Perceptron build and fit engine written in python 10.0.0 using Tensorflow 2.6 functional API. 
## This is useful for building residual MLP models both as stand alone residual MLP models and as tandem [convolutional | vision transformer | etc]->Residual MLP models where the resudual MLP aspect of the model is used to augment the performance of a commonly used base model for transfer learning (e.g. EffieientNet/ResnetNet/etc.).

1. Purpose: This is a core engine for recursively building, compiling, and testing Residual Multi Layer Perceptron models. This is meant to enable neural architecture search in this non-standard and non-sequential MLP architecture. This system abstracts away most of the tedious work of building residual multi layer perceptron models and reduces it to a selection of any suitable permutations of this neural architecture constructor's hyperparameters.
2. What is a residual multi layer perceptron? It is a non-sequential multi-layer-perceptron where as with all multi layer perceptrons, there is a block of sequential Dense layers (basically the layers we see below in blue), however, unlike a standard sequential multi - layer - perceptron, there is also a "residual tensor" (seen below in the layers in yellow) which forwards a duplicate of the input to this block of dense layers forward, bypassing the block of dense layers. At the output end of the block of dense layers, the residual tensor is concatenated with the output of the block of sequential dense layers before proceeding to the next block. This creates a memory effect and importantly, attenuates some of the deleterious effects introduced by very deep networks, such as overfitting, [exploding | valishing] gradients, and internal covariate shift. 
   ![/assets/residual_mlp_summary.drawio.png](/assets/residual_mlp_summary.drawio.png)
3. Why use a residual multi layer perceptron model?
    1. Better performance on small data sets.
    2. Residual MLPs enable greater model complexity and insanely deep neural network depth before overfitting, valishing and exploding gradients, and internal covariate shift become the model complexity limiting factors.
    3. Better precision, recall, and other performance metrics with less training data and far fewer epochs.
4. Are there any drawbacks with using a residual MLP model?
    1. They are a little computationallly expensive on an epoch - by epoch basis. However considering that this model architecture does enable great performance after training on considerably smaller data sets and for far fewer epochs, they do make up for this and then some.
    2. Although they can train quickly, for the time they are training, they do need larger hardware configurations consistent with the scale of their model complexity in order to complete training successfully. For example, when training a models on a subset of the CIFAR10 dataset, using the pre-trained EfficientNetB7 (having only the last conv2d layer set to trainable and the output of layers[-3] diverted into the input layer of a de-novo ResidualMLP model with optimal residual MLP neural network architecture hyperparameter settings for this problem), I found it will usually exhaust the RAM on hardware having 45GB of RAM, unless it is run on a container allocated with at least 2 A4000 GPUs. With this said, it was trained on a subset of only 10% of the training set minus a 30% validation split (trained on 3500 training images with 1500 validation images). This reached 89.6% top-1-val-accuracy at only 33 epochs, with 1 hour, 3 min, and 35 sec of training. Also, this specific case of an EfficientNetB7-> 14-layer-residual-MLP is by design an extreme case intended to pressure test this architecture and neural architecture automation. Most residual MLP applications should consume considerably less resources, but still more than the garden variety small sequential MPL model. This can make some jobs not financially worth the cost of training them. However, for most training jobs that have failed due to small sample size / not enough data (the niche of this model architecture), this may be an ideal overall algorithm for your business problem.
5. Use example:
    Under construction... (a more user friendly example is forthcoming).
    1. VisionTransformer->ResidualMLP model:
        1. The VisionTransformer base model used her is forked from Khalid Salama's work here: https://keras.io/examples/vision/image_classification_with_vision_transformer/
        2. With the removal of the training MLP and a reduction of the number of transformer_layers, you have a base model amenable to serve as a base model for a tandem VisionTransformer->ResidualMLP model.
        3. The construction of the VIT - ResidualMLP model is demonstrated here https://github.com/david-thrower/residual_MLP_Tests/blob/main/use-examples/VisionTransformer-ResidualMLP/vit-ResidualMLP-image-classifier.ipynb. This is a proof of concept, but hopefully with some hyperparameter optimization, we can release a new state of the art model. 
    2. Transfer learning on EfficientNetB7, previously trained on ImageNet, augmented with a 14 layer residual MLP). These are all components of the example. task_trigger.py and task.py are the main files. Unfortunately, with this generating several MB of logs and training for ~ an hour, it was not practical to run this in a Jupyter notebook, but here is the cascade of the training pipeline for this task: From the shell: 1. the shell command `python3 task_trigger.py` generates and runs run_2021-12-27_07-34_job.sh. 2. run_2021-12-27_07-34_job.sh runs task.py with the selected arguments. Lastly, task.py imports and uses the package residualmlp, as well as the training data and base model. These are the files below:
        1. This is the task trigger and hyperparameter selection for the job: https://github.com/david-thrower/residual_MLP_Tests/blob/add-use-example/use-examples/CIFAR10_Small_Training_Subset/task_trigger.py
        2. The self-generated, self-executing shell script created by task_trigger.py: https://github.com/david-thrower/residual_MLP_Tests/blob/add-use-example/use-examples/CIFAR10_Small_Training_Subset/run_2021-12-27_07-34_job.sh        
        3. Task which uses the python package residualmlp: https://github.com/david-thrower/residual_MLP_Tests/blob/add-use-example/use-examples/CIFAR10_Small_Training_Subset/task.py 
        4. The version of the python package residualmlp used in this example: https://github.com/david-thrower/residual_MLP_Tests/tree/add-use-example/use-examples/CIFAR10_Small_Training_Subset/residualmlp
        5. The model fit history exported as a csv: https://github.com/david-thrower/residual_MLP_Tests/blob/add-use-example/use-examples/CIFAR10_Small_Training_Subset/2021-12-27_07-34_lr0007_blocks1_final_layers3_flatten_dropout_ExportedModel-history.csv
        6. The shell logs from the run (This is a large file that can't be displayed in the browser. You must download the file to view.): https://github.com/david-thrower/residual_MLP_Tests/blob/add-use-example/use-examples/CIFAR10_Small_Training_Subset/2021-12-27_07-34_lr0007_blocks1_final_layers3_flatten_dropout_python3_shell_log.txt
        7. The exported model may be added later. This is ~ 295.9 MB in size, which considerably larger than the 100 MB maximum file size that Github can allow for commits. 
6. API:
    1. residualmlp is a python package
    2. residualmlp.residual_mlp.ResidualMLP is a class that currently provides 1 method, which easily buils and compiles a Keras model for you. It can build a stand alone residual MLP model either for classification or regression tasks or it can build a tandem model from nearly any base model, such as a pre-traineed image classifier.
    3. To build a ResidualMLP model:
        1. First make sure that the python package residualmlp is in the directory you are working in. This is found here https://github.com/david-thrower/residual_MLP_Tests/tree/main/new-api-as-a-class, just copy the directory residualmlp to the directory you are running your code or notebook in:  (until soon I will release this on pypi so you can install it using pip).  
        2. Import ResidualMLP: `from residualmlp.residual_mlp import ResidualMLP`
        3. Instantiate a ResidualMLP obkect `model_builder = ResidualMLP(args)` This may look like:
        ```python3
        {
        model_builder = ResidualMLP(problem_type = 'classification', #
                      learning_rate = .0007, #
                      input_shape = (32, 32, 3), #(32,32,3), #
                      bw_images = False, #
                      base_model = base_vit_model, #
                      base_model_input_shape = (32, 32, 3),  # (600,600,3), #
                      flatten_after_base_model = True, #
                      blocks = [[7, 75, 8], [5, 75, 10]], #
                      residual_bypass_dense_layers = [[5],[5]], #
                      b_norm_or_dropout_residual_bypass_layers = 'dropout', #
                      dropout_rate_for_bypass_layers = .7, #
                      inter_block_layers_per_block = [10],
                      b_norm_or_dropout_last_layers = 'dropout', # | 'bnorm'
                      dropout_rate = .18, #
                      activation = tf.keras.activations.relu, #
                      final_dense_layers = [15], #
                      number_of_classes = 10, # 1 if a regression problem
                      # final_activation = tf.keras.activations.softmax, #
                      #loss = tf.keras.losses.CategoricalCrossentropy(
                      #    from_logits=False)
                     )
        }
        ```
        4. Call the method make_tandem_model(). It will return a Keras model ready to be fit:
        ```python3
        {
        final_residual_mlp = model_builder.make_tandem_model()
        }
        ```
        5. Train the model using its .fit() method as you would with any Keras model.
        6.  Arguments:
            1. learning_rate: The learning rate for the optimizer.
            2. input_shape: The input shape for the model as a whole, In other words, the shape of one observation in your data. 
            3. base_model: If you were building a tandem model, starting with a pre-trained base model and passing the output of that model to the residual MLP we are building, then you would pass in that keras model object (e.g. an instantiated EfficientNetB7 model that you pulled from Kerras Applications and removed the flinal Dense layer or a BERT text embedding puleld from Tensorflow Hub.). If you are building a stand - alone ResidualMLP model with no base model feeding into it, base_model should be set to empty string '':
            4. base_model_input_shape: If you are using a base model with a different input shape than your data and your data can be re-scaled (e.g. images that are a different size then the input layer on your base model), you would enter the input shape expected by the base model here, and the model built by make_tandem_model() will automatically insert a rescaling layer before between your input layer and your base model. For example, if you have training / test images of shape (32,32,3) and were using EfficientNet, pretrained on imagenet from keras applications having an input shape of (600,600,3), as your base_model, you would set the parameter 'base_model_input_shape' to (600,600,3).
            5. flatten_after_base_model: Whether or not to put a Flatten layer before the model send your data through the residual multi layer perceptron (often used when base model outputs images or other 2D / 3D / nD tensors). This is usually set to True if you have a Conv2d layer as the last layer of your base model or otherwise have any data that will pass to your residualMLP model that is not a rank 1 tensor. Settign it to true will coerce the data being fed into your residualMLP model to a rank tensor. If your model raises an exception from the Concat layer(s) in the residualMLP model, this is a sign that you may need to set this parameter to True.
            6. blocks: A 2d array. Each ith nested array will create a residualMLP block. See the diagram above for a visual of what it wil build. In each ith nested 1D array, you will find 3 positive integers (except l which can be positive or may be 0): j,k,l (from left to right). The positive integer j on the left of each nested array gives you control of how many Dense layers that this block will consist of. The second positive integer, k sets the number of Dense units in the first layer of the block. The third [0 or positive integer] l is how many LESS Dense units each succesive layer in the block will consist of than its predecessor. There is one additional obvious rule that the product of the first and third numbers, j and l must be < k, the second number, otherwise, you are asking the API to add some layer(s) with O Dense units or a negative number of Dense units. This will of course raise an exception and make you feel as embarassed as I did the first time I did this 😳. As a reminder, you must be concious of this when trying to run an auto-ml algorithym or running a gridsearch over permutations of options for blocks. A try .. except ... or better yet, a pre-screening of these permutations will be needed.
            7. residual_bypass_dense_layers: Inserts (a) Dense layer(s) in the residual bypass path (This is the yellow path on the right in the diagram above.). If this is left as the default of '', then there will be no blocks in any of the residual bypass paths. If you do set this, there must be one nested list of positive integers for each Each block (same number of nested 1D arrays as the hyperparameter blocks). Each ith nested 1d array would control the Dense layers in the residual bypass for the ith residual block. One layer will be inserted for each positive integer in the nested list and each layer will have the number of units as the integer.  You may add layers to one blocks's bypass and not the other using an empty lsit for the block you don't want to have a Dense unit on its residual bypass path by doing something like this: [[],[5]] The first nested 1d array this will insert a Dense layer i units in the residual bypass of residual Block 1 for each number in that 1D array (in this example none). The second nested array will insert i units in the resisual bypass of block 2 for each number in the array (in this example, one Desnse layer with 5 units - Dense(5,...)) and so forth.
            8. b_norm_or_dropout_residual_bypass_layers: You may insert BatchNormalization or dropout layers after each layer of the residual bypass. Options: Default "dropout" | "bnorm".
            9. dropout_rate_for_bypass_layers: The dropout rate for the dropout laters in the RESIDUAL BYPASSES (ignored if B_NORM_OR_DROPOUT_RESIDUAL_BYPASS_LAYERS is set to 'bnorm'). This usually performs best as dropout and often with a higher DROPOUT_RATE_FOR_BYPASS_LAYERS (e.g. 25% or gerater).
            10. inter_block_layers_per_block: A 1d array of positive integers. Each ith positive integer adds one Dense layer with i units in each break between residualMLP blocks.
            11. b_norm_or_dropout_last_layers: After all last residual block, the hyperparameter FINAL_DENSE_LAYERS allows you to add a series (or just one) Dense layer(s). The hyperparameter b_norm_or_dropout_last_layers controls whether there will be a Dropout or BatchNormalization layer after each of these layer(s). This defaults to 'dropout' but can be set to 'bnorm'. It seems that about 60% of the time, dropout with the right dropout_rate will perform best, but be advised that these layers are more apt to internal covariate shift than the residual bypass layers. You amy want to experiment with both.
            12. dropout_rate: The dropout rate for the dropout layers after each final Dense layer inserted after the last residualMLP block by the parameter
            13. final_dense_layers: A 1d array of positive integers: Each ith positive integer will insert a Dense layer after the last ResidualMLP block.
            14. number_of_classes = how many Dense units the final layer of the network should consist of / also the number of classes in your labels. For example, a simple linear regression model would have this parameter set to 1. So would a binary logistic regression problem. For a multi - class - classification problem, it would be the number of classes, eg 10 for a classification problem where there are 10 possible classes.
            15. final_activation - The activation fucntion e.g. tf.keras.activations.sigmoid - [no parentheses after it] if you are doing binary classification. None if you are doing regression. The defaults is tf.keras.activations.softmax since most problems are multi class classification.
            16. loss - The loss that is appropriate for your problem. E.g. if we are doign simple linear regression, the default choice is usually tf.keras.losses.MeanSquaredError(), but tf.keras.losses.Huber() and tf.keras.losses.MeanAbsoluteError() are fair game in regression cases and one may perform better than the other depending on the distribution in the residuals against the domain of your data. For binary classification, you might set this as tf.keras.losses.BinaryCrossentropy. The default loss is tf.keras.losses.CategoricalCrossentropy(from_logits=False), since most problems are multi-class classification.

7. License: Licensed under a modified MIT license, but with the following exclusions (the following uses are considered abuse of my work and are strictly prohibited):
    1. Military use, except explicitly authorized by the author 
    2. Law enforcement use intended to lead to or manage incarceration 
    3. Use in committing crimes
    4. Use in any application supporting the adult films industry 
    5. Use in any application supporting the alcoholic beverages, firearms, and / or tobaco industries
    6. Any use supporting the trade, marketing of, or administration of prescription drugs which are commonly abused 
    7. Use in a manner intended to identify or discriminate against anyone on any ethnic, ideological,  religious, racial, demographic, or socioeconomic / *credit status (which includes lawful credit, tenant, and and HR screening* other than screening for criminal history).
    8. Any use supporting any operation which attempts to sway public opinion, political alignment, or purchasing habits via means such as:
        1. Misleading the public to beleive that the opinions promoted by said operation are those of a different group of people (commonly referred to as astroturfing).
        2. Leading the public to beleive premises that contradict duly accepted scientific findings, implausible doctrines, or premises that are generally regarded as heretical or occult.
    9. These or anything reasonably regarded as similar to these are prohibited uses of this codebase AND ANY DERIVITIVE WORK. Litigation will result upon discovery of any such violations.
