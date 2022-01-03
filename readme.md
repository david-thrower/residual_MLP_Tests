#  Residual Multi Layer Perceptron build and fit engine written in python 10.0.0 using Tensorflow 2.6 functional API. 
## This is useful for building residual MLP models both as stand alone models and as tandem [convolutional | vision transformer | etc]<->Residual MLP models where the resudual MLP aspect of the model is used to augment the performance of a transfer learning model.

1. Purpose: This is a core engine for recursively building, compiling, and testing Residual Multi Layer Perceptron models. This is meant to enable neural architecture search in this non-standard and non-sequential MLP architecture. This system abstracts away most of the tedious work of building residual multi layer perceptron models and reduces it to a selection of any suitable permutations of this neural architecture constructor's hyperparameters.
2. What is a residual multi layer perceptron? It is a non-sequential multi-layer-perceptron where as with all multi layer perceptrons, there is a block of sequential Dense layers, however, unlike a standard sequential multi - layer - perceptron, ther is a residual tensor which sends a duplicate of the input to this block of dense layers forward, bypassing the block of dense layers. At the output end of the block of dense layers, the residual tensor is concatenated with the output of the block of sequential dense layers before proceeding to the next block. This creates a memory effect and attenuates some of the deleterious effects introduced by very deep networks such as overfitting, [exploding | valishing] gradients, and internal covariate shift. 
3. Why use a residual multi layer perceptron?
    1. Better performance on small data sets.
    2. Residual MLPs enable greater model complexity and insanely deep neural network depth before overfitting, valishing and exploding gradients, and internal covariate shift runs you into a dead wall.
    3. Better precision, recall, and other performance metrics with less training data and far fewer epochs.
4. Are there any drawbacks with using a residual MLP model?
    1. They are a bit computationallly expensive.    
4. Use example:
    Under construction...