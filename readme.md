#  Residual Multi Layer Perceptron build and fit engine written in python 10.0.0 using Tensorflow 2.6 functional API. 
## This is useful for building residual MLP models both as stand alone residual MLP models and as tandem [convolutional | vision transformer | etc]->Residual MLP models where the resudual MLP aspect of the model is used to augment the performance of a commonly used base model for transfer learning (e.g. EffieientNet/ResnetNet/etc.).

1. Purpose: This is a core engine for recursively building, compiling, and testing Residual Multi Layer Perceptron models. This is meant to enable neural architecture search in this non-standard and non-sequential MLP architecture. This system abstracts away most of the tedious work of building residual multi layer perceptron models and reduces it to a selection of any suitable permutations of this neural architecture constructor's hyperparameters.
2. What is a residual multi layer perceptron? It is a non-sequential multi-layer-perceptron where as with all multi layer perceptrons, there is a block of sequential Dense layers, however, unlike a standard sequential multi - layer - perceptron, there is also a residual tensor which forwards a duplicate of the input to this block of dense layers forward, bypassing the block of dense layers. At the output end of the block of dense layers, the residual tensor is concatenated with the output of the block of sequential dense layers before proceeding to the next block. This creates a memory effect and importantly, attenuates some of the deleterious effects introduced by very deep networks, such as overfitting, [exploding | valishing] gradients, and internal covariate shift. 
3. Why use a residual multi layer perceptron model?
    1. Better performance on small data sets.
    2. Residual MLPs enable greater model complexity and insanely deep neural network depth before overfitting, valishing and exploding gradients, and internal covariate shift become the model complexity limiting factors.
    3. Better precision, recall, and other performance metrics with less training data and far fewer epochs.
4. Are there any drawbacks with using a residual MLP model?
    1. They are a bit computationallly expensive on an epoch - by epoch basis. When training on smaller data sets, they do make up for this when you conider they they need far fewer epochs to reach convergence.
    2. Although they can train quickly, for the time they are training, they do need large hardware configurations to complete training successfully. For example, attempting to train a model using the pre-trained EfficientNetB7-ResidualMLP (with optimal residual MLP hyperparameters and neural network architecture) will exhaust the RAM on a machine having 45GB of RAM, unless the machine has at least 2 A4000 GPUs. This can make some jobs not financially worth the cost of training them. However, for training jobs that have failed due to small sample size / not enough data which are worth $3 to $50 per training run, this may be an ideal overall algorithm.
4. Use example:
    Under construction...
5. License: Licensed under a modified MIT license, but with the following exclusions (the following uses are considered abuse of my work and are strictly prohibited): 
    1. Military use, 
    2. Law enforcement use intended to lead to or manage incarceration, 
    3. Use in committing crimes, 
    4. Use in any application supporting the adult films industry 
    5. Use in any application supporting the alcoholic beverages, firearms, and / or tobaco industries,
    6. Any use supporting the trade, marketing of, or administration of prescription drugs which are commonly abused 
    7. Use in a manner intended to identify or discriminate against anyone on any ethnic, ideological,  religious, racial, demographic, or socioeconomic / *credit status (which includes lawful credit, tenant, and and HR screening* other than screening for criminal history).
    8. Any use supporting any operation which attempts to sway public opinion, political alignment, or purchasing habits via means such as:
        1. Misleads the public to beleive thate the opinions promoted by said operation are those of a different group of people (commonly referred to as astroturfing).
        2. Leading the public to beleive premises that contradict duly accepted scientific findings, implausible doctrines, or premises that are generally regarded as heretical or occult.
    9. These or anythg reasonablu regarded as similar to these are prohibited uses of this codebase AND ANY DERIVITIVE WORK.