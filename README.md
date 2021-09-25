# pruning-nn

Code for my bachelor thesis about Pruning of Fully-connected feed-forward Neural Networks (FFN) using pyTorch

----------

**Topic**: A comparative evaluation of pruning techniques for Artificial Neural Networks

Submitted: April 2019

Institute: Chair of Data Science, University of Passau

[Download](https://github.com/pauleon2/pruning-nn/raw/master/thesis.pdf) the thesis as full text

----------

**Abstract**: Pruning of neural networks is one of several techniques that can be used to create sparse neural networks and to prevent neural networks from overfitting in general. In this thesis, different types of pruning techniques are compared with each other. Therefore, pruning techniques in general are divided into pruning strategies, which determine how many elements are deleted in each pruning step, and pruning methods that rank the weights based on their importance for the network. The different methods and strategies have been applied to some fully-connected networks that have previously been trained to a minimum on the MNIST dataset. Our experiments have shown that iterative pruning is especially useful if we want to create highly compressed networks, while fixed number pruning is able to create the highest performing ones. In contrast, single pruning is not able to create as good networks as the other two approaches do. From the different magnitude based methods that have been applied, the magnitude class blinded approach clearly outperforms the other ones. The Optimal Brain Damage method, which is based on the second order derivative of the cost function, performs nearly the same as the magnitude class blinded method, but with an increasing number of pruning steps the accuracy drops and the additional needed computations don’t pay of in our experiments. Furthermore, pruning in general is able to reduce the overfitting in the examined network dramatically and is even able to outperform other regularization techniques although more computational steps are required in the pruning approach compared to the other ones.
