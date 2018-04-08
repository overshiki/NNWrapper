### NNWrapper is just a Neural Network Wrapper
NNWrapper is designed to be a wrapper for a variaty of Neural Network framework, such as chainer, pytorch, tensorflow and so on.

It provides wrapper framework with transparency, flexibility and expressiviness, on both high-level neural network design(including parameters and layers registration, backpropogation and optimization) and low-level data structure(wrapper of tensor which support a wide-range of operations)

It makes use of autograd mechanism and basic tensor operation from original Neural Network framework, but provide a general syntax, allowing one common model be run on different backends.

Unlike keras only for tensorflow and providing a variaty of builting model bricks, NNWrapper do nothing but wrap program syntax of different framework into a common one.

Unlike tonnx provide a common code for different framework to transfer their model to, NNWrapper encourage directly designing on top of it, with the resulting model be able run universarily.