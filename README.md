## NNWrapper is just a Neural Network Wrapper
NNWrapper is designed to be a wrapper for a variaty of Neural Network frameworks, such as chainer, pytorch, tensorflow and so on.

It hopefully would provide wrapper framework with transparency, flexibility and expressiviness, on both high-level neural network design(including parameters and layers registration, backpropogation and optimization) and low-level data operation(wrapper of tensor which supports a wide-range of operations)

It directly makes use of autograd mechanism and basic tensor operation from original Neural Network framework, but provide a general syntax, allowing one common model be run on different backends.


In model design, we prefer the idea of define-by-run of chainer and pytorch, and use fun:register-weights and fun:add-nodes functions to add subgraph and weights dynamically to the models during run-time, while for frameworks without define-by-run feature like tensorflow, we use fun:register-weights and fun:add-nodes functions to add subgraph and weights to the model before run and automatically call feed operation once the model object is called, just mimicing the behaviour of chainer and pytorch.

In low-level data structure design, we prefer the syntax used by numpy, including index, sliding and broadcasting. Our future work includes designing tensor wrapper to be more numpy like, so that common numpy syntax is supported automatically.


### Currently support:
* chainer wrapper for basic model design, with layers including Linear and Conv2d
* pytorch wrapper for basic model design, with layers including Linear and Conv2d

### Next step development:
* tensorflow wrapper for basic model design, with layers including Linear and Conv2d
* more layers support for chainer and pytorch wrapper
* enabling pytorch torch tensor wrapper performs more like cupy ndarrays



