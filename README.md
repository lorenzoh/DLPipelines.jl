# DLPipelines

*DLPipelines.jl* is an interface for defining deep learning data pipelines. This includes every data transformation beside the model itself: preprocessing and augmenting the data before feeding it into the model, and decoding the model's outputs to targets.

With the interface defined, it's dead simple to create data iterators for training and inference pipelines. See [image classification](docs/methods/imageclassification.md) as a motivating example.

The package was born from the realization that the data pipeline plays a large role in many deep learning projects. It abstracts the pipeline into steps that lend themselves to building training, inference and other pipelines.

[Read on here](docs/introduction.md) to find out more about the interface.

***DLPipelines.jl* is currently in a pre-implementation phase where the focus is on exploring suitable interfaces.**