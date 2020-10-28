# DLPipelines

*DLPipelines.jl* is an interface for defining deep learning data pipelines. This includes every data transformation beside the model itself: preprocessing and augmenting the data before feeding it into the model, and decoding the model's outputs to targets.

The package was born from the realization that the data pipeline plays a large role in many deep learning projects. It abstracts the pipeline into steps that lend themselves to building training, inference and other pipelines.

**It is currently in a pre-implementation phase where the focus is on exploring suitable interfaces.**