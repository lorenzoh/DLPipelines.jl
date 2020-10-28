# DLPipelines.jl

*DLPipelines.jl* is an interface for defining deep learning data pipelines. This includes every data transformation before and after the model. That means loading, preprocessing and augmenting the data before feeding it into the model, but also decoding the model's outputs to targets.

The package was born from the realization that the data pipeline plays a large role in many deep learning projects. It abstracts the pipeline into steps that lend themselves to building training, inference and other pipelines.

Let's get some terminology out of the way.

A (supervised) **task** is a learnable mapping from **input**s to **target**s. The learnable part of the mapping is represented by a **model**.


| Task                  | Input | Target             |
| :-------------------- | :---- | :----------------- |
| Image classification  | Image | Category           |
| Semantic segmentation | Image | Category per pixel |
| Object detection      | Image | Bounding boxes     |

A **method** is a concrete way to solve a task and defines how

- inputs and targets are *encoded* into a form that can be fed to and output by the model
- model outputs are *decoded* to obtain target predictions; and
- the general model structure, like input and output sizes

As an example method, consider the commmon way of approaching the task of **image classification**:

- images are encoded into normalized 3D-Arrays of `Float32`s and categories into one-hot vectors
- the predicted probability distributions can be decoded into a category by finding the index of the highest score; and
- the model takes in batches of encoded inputs and output batches of encoded targets

An additional complication comes from the fact that the encoding and decoding step may differ based on situation. For example, during training we often want to apply some augmentation when encoding that would be detrimental to performance during inference.

Let's look at the interface provided by *DLPipelines.jl* to represent these abstractions.
