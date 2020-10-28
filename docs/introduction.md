# Introduction

In supervised deep learning, we're usually trying to solve a problem by finding a mapping from some inputs to some targets. Let's call this a *task*. Consider for example this table:

| Task                  | Input | Target             |
| :-------------------- | :---- | :----------------- |
| Image classification  | Image | Category           |
| Semantic segmentation | Image | Category per pixel |
| Object detection      | Image | Bounding boxes     |
| Text completion       | Text  | Next character     |

There are usually multiple ways to go about solving a task. We call a *method* a concrete approach to a task that has a learnable part (the model) but also defines how

- inputs and targets are *encoded* into a form that can be fed to and output by the model
- model outputs are *decoded* to obtain target predictions; and

os an example method, consider the commmon way of approaching the task of image classification:

- images are encoded into normalized 3D-Arrays of `Float32`s and categories into one-hot vectors
- the predicted probability distributions can be decoded into a category by finding the index of the highest score; and
- the model takes in batches of encoded inputs and output batches of encoded targets

An additional complication comes from the fact that the encoding and decoding step may differ based on situation. For example, during training we often want to apply some augmentation when encoding that would be detrimental to performance during inference.

## In code

The most important type is [`Method`](#) which represents a method for a [`Task`](#). All interface functions will dispatch on `Method`. It should be a concrete `struct` containing necessary configuration.

Let's give the other concepts variable names and generic types so we can refer to them more easily:

- `task::Task`: a task mapping `I` to `T`
- `method::Method{Task}`: a method implementing a task using encodings `X` and `Y`
- `input::I`: an input
- `target::T`: a target
- `x::X`: an encoded input
- `y::Y`: an encoded target
- `ŷ::Ŷ`: a model output
- `inputs`, `targets`, `xs`, `ys`: batches of the respective data
- `model`: a function `(X) -> Ŷ`
- `lossfn`: a function `(Ŷ, Y) -> Number`

So a `Task` is an abstract type representing a mapping from some input type `I` to target type `T`. A `Method{T}` implements the task `T` by using encoded representations `X` and `Y`.

As a concrete example, consider image classification:

- `task::ImageClassificationTask`: subtype of `Task`
- `method::ImageClassification`: subtype of `Method{ImageClassificationTask}`
- `input::AbstractMatrix{2, <:Colorant}`: an image
- `target::Int` the category that the image belongs to
- `x::AbstractArray{Float32, 3}`: a normalized 3D-array with dimensions *height, width, channels*
- `y::AbstractVector{Float32}`: one-hot encoding of category
- `ŷ::AbstractVector{Float32̂}`: softmax probabilities

## Core pipelines

To give a motivation for the interface, consider the two most important pipelines in a deep learning application: training and inference.

*Note: I'm purposely neglecting batching here (for now), as it doesn't change the semantics of the data pipeline, just the practical implementation.*

During **inference**, we have an input and obtain a target prediction. Writing this with types gives us:

```text
     encode       model       decode
::I -------> ::X ------> ::Ŷ -------> ::T
```

When **training**, we first encode both input and target, including any augmentation. We then feed the encoded input to the model and compare its output with the true encoded target. 

```text
          encode            lossfn(model(X), Y)
::(I, 0) -------> ::(X, Y) --------------------> loss
```

From those two pipelines, we can extract the following transformations:

- `I -> X` encoding input
- `Ŷ -> T` decoding model output
- `(I, T) -> (X, Y)` encoding input and target

These make up the [core interface](interfaces/core.md).