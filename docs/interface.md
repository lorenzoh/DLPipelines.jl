# Methods interface

The Method interface specifies building blocks for constructing data pipelines. If you haven't already, see [overview](./overview.md) for terminology.

The interface is built around the abstract types [`Task`](#) and [`Method{Task}`](#).

Let's give the concepts introduced variable names and types so we can refer to them more easily:

- `task::Task`: a task
- `method::Method{Task}`: a method implementing a task
- `input::I`: an input
- `target::T`: a target
- `x::X`: an encoded input
- `y::Y`: an encoded target
- `ŷ::Ŷ`: a model output
- `inputs`, `targets`, `xs`, `ys`: batches of the respective data
- `model`: a function `(X) -> Ŷ`
- `lossfn`: a function `(Ŷ, Y) -> Number`

So a `Task` is an abstract type representing a mapping from some input type `I` to target type `T`. A `Method{T}` implements the task `T` by using encoded representations `X` and `Y`.

Let's make this more concrete by filling in the types for image classification:

- `task::ImageClassificationTask` (subtype of `Task`)
- `method::ImageClassification` (subtype of `Method{ImageClassificationTask}`)
- `input::AbstractMatrix{2, <:Colorant}` (an image)
- `target::Int` (the category that the image belongs to)
- `x::AbstractArray{Float32, 3}`: a normalized 3D-array with dimensions *height, width, channels*
- `y::AbstractVector{Float32}`: one-hot encoding of category
- `ŷ::AbstractVector{Float32̂}`: softmax probabilities


## Pipelines

Now we can break down the steps of the most common pipelines in deep learning.

**Inference** is simple. We have an input and obtain a target prediction. Writing this with types gives us:

```text
     encode       model       decode
::I -------> ::X ------> ::Ŷ -------> ::T
```

**Training** is a bit more complicated: we have a pair of input and target and want to find out how to update the model parameters such that the model better predicts the target from the input.

First, we encode both input and target. This can also include augmentation. 



