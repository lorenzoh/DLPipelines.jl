
"""
    Task{I, O, X, Y}

A `Task`

A `Task` represents a mapping from high-level
types `I` to `O`. The learnable part of the mapping
is from types `X` to `Y`.

```
Inference

     encode       model       decode
::I -------> ::X ------> ::Y -------> ::O


Training step

          encode            lossfn(model(X), Y)
::(I, 0) -------> ::(X, Y) --------------------> loss
```

To give an example, image classification is task that
maps images to classes (represented as a matrix of pixels and
an integer). As inputs and outputs to the learning algorithm,
a more optimization-friendly form is used: the image is
converted to a normalized 3D array with dimensions (h, w, ch)
and the class is converted to a one-hot encoded vector.

So, for image classification, we have:
```
I = AbstractMatrix{Colorant}  # image
O = Int
X = AbstractArray{AbstractFloat, 3}  # (w, h, ch)
Y = AbstractVector{AbstractFloat}  # one-hot encoded class vector

ImageClassification <: Task{I, O, X, Y}
```

In this representation, the pieces to build a training
and inference pipeline can be cleanly separated.

The first is encoding, i.e. mapping `I` to `X` and `O` to `Y`.

The second is decoding, i.e. mapping Y -> O.

Both of these can highly configurable. For encoding an input image,
we may want to resizing and some stochastic data augmentation, but
not during inference.

Any configuration/hyperparameters should be stored in the `Task`
struct which is passed to every method.
"""
abstract type Task{I,O,X,Y} end


# Encoding

"""
    encodeinput(task, input; augment = false, inference = false) -> x

Encode `input` into a representation that a model for `task`
takes as input.
"""
function encodeinput(task, input) end

"""
    encodetarget(task, target; augment = false, inference = false) -> y

Encode `target` into a representation that a model for `task` outputs.
"""
function encodetarget(task, target) end

"""
    encode(task, sample; augment = false, inference = false) -> (x, y)

Encode a `sample` containing both input and target into representations
that a model for `task` takes in and outputs, respectively.

If `sample` is a `Tuple` of (input, target), the default behavior is to
pass them to `encodeinput` and `encodetarget`
"""
encode(task, (input, target)::Tuple; kwargs...) =
    (encodeinput(task, input; kwargs...), encodetarget(task, target; kwargs...))


# Decoding

"""
    decodeoutput(task, y) -> target
"""
function decodeoutput(task, y) end


# Inference

function predict(task, model, input; device = cpu, batch = true)
    x = encodeinput(task, input; inference = true)
    xs = device(reshape(x, size(x)..., 1))
    ŷs = model(xs)
    return decodeoutput(task, )
    ŷ -> decodeoutput(task, ŷ)

end


# Interpretation

function interpretinput(task, input) end
function interprettarget(task, target) end
function interpretx(task, x) end
interprety(task, y) = interprettarget(task, decodeoutput(task, y))



#= Ideas

- interface for input/output size


=#
