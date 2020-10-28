
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
function encodeinput(task, input; inference = false) end

"""
    encodetarget(task, target; augment = false, inference = false) -> y

Encode `target` into a representation that a model for `task` outputs.
"""
function encodetarget(task, target) end

"""
    encode(method, context, sample) -> (x, y)
    encode(method, context, (input, target)) -> (x, y)

Encode a `sample` containing both input and target .

If `sample` is a `Tuple` of (input, target), the default behavior is to
pass them to [`encodeinput`](#) and [`encodetarget`](#)

### Remarks

- *When should I implement `encode` vs. `encodeinput` and `encodetarget`?*

  In simple cases like image classification we can encode the inputs and
  targets separately and you should prefer `encodeinput` and `encodetarget`.
  The default implementation for `encode` when given an `(input, target)`-tuple
  is to delegate to `encodeinput` and `encodetarget`.

  In other cases like semantic segmentation, however, we want to apply stochastic augmentations to both image and segmentation mask. In that case you need to encode both at the same time using `encode`

  Another situation where `encode` is needed is when `sample` is not a tuple of `(input, target)`, for example a `Dict` that includes additional information. `encode` still needs to return an `(x, y)`-tuple, though.
"""
encode(method, context, (input, target)::Tuple) =
    (encodeinput(method, context, input), encodetarget(method, context, target))


# Decoding

"""
    decodeŷ(method, context, ŷ) -> target

Decodes a model output into a target.
"""
function decodeŷ(method, context, ŷ) end


# Inference

# TODO: refactor to respect `shouldbatch`
function predict(task, model, input; device = cpu, batch = true)
    x = encodeinput(task, input; inference = true)
    xs = device(reshape(x, size(x)..., 1))
    ŷs = cpu(model(xs))
    return decodeoutput(task, ŷs[:, :, :, 1])
end

# TODO: implement
function predictbatch end


# Interpretation

# TODO: refactor and document
function interpretinput(task, input) end
function interprettarget(task, target) end
function interpretx(task, x) end
interprety(task, y) = interprettarget(task, decodeoutput(task, y))



#= Ideas

- interface for input/output size


=#
