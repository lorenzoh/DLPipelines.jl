# See [introduction](../docs/introduction.md) and [core interface](../docs/interfaces/core.md).

# # Types
#
# We start off by defining the abstract types that will be used for dispatch.


# ### [`LearningMethod`](#)
"""
    abstract type LearningMethod

Represents a concrete approach for solving a learning task.

See [core interface](../docs/interfaces/core.md) for more on
how to implement custom `LearningMethod`s
"""
abstract type LearningMethod end


# ### [`Context`](#) and concrete types
"""
    abstract type Context

Represents a context in which a data transformation
is made. This allows using dispatching for varying behavior,
for example, to apply augmentations only during training or
use non-destructive cropping during inference.

Available contexts are [`Training`](#), [`Validation`](#) and
[`Inference`](#).
"""
abstract type Context end

struct Training <: Context end
struct Validation <: Context end
struct Inference <: Context end

# # Core interface
#
# Next the core interface is defined:
# - [`encode`](#)
# - [`encodeinput`](#)
# - [`encodetarget`](#)
# - [`decodeŷ`](#)
# - [`decodey`](#)
# - [`shouldbatch`](#)
"""
    encodeinput(method, context, input) -> x

Encode `input` into a representation that a model for `method`
takes as input.

See also [`LearningMethod`](#), [`encode`](#), and [`encodetarget`](#).
"""
function encodeinput end

"""
    encodetarget(task, target; augment = false, inference = false) -> y

Encode `target` into a representation that a model for `task` outputs.
"""
function encodetarget end

"""
    encode(method, context, sample) -> (x, y)
    encode(method, context, (input, target)) -> (x, y)

Encode a `sample` containing both input and target.

If `sample` is a `Tuple` of (input, target), the default behavior is to
pass them to [`encodeinput`](#) and [`encodetarget`](#).

### Remarks

- *When should I implement `encode` vs. `encodeinput` and `encodetarget`?*

  In simple cases like image classification we can encode the inputs and
  targets separately and you should prefer `encodeinput` and `encodetarget`.
  The default implementation for `encode` when given an `(input, target)`-tuple
  is to delegate to `encodeinput` and `encodetarget`.

  In other cases like semantic segmentation, however, we want to apply
  stochastic augmentations to both image and segmentation mask. In that
  case you need to encode both at the same time using `encode`.

  Another situation where `encode` is needed is when `sample` is not a
  tuple of `(input, target)`, for example a `Dict` that includes additional
  information. `encode` still needs to return an `(x, y)`-tuple, though.
"""
function encode end


"""
    decodeŷ(method, context, ŷ) -> target

Decodes a model output into a target.
"""
function decodeŷ end


"""
    decodey(method, context, y) -> target

Decodes an encoded target back into a target.

Defaults to using [`decodeŷ`](#)
"""
decodey(method, context, y) = decodeŷ(method, context, y)


"""
    shouldbatch(method) = true

Whether models for `method` take in batches of inputs. Default
is `true`.
"""
shouldbatch(method) = true
