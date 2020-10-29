
# Core interface

First you need to **define a custom [`Method`](#)** (and possibly a `Task`).

Then implement the following functions by dispatching on the method.

## Required functions

- [`encodeinput`](#)`(method, context, input) -> x` encodes an input so that it can be fed to a model
- [`decodeŷ`](#)`(method, context, ŷ) -> target` decodes the output of a model into a target
- either
  - [`encode`](#)`(method, context, sample) -> (x, y)` encodes a sample containing input and target so they can be used for a training step
  - [`encodetarget`](#)`(method, context, target) -> y`
  encodes a target so that it can be compared to a model output using a loss function

## Optional functions

- [`shouldbatch`](#)`(method)` defines whether the model for a method should take batches of encoded inputs. The default is `true`.

## Remarks

- *What is the `context` argument?*

  We often want to apply data augmentation during training but not during validation or inference. You can dispatch on `context` to define different behavior for the different situations. See [`Context`](#) to see the available options.

- *When should I implement `encode` vs. `encodetarget`?* 

  See [`encode`](#).
