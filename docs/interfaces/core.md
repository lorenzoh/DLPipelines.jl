
# Core interface

First you need to **define a custom [`Method`](#)** (and possibly a `Task`).

Then implement the following functions by dispatching on the method.

## Required functions

- [`encodeinput`](#)`(method, context, input) -> x`
- [`decodeŷ`](#)`(method, context, ŷ) -> target`
- either
  - [`encode`](#)`(method, context, sample) -> (x, y)` or
  - [`encodetarget`](#)`(method, context, target) -> y`

Remarks:

- *What is the `context` argument?*

  We often want to apply data augmentation during training but not during validation or inference. You can dispatch on `context` to define different behavior for the different situations. See [`PipelineContext`](#) to see the available options.

- *When should I implement `encode` vs. `encodetarget`?* 

  See [`encode`](#).
