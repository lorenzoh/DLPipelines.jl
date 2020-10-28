# Usage

From a [`Method`](#) implementation, we can derive some useful functionality.

- [`predict`](#)`(method, model, input)` predicts a target from an input
- [`dataiter`](#)`(method, data, batchsize)` creates a data iterator from container `data` using [*DataLoaders.jl*](https://lorenzoh.github.io/DataLoaders.jl/dev) that can be used directly in a training loop