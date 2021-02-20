# Functionality

With the `LearningMethod` interface implemented, you can use some utilities included in DLPipelines.jl:

- [`methoddataset`](#): Transform a data container of inputs and targets to one of $x$s and $y$s.
- [`methoddataloaders`](#): Create a pair of training and validation data iterators that can be used directly in training loops. Uses [DataLoaders.jl](https://github.com/lorenzoh/DataLoaders.jl)
- [`checkmethod`](#): Check interface conformance of a `method` implementation.
- [`predict`](#), [`predictbatch`](#): Use a model to predict a target from an input (or batches of each)

    You may also use [`checkmethod_core`](#) and [`checkmethod_interpretation`](#) to check the interfaces separately.

