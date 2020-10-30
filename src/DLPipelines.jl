# # DLPipelines.jl
#
# *DLPipelines.jl* is an interface for defining deep learning data
# pipelines. This includes every data transformation beside the model
# itself: preprocessing and augmenting the data before feeding it
# into the model, and decoding the model's outputs to targets.
module DLPipelines #src

using Colors
using DataLoaders
using DataAugmentation
using Flux
using FluxTraining
using MLDataUtils
using LearnBase
using StaticArrays
using Parameters

# [`task.jl`](./task.jl) defines the [task and method interface](../docs/interfaces/core.md).
include("./task.jl") #src
export Task, Method, Context, Training, Validation, Inference, encode, encodeinput, encodetarget
# [`api.jl`](/api.jl) defines the user-facing functions.
export dataiter, predict
#
# The transforms are defined under `transforms/`:
# - [`transforms/spatial.jl`](transforms/spatial.jl)
# - [`transforms/imagepreprocessing.jl`](transforms/imagepreprocessing.jl)
include("./transforms/spatial.jl")  #src
include("./transforms/imagepreprocessing.jl")  #src
export SpatialTransforms, ImagePreprocessing
#
# The [`Method`](#) implementations live under `methods/`:
# - [`methods/imageclassification.jl`](./methods/imageclassification.jl)
include("./methods/imageclassification.jl")  #src
export ImageClassification


end  # module #src
