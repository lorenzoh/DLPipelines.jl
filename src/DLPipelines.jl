# # DLPipelines.jl
#
# *DLPipelines.jl* is an interface for defining deep learning data
# pipelines. This includes every data transformation beside the model
# itself: preprocessing and augmenting the data before feeding it
# into the model, and decoding the model's outputs to targets.

module DLPipelines

using Colors
using DataLoaders
using DataLoaders: obsslices
using DataAugmentation
using DataAugmentation: BufferedThreadsafe, getbounds, makebounds
using FixedPointNumbers
using Flux
using MLDataPattern
using MLDataUtils
using MosaicViews
using LearnBase
using Parameters
using StaticArrays
using Test


include("./task.jl")
include("./interfaces.jl")
include("./datautils.jl")
include("./inference.jl")
include("./check.jl")
include("./steps/utils.jl")
include("./steps/step.jl")
include("./steps/spatial.jl")
include("./steps/imagepreprocessing.jl")
include("./methods/imageclassification.jl")


export
    # Core types
    Task, Method, Context, Training, Validation, Inference,

    # Core interface
    encode, encodeinput, encodetarget,

    # Interpretation interface
    interpretsample, interpretinput, interprettarget,
    interpretx, interprety, interpretŷ

    # Derived functionality
    methoddataset, methoddataloaders, predict, predictbatch,

    # Training interface
    methodmodel, methodlossfn,

    # Pipeline steps
    SpatialTransforms, ImagePreprocessing,

    # Methods
    ImageClassification


end  # module
