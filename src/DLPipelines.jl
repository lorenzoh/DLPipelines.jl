# # DLPipelines.jl
#
# *DLPipelines.jl* is an interface for defining deep learning data
# pipelines. This includes every data transformation beside the model
# itself: preprocessing and augmenting the data before feeding it
# into the model, and decoding the model's outputs to targets.

module DLPipelines


using DataLoaders
using LearnBase
using MLDataPattern
using Parameters
using Test


include("./task.jl")
include("./interfaces.jl")
include("./datautils.jl")
include("./inference.jl")
include("./check.jl")


export
    # Core types
    LearningTask, LearningMethod, Context, Training, Validation, Inference,

    # Core interface
    encode, encodeinput, encodetarget, decodeŷ,

    # Buffered interface
    encode!, encodeinput!, encodetarget!, decodeŷ!,

    # Interpretation interface
    interpretsample, interpretinput, interprettarget,
    interpretx, interprety, interpretŷ

    # Derived functionality
    methoddataset, predict, predictbatch,

    # Training interface
    methodmodel, methodlossfn

end  # module
