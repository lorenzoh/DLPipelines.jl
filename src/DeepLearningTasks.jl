module DeepLearningTasks

using Colors
using DataLoaders
using DataAugmentation
using FluxTraining
using LearnBase
using StaticArrays
using Parameters

include("./task.jl")
include("./types.jl")
include("./datautils.jl")
include("./transforms.jl")

include("./tasks/imageclassification.jl")


export
    ImageClassification,
    taskdataset,
    taskdataloader,

    encode,
    encodeinput,
    encodetarget,
    decodeoutput,

    interpretinput,
    interprettarget,
    interpretx,
    interprety

end
