module DeepLearningTasks

using Colors
using DataLoaders
using DataAugmentation
using FluxTraining
using LearnBase
using Parameters

include("./task.jl")
include("./types.jl")
include("./datautils.jl")
include("./transforms.jl")

include("./tasks/imageclassification.jl")


export
    ImageClassification,
    taskdataset,
    taskdataloader

end
