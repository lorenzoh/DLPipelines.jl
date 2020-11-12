# # Using with FluxTraining.jl
#
# *DLPipelines.jl* works great together with
# [*FluxTraining.jl*](https://github.com/lorenzoh/FluxTraining.jl).
#
# We'll train an image classifier to show how.
#
# If you want to follow along, you'll have to install some other libraries

using Pkg
Pkg.add("DataLoaders")
Pkg.add("Flux")
Pkg.add("LearnBase")
Pkg.add("MLDataPattern")
Pkg.add(url="https://github.com/lorenzoh/DLDatasets.jl")
Pkg.add(url="https://github.com/lorenzoh/FluxTraining.jl")
Pkg.add(url="https://github.com/lorenzoh/ModelUtils.jl")
Pkg.add(url="https://github.com/lorenzoh/FluxModels.jl")

# and import the following:
# {cell=main}

using DataLoaders: DataLoader
using DLPipelines
using DLDatasets: ImageNette, loaddataset, metadata
using Flux: ADAM, logitcrossentropy
using FluxModels: xresnet18
using FluxTraining: Learner, fit!
using LearnBase: getobs
using MLDataPattern: splitobs

# Recall the 4 things needed to start training with *FluxTraining.jl*:
#
# - a model to train,
# - a loss function,
# - an optimizer; and
# - training and validation data iterators
#
# For the first three we use:
#
# {cell=main, result=false}

lossfn = logitcrossentropy
optimizer = ADAM()
model = xresnet18()

# The last part is where *DLPipelines.jl* comes in.
#
# As in the [`ImageClassification` overview](../methods/imageclassification.jl),
# we first need to load an image classification dataset:
#
# {cell=main, result = false}

dataset = loaddataset(ImageNette, "v2_160px")
categories = metadata(ImageNette).labels

# Now we construct an [`ImageClassification`](#) object with
# the proper configuration:
#
# {cell=main}

method = ImageClassification(categories, sz = (128, 128))

# After splitting the dataset into training and validation split, we can use
# [`MethodDataset`](#) to get data containers that will encode the
# observations.
#
# {cell=main}

traindataset_, validdataset_ = splitobs(dataset, 0.8)
traindataset = MethodDataset(traindataset_, method, Training())
validdataset = MethodDataset(validdataset_, method, Validation())

# By passing either [`Training`](#) or [`Validation`](#) as [`Context`](#)
# to the datasets, the correct transformations will be automatically applied.
# For example, a random crop will be used for the `traindataset`, while
# images in `validdataset` will always be cropped from the center.
#
# Loading an observation from either will now correctly return a normalized
# 3D-array with the image data and a one-hot encoded category:
#
# {cell=main}

x, y = getobs(traindataset, 1)
summary.((x, y))

# To finally get an iterator over batches, we can use `DataLoader` from
# [DataLoaders.jl](https://lorenzoh.github.io/DataLoaders.jl/dev/README.html).
# It will also make sure to prefetch the data on background threads so that
# the training loop isn't slowed down waiting for the next batch of data.
#
# {cell=main, result=false}

batchsize = 16
traindl = DataLoader(traindataset, batchsize)
validdl = DataLoader(validdataset, batchsize)

# Now we can create a `Learner`:
#
# {cell=main}

learner = Learner(model, (traindl, validdl), optimizer, lossfn)

# You can pass any number of
# [callbacks](https://lorenzoh.github.io/FluxTraining.jl/dev/docs/callbacks/reference.html)
# to `Learner`, for example `ToGPU()` to utilize your GPU when training.
#
# And that's it, you can start training with a call to `fit!`:

fit!(learner, 10)

# To find out more about what you can do with *FluxTraining.jl*, see
# [its documentation](https://lorenzoh.github.io/FluxTraining.jl/dev/README.html).
