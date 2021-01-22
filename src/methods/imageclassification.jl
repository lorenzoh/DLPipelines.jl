
abstract type ImageClassificationTask <: Task end

"""
    ImageClassification(categories, sz[; augmentations, ...]) <: Method{ImageClassificationTask}
    ImageClassification(n, ...)

A [`Method`](#) for multi-class image classification using softmax probabilities.

`categories` is a vector of the category labels. Alternatively, you can pass an integer.
Images are resized to `sz`.

During training, a random crop is used and `augmentations`, a `DataAugmentation.Transform`
are applied.

### Types

- `input::AbstractMatrix{2, <:Colorant}`: an image
- `target::Int` the category that the image belongs to
- `x::AbstractArray{Float32, 3}`: a normalized 3D-array with dimensions *height, width, channels*
- `y::AbstractVector{Float32}`: one-hot encoding of category

### Model

- input size: `(sz..., ch, batch)` where `ch` depends on color type `C`.
- output size: `(nclasses, batch)`
"""
mutable struct ImageClassification <: Method{ImageClassificationTask}
    sz::Tuple{Int, Int}
    categories::AbstractVector
    spatialtransforms::SpatialTransforms
    imagepreprocessing::ImagePreprocessing
end

Base.show(io::IO, method::ImageClassification) = print(
    io, "ImageClassification() with $(length(method.categories)) categories")

function ImageClassification(
        categories::AbstractVector,
        sz = (224, 224);
        augmentations = Identity(),
        means = IMAGENET_MEANS,
        stds = IMAGENET_STDS,
        C = RGB{N0f8},
        T = Float32
    )
    spatialtransforms = SpatialTransforms(sz, augmentations = augmentations)
    imagepreprocessing = ImagePreprocessing(means, stds; C = C, T = T)
    ImageClassification(sz, categories, spatialtransforms, imagepreprocessing)
end

ImageClassification(n::Int, args...; kwargs...) = ImageClassification(1:n, args...; kwargs...)


# Core interface implementation

function encodeinput(
        method::ImageClassification,
        context,
        image)
    imagecropped = apply(method.spatialtransforms, context, image)
    x = apply(method.imagepreprocessing, context, imagecropped)
    return x
end


function encodetarget(
        method::ImageClassification,
        context,
        category)
    idx = findfirst(isequal(category), method.categories)
    return DataAugmentation.onehot(idx, length(method.categories))
end

decodeŷ(method::ImageClassification, context, ŷ) = method.categories[argmax(ŷ)]

# Interpetration interface

interpretinput(task::ImageClassification, image) = image

function interpretx(task::ImageClassification, x)
    return invert(task.imagepreprocessing, x)
end


function interprettarget(task::ImageClassification, class)
    return "Class $class"
end


# Training interface

function methodmodel(method::ImageClassification, backbone)
    h, w, ch, b = Flux.outdims(backbone, (method.sz..., 3, 1))
    return Chain(
        backbone,
        Chain(
            AdaptiveMeanPool((1,1)),
            flatten,
            Dense(ch, length(method.categories)),
        )
    )
end

# Testing interface

function mockinput(method)
    inputsz = rand.(UnitRange.(method.sz, method.sz .* 2))
    return rand(RGB{N0f8}, inputsz)
end


function mocktarget(method)
    rand(1:length(method.categories))
end


function mockmodel(method)
    return xs -> rand(Float32, length(method.categories), size(xs)[end])
end
