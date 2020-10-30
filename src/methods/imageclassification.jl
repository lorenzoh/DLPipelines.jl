
abstract type ImageClassificationTask <: Task end

"""
    ImageClassification(categories; [sz, augmentations, ...]) <: Method{ImageClassificationTask}
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
    categories::AbstractVector
    spatialtransforms::SpatialTransforms
    imagepreprocessing::ImagePreprocessing
end

Base.show(io::IO, method::ImageClassification) = print(
    io, "ImageClassification() with $(length(method.categories)) categories.")

function ImageClassification(
        categories::AbstractVector;
        sz = (224, 224),
        augmentations = Identity(),
        means = IMAGENET_MEANS,
        stds = IMAGENET_STDS,
        C = RGB,
    )
    spatialtransforms = SpatialTransforms(sz, augmentations = augmentations)
    imagepreprocessing = ImagePreprocessing(C, means, stds)
    ImageClassification(categories, spatialtransforms, imagepreprocessing)
end

ImageClassification(n::Int, args...; kwargs...) = ImageClassification(1:n, args...; kwargs...)


# core interface implementation

function encodeinput(
        method::ImageClassification,
        context,
        image)
    imagecropped = method.spatialtransforms(context, image)
    x = method.imagepreprocessing(imagecropped)
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

interpretinput(task::ImageClassification, image) = image

function interpretx(task::ImageClassification, x)
    return invert(task.imagepreprocessing, x)
end

function interprettarget(task::ImageClassification, class)
    return "Class $class"
end
