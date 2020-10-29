
abstract type ImageClassificationTask <: Task end


"""
    ImageClassification(nclasses[; sz, augmentations, ...]) <: Method{ImageClassificationTask}

A [`Method`](#) for multi-class image classification using softmax probabilities.

### Types

- `input::AbstractMatrix{2, <:Colorant}`: an image
- `target::Int` the category that the image belongs to
- `x::AbstractArray{Float32, 3}`: a normalized 3D-array with dimensions *height, width, channels*
- `y::AbstractVector{Float32}`: one-hot encoding of category

### Model

- input size: `(sz..., ch, batch)` where `ch` depends on color type `C`.
- output size: `(nclasses, batch)`
"""
@with_kw mutable struct ImageClassification <: Method{ImageClassificationTask}
    nclasses::Int
    spatialtransforms::SpatialTransforms = SpatialTransforms()
    imagepreprocessing::ImagePreprocessing = ImagePreprocessing()
end

function ImageClassification(
        nclasses::Int;
        sz = (224, 224),
        augmentations = Identity(),
        means = IMAGENET_MEANS,
        stds = IMAGENET_STDS,
        C = RGB,
    )
    spatialtransforms = SpatialTransforms(sz, augmentations = augmentations)
    imagepreprocessing = ImagePreprocessing(C, means, stds)
    ImageClassification(nclasses, spatialtransforms, imagepreprocessing)
end



function encodeinput(
        task::ImageClassification,
        context,
        image)
    return task.spatialtransforms(context, image)[1] |> task.imagepreprocessing
end


function encodeoutput(
        task::ImageClassification,
        class;
        inference = false,
        augment = false)
    return onehotencode(class, task.nclasses)
end


encodetarget(task::ImageClassification, class; kwargs...) =
    DataAugmentation.onehot(class, 1:task.nclasses)

decodeoutput(task::ImageClassification, ŷ) = argmax(ŷ)

interpretinput(task::ImageClassification, image) = image

function interpretx(task::ImageClassification, x)
    return invert(task.imagepreprocessing, x)
end

function interprettarget(task::ImageClassification, class)
    return "Class $class"
end
