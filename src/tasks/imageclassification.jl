
@with_kw mutable struct ImageClassification <: Task{Image, ImageTensor, OneHotVector, Class}
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
        image::Image;
        inference = false,
        augment = false)
    return task.spatialtransforms(image; augment)[1] |> task.imagepreprocessing
end


function encodeoutput(
        task::ImageClassification,
        class::Class;
        inference = false,
        augment = false)
    return onehotencode(class, task.nclasses)
end


encodetarget(task::ImageClassification, class::Class; kwargs...) =
    DataAugmentation.onehot(class, 1:task.nclasses)

decodeoutput(task::ImageClassification, ŷ) = argmax(ŷ)

interpretinput(task::ImageClassification, image::Image) = image

function interpretx(task::ImageClassification, x)
    return invert(task.imagepreprocessing, x)
end

function interprettarget(task::ImageClassification, class::Class)
    return "Class $class"
end
