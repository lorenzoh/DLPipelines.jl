
IMAGENET_MEANS = [0.485, 0.456, 0.406]
IMAGENET_STDS = [0.229, 0.224, 0.225]


"""
    SpatialTransforms(size, [augmentations])
"""
struct SpatialTransforms
    size
    augmentations
end

function (spatial::SpatialTransforms)(img::Image; augment = true)
    tfms = (augment ? RandomResizeCrop : CenterResizeCrop)(spatial.size)
    if augment
        tfms = tfms |> spatial.augmentations
    end
    return apply(tfms, DataAugmentation.Image(img)) |> itemdata
end


"""
    ImagePreprocessing([C], means, stds)

Converts an image to a color, then to a 3D-tensor and
finally normalizes the values using `means` and `stds`.

If no `means` or `stds` are given, use ImageNet statistics.
"""
@with_kw struct ImagePreprocessing
    C::Type{<:Colorant} = RGB
    means = IMAGENET_MEANS
    stds = IMAGENET_STDS
end

function (ip::ImagePreprocessing)(img::Image)
    tfms = ToEltype{ip.C}() |> SplitChannels() |> Normalize(ip.means, ip.stds)
    apply(tfms, DataAugmentation.Image(img)) |> itemdata
end
