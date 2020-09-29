
IMAGENET_MEANS = [0.485, 0.456, 0.406]
IMAGENET_STDS = [0.229, 0.224, 0.225]


"""
    SpatialTransforms(size, [augmentations])
"""
@with_kw struct SpatialTransforms
    traintfm
    validtfm
    inferencetfm
end

function SpatialTransforms(
        size;
        augmentations = Identity(),
        inferencefactor = 1)

    return SpatialTransforms(
        RandomResizeCrop(size) |> augmentations,
        CenterResizeCrop(size),
        ResizeDivisible(size, divisible = inferencefactor),
    )

end

function (spatial::SpatialTransforms)(image, others...; augment = true, inference = false)
    if inference
        tfm = spatial.inferencetfm
    elseif augment
        tfm = spatial.traintfm
    else
        tfm = spatial.validtfm
    end
    items = makeitems(image, others...)
    return itemdata.(apply(tfm, items))
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


function invert(ip::ImagePreprocessing, x)
    return DataAugmentation.tensortoimage(DataAugmentation.denormalize(x, ip.means, ip.stds))
end


# Utils

function makeitems(image::Image, others...)
    bounds = DataAugmentation.makebounds(image)
    return (
        DataAugmentation.Image(image, bounds),
        Tuple(makeitem(other, bounds) for other in others)...
    )
end


makeitem(img::Image, bounds) = DataAugmentation.Image(img, bounds)
makeitem(x::Vector{<:SVector}, bounds) =
    DataAugmentation.Keypoints(x, bounds)
makeitem(x::Vector{<:Union{Nothing, SVector}}, bounds) =
    DataAugmentation.Keypoints(x, bounds)
makeitem(xs::Vector, bounds) = [makeitem(x, bounds) for x in xs]
