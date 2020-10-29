
# See **[`SpatialTransforms`](#).**



"""
    SpatialTransforms(size, [augmentations])

Transformation that resizes images and keypoints to `size`.

In context [`Training`](#), applies `augmentations`.
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
