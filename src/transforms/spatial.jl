
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


function (spatial::SpatialTransforms)(context, datas::Tuple)
    items = makespatialitems(datas)
    tfm = _gettfm(spatial, context)
    return itemdata.(apply(tfm, items))
end

(spatial::SpatialTransforms)(context, data) = spatial(context, (data,)) |> only


_gettfm(spatial::SpatialTransforms, context::Training) = spatial.traintfm
_gettfm(spatial::SpatialTransforms, context::Validation) = spatial.validtfm
_gettfm(spatial::SpatialTransforms, context::Inference) = spatial.inferencetfm


makespatialitems(datas::Tuple) = makespatialitems(datas, getbounds(first(datas)))
function makespatialitems(datas::Tuple, bounds)
    return Tuple(makeitem(data, bounds) for data in datas)
end


"""
    makeitem(data, args...)

Tries to assign a `DataAugmentation.Item` from `data` based on its type.
`args` are passed to the chosen  `Item` constructor.

- `AbstractMatrix{<:Colorant}` -> `Image`
- `Vector{<:Union{Nothing, SVector}}` -> `Keypoints`
"""
makeitem(data, args...) = itemtype(data)(data, args...)
makeitem(item::Item, args...) = item


itemtype(data::AbstractMatrix{<:Colorant}) = DataAugmentation.Image
itemtype(data::Vector{<:Union{Nothing, SVector}}) = DataAugmentation.Keypoints
