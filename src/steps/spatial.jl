
# See **[`SpatialTransforms`](#).**



"""
    SpatialTransforms(size, [augmentations])

Pipeline step that resizes images and keypoints to `size`.

In context [`Training`](#), applies `augmentations`.
"""
@with_kw struct SpatialTransforms <: PipelineStep
    traintfm
    validtfm
    inferencetfm
end

function SpatialTransforms(
        size;
        augmentations = Identity(),
        inferencefactor = 1,
        buffered = true)
    tfms = (
        RandomResizeCrop(size) |> augmentations,
        CenterResizeCrop(size),
        ResizeDivisible(size, divisible = inferencefactor),
    )
    if buffered
        tfms = InplaceThreadsafe.(tfms)
    end
    return SpatialTransforms(tfms...)
end


function apply(spatial::SpatialTransforms, context, datas::Tuple)
    items = makespatialitems(datas)
    tfm = _gettfm(spatial, context)
    tdatas = itemdata.(DataAugmentation.apply(tfm, items))
    return copy.(tdatas)
end

function apply!(bufs, spatial::SpatialTransforms, context, datas::Tuple)
    items = makespatialitems(datas)
    tfm = _gettfm(spatial, context)
    t = tfm.inplaces[1]
    #@time DataAugmentation.apply(tfm.inplaces[1], items)
    tdatas = itemdata.(DataAugmentation.apply(tfm, items))
    copy!.(bufs, tdatas)
    return bufs
end

apply(spatial::SpatialTransforms, context, data) = apply(spatial, context, (data,)) |> only
apply!(buf, spatial::SpatialTransforms, context, data) = apply!((buf,), spatial, context, (data,)) |> only


## Utils

_gettfm(spatial::SpatialTransforms, context::Training) = spatial.traintfm
_gettfm(spatial::SpatialTransforms, context::Validation) = spatial.validtfm
_gettfm(spatial::SpatialTransforms, context::Inference) = spatial.inferencetfm


makespatialitems(items::NTuple{N, Item}) where N = items
makespatialitems(datas::Tuple) = makespatialitems(datas, size(first(datas)))
function makespatialitems(datas::Tuple, sz)
    return Tuple(makeitem(data, sz) for data in datas)
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
makeitem(datas::Tuple, args...) = Tuple(makeitem(data, args...) for data in datas)


itemtype(data::AbstractMatrix{<:Colorant}) = DataAugmentation.Image
itemtype(data::Vector{<:Union{Nothing, SVector}}) = DataAugmentation.Keypoints
