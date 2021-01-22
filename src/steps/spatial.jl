
# See **[`SpatialTransforms`](#).**



"""
    SpatialTransforms(size, [augmentations])

Pipeline step that resizes images and keypoints to `size`.

In context [`Training`](#), applies `augmentations`.
"""
@with_kw_noshow struct SpatialTransforms <: PipelineStep
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
        augmentations |> RandomResizeCrop(size),
        CenterResizeCrop(size),
        ResizePadDivisible(size, inferencefactor),
    )

    if buffered
        # FIXME: Using inplace spatial transforms leads to
        # out-of-bounds crops
        #tfms = BufferedThreadsafe.(tfms)
    end

    return SpatialTransforms(tfms...)
end


function Base.show(io::IO, spatial::SpatialTransforms)
    outsize = _parenttfm(spatial.validtfm).transforms[1].crop.size
    print(io, "SpatialTransforms($(outsize))")
end


function apply(spatial::SpatialTransforms, context, datas::Tuple)
    items = makespatialitems(datas)
    tfm = _gettfm(spatial, context)
    return itemdata.(DataAugmentation.apply(tfm, items))
end


function apply!(bufs, spatial::SpatialTransforms, context, datas::Tuple)
    items = makespatialitems(datas)
    tfm = _gettfm(spatial, context)
    tdatas = itemdata.(DataAugmentation.apply(tfm, items))
    return tdatas
    _copyrec!(bufs, tdatas)
    return bufs
end

apply(spatial::SpatialTransforms, context, data) = apply(spatial, context, (data,)) |> only
apply!(buf, spatial::SpatialTransforms, context, data) = apply!((buf,), spatial, context, (data,)) |> only


## Utils

_gettfm(spatial::SpatialTransforms, context::Training) = spatial.traintfm
_gettfm(spatial::SpatialTransforms, context::Validation) = spatial.validtfm
_gettfm(spatial::SpatialTransforms, context::Inference) = spatial.inferencetfm


function makespatialitems(datas::Tuple)
    if datas[begin] isa Item
        return makespatialitems(datas, getbounds(datas[begin]))
    else
        return makespatialitems(datas, makebounds(size(datas[begin])))
    end
end
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
makeitem(datas::Tuple, args...) = Tuple(makeitem(data, args...) for data in datas)


itemtype(data::AbstractMatrix{<:Colorant}) = DataAugmentation.Image
itemtype(data::Vector{<:Union{Nothing, SVector}}) = DataAugmentation.Keypoints
