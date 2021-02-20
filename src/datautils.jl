
"""
    methoddataset(data, method, context)

Transform data container `data` of samples into a data container of `(x, y)`-pairs.
Maps `encode(method, context, sample)` over the observations in `data`.
"""
@with_kw struct MethodDataset{M<:Method}
    data
    method::M
    context::Context
end

LearnBase.nobs(ds::MethodDataset) = nobs(ds.data)

function LearnBase.getobs(ds::MethodDataset, idx)
    return encode(ds.method, ds.context, getobs(ds.data, idx))
end

function LearnBase.getobs!(buf, ds::MethodDataset, idx)
    return encode!(buf, ds.method, ds.context, getobs(ds.data, idx))
end

const methoddataset = MethodDataset


"""
    methoddataloaders((traindata, validdata), method[; batchsize, dlkwargs...])
    methoddataloaders(data, method[; pctgvalid, batchsize, dlkwargs])

Create training and validation `DataLoader`s from two data containers `(traindata, valdata)`.
If only one container `data` is passed, splits it into two with `pctgvalid`% of the data
going into the validation split.

Other keyword arguments are passed to `DataLoader`s.
"""
function methoddataloaders(
        (traindata, valdata)::NTuple{2},
        method;
        batchsize::Int = 16,
        validbsfactor::Int = 2,
        pctgval = nothing,
        kwargs...)
    return (
        DataLoader(methoddataset(traindata, method, Training()), batchsize; kwargs...),
        DataLoader(methoddataset(validdata, method, Validation()), batchsize * validbsfactor; kwargs...),
    )
end

methoddataloaders(data, method; pctgval = 0.2, kwargs...) =
    methoddataloaders(splitobs(data, at = pctgval), method; kwargs...)
