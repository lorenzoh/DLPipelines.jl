
"""
    methoddataset(data, method, context)

Transform data container `data` of samples into a data container of `(x, y)`-pairs.
Maps `encode(method, context, sample)` over the observations in `data`.
"""
@with_kw struct MethodDataset{M<:LearningMethod}
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


"""
    methoddataset(data, method, context)

Transform data container `data` of samples into a data container of `(x, y)`-pairs.
Maps `encode(method, context, sample)` over the observations in `data`.
"""
const methoddataset = MethodDataset


"""
    methoddataloaders(data, method)
    methoddataloaders(traindata, validdata, method[, batchsize; shuffle = true, dlkwargs...])

Create training and validation `DataLoader`s from two data containers `(traindata, valdata)`.
If only one container `data` is passed, splits it into two with `pctgvalid`% of the data
going into the validation split.

## Keyword arguments

- `batchsize = 16`
- `shuffle = true`: Whether to shuffle the training data container
- `validbsfactor`: Factor to multiply batchsize for validation data loader with (validation
    batches can be larger since no GPU memory is needed for the backward pass)

All remaining keyword arguments are passed to [`DataLoader`](#).
"""
function methoddataloaders(
        traindata,
        validdata,
        method::LearningMethod,
        batchsize = 16;
        shuffle = true,
        validbsfactor = 2,
        kwargs...)
    traindata = shuffle ? shuffleobs(traindata) : traindata
    return (
        DataLoader(methoddataset(traindata, method, Training()), batchsize; kwargs...),
        DataLoader(methoddataset(validdata, method, Validation()), validbsfactor * batchsize; kwargs...),
    )
end


function methoddataloaders(
        data,
        method::LearningMethod,
        batchsize = 16;
        pctgval = 0.2,
        shuffle = true,
        kwargs...)
    traindata, validdata = splitobs(shuffleobs(data), at = 1-pctgval)
    methoddataloaders(traindata, validdata, method, batchsize; shuffle = false, kwargs...)
end
