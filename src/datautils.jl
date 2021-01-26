
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

const methoddataset = MethodDataset
