
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
