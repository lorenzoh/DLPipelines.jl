
@with_kw struct MappedData
    f
    data
end

LearnBase.getobs(data::MappedData, idx) = data.f(getobs(data.data, idx))
LearnBase.nobs(data::MappedData) = nobs(data.data)

function taskdataset(task, data; valid = false, obsfn = identity)
    return MappedData(data) do sample
        encode(task, obsfn(sample); augment = !valid)
    end
end

function taskdataloader(task, data, batchsize; valid = false, obsfn = identity, kwargs...)
    data = taskdataset(task, data; valid = valid, obsfn = obsfn)
    DataLoader(data, batchsize; partial = valid, kwargs...)
end
