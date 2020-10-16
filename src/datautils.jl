
@with_kw struct MappedData
    f
    data
end

LearnBase.getobs(data::MappedData, idx) = data.f(getobs(data.data, idx))
LearnBase.nobs(data::MappedData) = nobs(data.data)

function taskdataset(task, data; valid = false, obsfn = identity, shuffle = !valid)
    if shuffle
        data = shuffleobs(data)
    end
    return MappedData(data) do sample
        encode(task, obsfn(sample); augment = !valid)
    end
end

function taskdataloader(task, data, batchsize; valid = false, obsfn = identity, shuffle = !valid, kwargs...)
    data = taskdataset(task, data; valid = valid, obsfn = obsfn, shuffle = shuffle)
    DataLoader(data, batchsize; partial = valid, kwargs...)
end
