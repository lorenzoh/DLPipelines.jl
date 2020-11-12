
## TODO: refactor to respect `shouldbatch`
function predict(method, model, input; device = gpu, context = Inference())
    if shouldbatch(method)
        return predictbatch(method, model, (input,); device = device, context = context)[1]
    else
        error("Not implemented")
    end
end


function predictbatch(method, model, inputs; device = gpu, context = Inference())
    xs = device(DataLoaders.collate([encodeinput(method, context, input) for input in inputs]))
    @time ŷs = model(xs)
    targets = [decodeŷ(method, context, ŷ) for ŷ in obsslices(cpu(ŷs))]
end
