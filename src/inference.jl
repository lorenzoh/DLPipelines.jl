"""
    predict(method, model, input[; device, context])

Predict a `target` from `input` using `model`. Optionally apply function `device`
to `x` before passing to `model` and use `context` instead of the default, `Inference`.
"""
function predict(method, model, input; device = identity, context = Inference())
    if shouldbatch(method)
        return predictbatch(method, model, [input,]; device = device, context = context)[1]
    else
        return decodeŷ(method, context, model(device(encodeinput(method, context, input))))
    end
end


"""
    predictbatch(method, model, inputs[; device, context])

Predict `targets` from a batch of `inputs` using `model`. Optionally apply function `device`
to `xs` before passing to `model` and use `context` instead of the default, `Inference`.
"""
function predictbatch(method, model, inputs; device = gpu, context = Inference())
    xs = device(DataLoaders.collate([encodeinput(method, context, input) for input in inputs]))
    ŷs = model(xs)
    targets = [decodeŷ(method, context, ŷ) for ŷ in obsslices(cpu(ŷs))]
end


function _predictx(method, model, x, device = identity)
    if shouldbatch(method)
        x = DataLoaders.collate([x])
    end
    ŷs = device(model)(device(x))
    if shouldbatch(method)
        ŷ = ŷs[((:) for _ in 1:ndims(ŷs)-1)..., 1]
    end
    return ŷ
end
