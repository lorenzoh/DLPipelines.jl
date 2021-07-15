"""
    predict(method, model, input[; device, context])

Predict a `target` from `input` using `model`. Optionally apply function `device`
to `x` before passing to `model` and use `context` instead of the default
context [`Inference`](#).
"""
function predict(method, model, input; device = identity, undevice = identity, context = Inference())
    if shouldbatch(method)
        return predictbatch(method, model, [input,]; device = device, undevice = undevice, context = context) |> only
    else
        return decodeŷ(method, context, undevice(model(device(encodeinput(method, context, input)))))
    end
end


"""
    predictbatch(method, model, inputs[; device, context])

Predict `targets` from a vector of `inputs` using `model` by batching them.
Optionally apply function `device` to batch before passing to `model` and
use `context` instead of the default [`Inference`](#).
"""
function predictbatch(method, model, inputs; device = identity, undevice = identity, context = Inference())
    xs = device(DataLoaders.collate([copy(encodeinput(method, context, input)) for input in inputs]))
    ŷs = undevice(model(xs))
    targets = [decodeŷ(method, context, ŷ) for ŷ in DataLoaders.obsslices(ŷs)]
    return targets
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
