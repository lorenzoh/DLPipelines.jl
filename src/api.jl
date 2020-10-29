
## TODO: refactor to respect `shouldbatch`
function predict(task, model, input; device = cpu, batch = true)
    x = encodeinput(task, input; inference = true)
    xs = device(reshape(x, size(x)..., 1))
    ŷs = cpu(model(xs))
    return decodeoutput(task, ŷs[:, :, :, 1])
end

## TODO: implement
function predictbatch end


## TODO: implement
function dataiter end
