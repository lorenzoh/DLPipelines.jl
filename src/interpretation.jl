
# Interpretation
function interpretsample(method, sample) end
function interpretinput(method, input) end
function interprettarget(method, target) end
function interpretx(method, x) end
function interpretŷ(method, ŷ) end
function interprety(method, y) end
function interpretstep(method, x, y, ŷ) end


function interpretstepbatch(method, xs, ys, ŷs)
    ncol = 2 * round(Int, sqrt(size(xs)[end]))

    return mosaicview([
        interpretstep(method, x, y, ŷ)
        for (x, y, ŷ) in zip(obsslices(xs), obsslices(ys), obsslices(ŷs))
        ], ncol = ncol)
end
