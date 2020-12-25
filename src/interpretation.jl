
# Interpretation
function interpretsample end
function interpretinput end
function interprettarget end
function interpretx end
function interpretŷ end
function interprety end
function interpretstep end


function interpretstepbatch(method, xs, ys, ŷs)
    ncol = 2 * round(Int, sqrt(size(xs)[end]))

    return mosaicview([
        interpretstep(method, x, y, ŷ)
        for (x, y, ŷ) in zip(obsslices(xs), obsslices(ys), obsslices(ŷs))
        ], ncol = ncol)
end
