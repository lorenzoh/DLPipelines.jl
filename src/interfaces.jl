# # `interfaces.jl`
#
# Definitions of secondary interfaces.
#
# ## Inplace interface

encode!(buf, method, context, sample) = encode(method, context, sample)
encodeinput!(buf, method, context, input) = encodeinput(method, context, input)
encodetarget!(buf, method, context, target) = encodetarget(method, context, target)
decodey!(buf, method, context, y) = decodey(method, context, y)
decodeŷ!(buf, method, context, ŷ) = decodeŷ(method, context, ŷ)

# ## Interpretation interface

function interpretsample end
function interpretinput end
function interprettarget end
function interpretx end
function interpretŷ end
function interprety end



# ## Training interface

"""
    methodlossfn(method)

Default loss function to use when training models for `method`.
"""
function methodlossfn end


"""
    methodmodel(method, backbone)

Construct a model for `method` from a backbone architecture, for example
by attaching a method-specific head model.
"""
function methodmodel end


# ## Testing interface

"""
    mocksample(method)

Generate a random `sample` compatible with `method`.
"""
mocksample(method) = (mockinput(method), mocktarget(method))

"""
    mockinput(method)

Generate a random `input` compatible with `method`.
"""
function mockinput end

"""
    mocktarget(method)

Generate a random `target` compatible with `method`.
"""
function mocktarget end

"""
    mockmodel(method)

Generate a random `model` compatible with `method`.
"""
function mockmodel end
