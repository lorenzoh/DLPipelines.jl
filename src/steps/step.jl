abstract type PipelineStep end


"""
    apply(step::PipelineStep, context, data)

Applies the operation `step` to `data`
"""
function apply end

"""
    apply!(buf, step::PipelineStep, context)

Applies the operation `step` inplace to `buf`. `buf` is mutated.
"""
function apply! end


"""
    invert(step::PipelineStep, data, context)

Applies the inverse of the operation `step` to `data`
"""
function invert end


"""
    invert!(buf, step::PipelineStep, context)

Applies the inverse of the operation `step` to `buf` inplace. `buf` is mutated,
"""
function invert! end
