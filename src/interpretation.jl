
# Interpretation

# TODO: refactor and document
function interpretinput(task, input) end
function interprettarget(task, target) end
function interpretx(task, x) end
interprety(task, y) = interprettarget(task, decodeoutput(task, y))
