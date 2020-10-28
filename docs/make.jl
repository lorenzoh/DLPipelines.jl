using Publish
using DLPipelines

p = Publish.Project(DLPipelines)
rm("dev", recursive = true, force = true)
rm(p.env["version"], recursive = true, force = true)
deploy(DLPipelines; root = "/DLPipelines.jl", force = true, label = "dev")
