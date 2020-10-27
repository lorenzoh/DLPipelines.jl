using Publish
using DeepLearningTasks

p = Publish.Project(DeepLearningTasks)
rm("dev", recursive = true, force = true)
rm(p.env["version"], recursive = true, force = true)
deploy(DeepLearningTasks; root = "/DeepLearningTasks.jl", force = true, label = "dev")
