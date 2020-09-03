using DeepLearningPipelines
using Documenter

makedocs(;
    modules=[DeepLearningPipelines],
    authors="Lorenz Ohly <lorenz.ohly@gmail.com> and contributors",
    repo="https://github.com/lorenzoh/DeepLearningPipelines.jl/blob/{commit}{path}#L{line}",
    sitename="DeepLearningPipelines.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://lorenzoh.github.io/DeepLearningPipelines.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/lorenzoh/DeepLearningPipelines.jl",
)
