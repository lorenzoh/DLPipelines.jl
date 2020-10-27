# Method Interface


The Method interface specifies building blocks for constructing data pipelines. If you haven't already, see [./overview.md] for terminology.

The interface is built around are the abstract types [`Task`](#) and [`Method{Task}`](#). To define the steps in the data pipeline, you dispatch on `Method`.

Let's first translate the concepts introduced in [overview](./overview.md) into variables and types:

- `task::Task`: a task
- `method::Method{Task}`: a method implementing a task
- `input::I`: an input
- `target::T`: a target
- `x::X`: an encoded input
- `y::Y`: an encoded target
- `ŷ::Ŷ`: a model output
- `inputs`, `targets`, `xs`, `ys`: batches of the respective data