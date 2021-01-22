# Interfaces

DLPipelines.jl has multiple interfaces that you can implement for your [`Method`](#)s.

[Core interface](./core.md), for inference and training pipelines. Everything you need to use [`predict`](#), [`predictbatch`](#), [`methoddataset`](#) and [`methoddataloaders`](#).

- [`encode`](#)
- [`encodeinput`](#)
- [`encodetarget`](#)
- [`decodeŷ`](#)

Buffered interface, to enable allocation-free pipelines. Mimicks the core interface.

- [`encode!`](#)
- [`encodeinput!`](#)
- [`encodetarget!`](#)
- [`decodeŷ!`](#)

Interpretation interface, for visualizing and making sense of the data at different steps.

- [`interpretinput`](#)
- [`interprettarget`](#)
- [`interpretx`](#)
- [`interprety`](#)
- [`interpretŷ`](#)

Training interface.

- [`methodlossfn`](#)
- [`methodmodel`](#)
