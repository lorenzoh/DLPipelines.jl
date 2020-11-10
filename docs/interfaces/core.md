
# Core interface

First you need to define a custom [`Method`](#) (and possibly a `Task`).

Then implement the following functions by dispatching on the method.

|  Required functions | Description                                                                          |
| ------------------: | :----------------------------------------------------------------------------------- |
|  [`encodeinput`](#) | Encodes an input so that it can be fed to a model                                    |
|      [`decodeŷ`](#) | Decodes the output of a model into a target                                          |
|       [`encode`](#) | Encodes a sample containing input and target so they can be used for a training step |
|  [`encodeinput`](#) | Encodes an input so that it can be fed to a model                                    |
| [`encodetarget`](#) | Encodes a target so that it can be compared to a model output using a loss function  |

#

| Optional functions | Description                                                   |
| -----------------: | :------------------------------------------------------------ |
| [`shouldbatch`](#) | Defines whether the model for a method should take batches of |
|                    | encoded inputs. The default is `true`.                        |


- *What is the `context` argument?*

  We often want to apply data augmentation during training but not during validation or inference. You can dispatch on `context` to define different behavior for the different situations. See [`Context`](#) to see the available options.

- *When should I implement `encode` vs. `encodetarget`?* 

  See [`encode`](#).
