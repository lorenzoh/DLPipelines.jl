# # Image classification
#
# *See [`ImageClassification`](#) for complete documentation of its arguments.*
#
# Let's explore what you can do with the [`LearningMethod`](#) interface implemented. We're using
# [DLDatasets.jl](https://github.com/lorenzoh/DLDatasets.jl) to access *ImageNette*, a small
# image classification dataset. Install the package using
#
#   `]add https://github.com/lorenzoh/DLDatasets.jl`
# {cell=main style="display:none" result=false}

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
ENV["CI"] = "false"

# {cell=main output=false}

using DLPipelines
using DLDatasets
using LearnBase: getobs

# With the packages imported, loading the dataset is straightforward:
# {cell=main}

dataset = DLDatasets.loaddataset(ImageNette, "v2_160px")

# As you can see, every observation consists of an image and a category label:
# {cell=main}
image, category = getobs(dataset, 4000)
image
# {cell=main, result = false style="display:none;"}
@show category

# We can also retrieve the 10 different categories from the dataset's metadata:
# {cell=main}

categories = DLDatasets.metadata(ImageNette).labels

# ## Image classification `LearningMethod`
#
# With the dataset ready, we can create an instance of [`ImageClassification`](#).
#
# Note that you could use any image classification dataset, as long as `getobs(ds, idx)`
# returns an image and a category.
# {cell=main}

method = ImageClassification(categories, sz = (128, 128))

# We can now use this `method` with a [`Context`](#) to transform the data.
# The image is encoded as a normalized 3D-array:
# {cell=main}

x = encodeinput(method, Training(), image)
summary(x)

# And the category is one-hot encoded:
# {cell=main}

y = encodetarget(method, Training(), category)

# You can also use [`MethodDataset`](#) to create a wrapper around your existing dataset
# that directly returns encoded observations:
# {cell=main}

methoddataset = MethodDataset(dataset, method, Training())
x, y = getobs(methoddataset, 1)
summary.((x, y))
