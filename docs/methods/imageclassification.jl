# # Image classification
#
# *See [`ImageClassification`](#)'s for complete documentation of its arguments*
#
# Let's explore what you can do with the [`Method`](#) interface implemented. We're using
# [DLDatasets.jl](https://github.com/lorenzoh/DLDatasets.jl) to access *ImageNette*, a small
# image classification dataset. Install the package using
#
#   `]add https://github.com/lorenzoh/DLDatasets.jl`
# {cell=main style="display:none"}

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"  #src

# {cell=main}

using DLPipelines
using DLDatasets
using LearnBase: getobs

# With the packages imported, loading the dataset is straightforward:
# {cell=main}

ds = DLDatasets.loaddataset(ImageNette, "v2_160px")

# As you can see, every observation consists of an image and a category label:
# {cell=main}
image, category = getobs(ds, 1)
image
# {cell=main, result = false style="display:none;"}
@show category

# We can also retrieve the 10 different categories from the dataset's metadata:
# {cell=main}

categories = DLDatasets.metadata(ImageNette).labels

# ## Image classification `Method`
#
# With the dataset ready, we can create an instance of [`ImageClassification`](#).
#
# Note that you could use any image classification dataset, as long as `getobs(ds, idx)`
# returns an image and a category.
# {cell=main}

method = ImageClassification(categories)

# We can now use this `method` with a [`Context`](#) to transform the data.
# The image is encoded as a normalized 3D-array:
# {cell=main}

x = encodeinput(method, Training(), image)
summary(x)

# And the category is one-hot encoded:
# {cell=main}

y = encodetarget(method, Training(), category)

# [`dataiter`](#) will create a data iterator over batches of properly encoded inputs and targets:
# {cell=main}

traindata = dataiter(method, ds, Training(), 16)
summary.(first(traindata))
