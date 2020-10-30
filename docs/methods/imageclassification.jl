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

ENV["DATADEPS_ALWAYS_ACCEPT"] = "yes"  #src

# {cell=main}

using DLPipelines
using DLDatasets
using LearnBase: getobs

# With the packages imported, loading the dataset is straightforward:
# {cell=main}

ds = DLDatasets.loaddataset(ImageNette, "v2_320px")

# As you can see, every observation consists of an image and a category label:
# {cell=main}
image, label = getobs(ds, 1)
image
# {cell=main, result = false style="display:none;"}
@show label

# We can also retrieve the 10 different categories from the dataset's metadata:
# {cell=main}

categories = DLDatasets.metadata(ImageNette).labels

# Of

# ## Image classification `Method`
#
# With the dataset ready, we can create an instance of [`ImageClassification`](#).
# The only
# required argument is `categories`:
# {cell=main}
method = ImageClassification(categories)
