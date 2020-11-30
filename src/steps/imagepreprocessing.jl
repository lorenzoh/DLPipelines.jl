# See **[`ImagePreprocessing`](#).**

"""
    ImagePreprocessing(means, stds[; C = RGB{N0f8}, T = Float32])

Converts an image to a color `C`, then to a 3D-array of type `T` and
finally normalizes the values using `means` and `stds`.

If no `means` or `stds` are given, uses ImageNet statistics.
"""
struct ImagePreprocessing
    tfm
end

function ImagePreprocessing(
        means::SVector{N} = IMAGENET_MEANS,
        stds::SVector{N} = IMAGENET_STDS;
        C = RGB{N0f8},
        T = Float32) where N
    # TODO: tensor of type T
    tfms = ToEltype(C) |> ImageToTensor() |> Normalize(means, stds)

    return ImagePreprocessing(InplaceThreadsafe(tfms))
end


function ImagePreprocessing(means::NTuple{N}, stds::NTuple{N}; kwargs...) where N
    return ImagePreprocessing(SVector{N}(means), SVector{N}(stds); kwargs...)
end


function apply(ip::ImagePreprocessing, ::Context, image)
    return DataAugmentation.apply(ip.tfm, DataAugmentation.Image(image)) |> itemdata
end


function apply!(x, ip::ImagePreprocessing, ::Context, image)
    return DataAugmentation.apply!(ArrayItem(x), ip.tfm, DataAugmentation.Image(image)) |> itemdata
end


function invert(ip::ImagePreprocessing, x)
    return DataAugmentation.tensortoimage(DataAugmentation.denormalize(x, ip.means, ip.stds))
end


const IMAGENET_MEANS = SVector{3, Float32}(.485, 0.456, 0.406)
const IMAGENET_STDS = SVector{3, Float32}(0.229, 0.224, 0.225)
