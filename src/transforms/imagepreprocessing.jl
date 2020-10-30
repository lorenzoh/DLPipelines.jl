# See **[`ImagePreprocessing`](#).**

"""
    ImagePreprocessing([C], means, stds)

Converts an image to a color, then to a 3D-tensor and
finally normalizes the values using `means` and `stds`.

If no `means` or `stds` are given, use ImageNet statistics.
"""
@with_kw struct ImagePreprocessing
    C::Type{<:Colorant} = RGB
    means = IMAGENET_MEANS
    stds = IMAGENET_STDS
end

function (ip::ImagePreprocessing)(image)
    tfms = ToEltype{ip.C}() |> SplitChannels() |> Normalize(ip.means, ip.stds)
    apply(tfms, DataAugmentation.Image(image)) |> itemdata
end


function invert(ip::ImagePreprocessing, x)
    return DataAugmentation.tensortoimage(DataAugmentation.denormalize(x, ip.means, ip.stds))
end


const IMAGENET_MEANS = [0.485, 0.456, 0.406]
const IMAGENET_STDS = [0.229, 0.224, 0.225]
