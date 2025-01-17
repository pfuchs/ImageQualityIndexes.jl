module ImageQualityIndexes

using LazyModules
using OffsetArrays
using ImageCore
# Where possible we avoid a direct dependency to reduce the number of [compat] bounds
using ImageCore.MappedArrays
using ImageCore: NumberLike, Pixel, GenericImage, GenericGrayImage
@lazy import ImageDistances = "51556ac3-7006-55f5-8cb3-34580c88182d"
@lazy import ImageContrastAdjustment = "f332f351-ec65-5f6a-b3d1-319c6670881a"
@lazy import ImageFiltering = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
using Statistics: mean, std
using Base.Iterators: repeated, flatten

include("generic.jl")
include("psnr.jl")
include("ssim.jl")
include("xsim.jl")
include("msssim.jl")
include("colorfulness.jl")
include("entropy.jl")

include("precompile.jl")

export
    # generic
    assess,

    # Peak signal-to-noise ratio
    PSNR,
    assess_psnr,

    # Structral Similarity
    SSIM,
    assess_ssim,

    # Susceptibility Similarity
    XSIM,
    assess_xsim,

    # Multi Scale Structural Similarity
    MSSSIM,
    assess_msssim,

    # Colorfulness
    HASLER_AND_SUSSTRUNK_M3,
    hasler_and_susstrunk_m3,
    colorfulness,

    # used to live in Images.jl
    entropy

@static if VERSION < v"1.1"
    isnothing(x) = x === nothing
end

end # module
