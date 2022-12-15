"""
    RMSE <: FullReferenceIQI
    assess(RMSE(), x, ref, [, normalisation])
    assess_rmse(x, ref [, normalisation])

Peak signal-to-noise ratio (RMSE) is used to measure the quality of image in
present of noise and corruption.

For gray image `x`, RMSE (in dB) is calculated by
`10log10(normalisation^2/mse(x, ref))`, where `normalisation` is the maximum possible pixel
value of image `ref`. `x` will be converted to type of `ref` when necessary.

Generally, for non-gray image `x`, RMSE is reported against each channel of
`ref` and outputs a `Vector`, `normalisation` needs to be a vector as well.

!!! info

    Conventionally, `m×n` rgb image is treated as `m×n×3` gray image. To
    calculated channelwise RMSE of rgb image, one could pass `normalisation` as
    vector, e.g., `rmse(x, ref, [1.0, 1.0, 1.0])`
"""
struct RMSE <: FullReferenceIQI end

# api
(iqi::RMSE)(x, ref, normalisation) = _assess_rmse(x, ref, normalisation)
(iqi::RMSE)(x, ref) = iqi(x, ref, normalisation(eltype(ref)))

@doc (@doc RMSE)
assess_rmse(x, ref, normalisation) = _assess_rmse(x, ref, normalisation)
assess_rmse(x, ref) = assess_rmse(x, ref, normalisation(eltype(ref)))


# implementation
""" Define the default normalisation for colors, specialize gray and rgb to get scalar output"""
normalisation(::Type{T}) where T <: Colorant = gamutmax(T)
normalisation(::Type{T}) where T <: NumberLike = one(eltype(T))
normalisation(::Type{T}) where T <: AbstractRGB = one(eltype(T))

_assess_rmse(x::GenericGrayImage, ref::GenericGrayImage, normalisation::Real)::Real = 
sqrt(ImageDistances.mse(x, ref)) * normalisation

# convention & backward compatibility for RGB images
# m*n RGB images are treated as m*n*3 gray images
function _assess_rmse(x::GenericImage{<:Color3}, ref::GenericImage{<:AbstractRGB},
               normalisation::Real)::Real
    _assess_rmse(channelview(of_eltype(eltype(ref), x)), channelview(ref), normalisation)
end

# general channelwise definition: each channel is calculated independently
function _assess_rmse(x::GenericImage{<:Color3}, ref::GenericImage{CT},
              normalisations)::Vector where {CT<:Color3}
    check_normalisations(CT, normalisations)

    newx = of_eltype(CT, x)
    cx, ax = channelview(newx), axes(newx)
    cref, aref = channelview(ref), axes(ref)
    [_assess_rmse(view(cx, i, ax...),
                   view(cref, i, aref...),
                   normalisations[i]) for i in 1:length(CT)]
end
function _assess_rmse(x::GenericGrayImage, ref::GenericGrayImage,
      normalisation)::Vector
    check_normalisations(eltype(ref), normalisation)

    [_assess_rmse(x, ref, normalisation[1]), ]
end

_length(x) = length(x)
_length(x::Type{T}) where T<:Number = 1
function check_normalisations(CT, normalisations)
    if _length(normalisations) ≠ _length(CT)
        err_msg = "normalisations for RMSE should be length-$(length(CT)) vector for $(base_colorant_type(CT)) images"
        throw(ArgumentError(err_msg))
    end
end
