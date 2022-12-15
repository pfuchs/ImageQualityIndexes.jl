"""
    XSIM([kernel], [(α, β, γ)]; crop=false) <: FullReferenceIQI
    assess(iqi::XSIM, img, ref)
    assess_xsim(img, ref; crop=false)

Structural similarity (XSIM) index is an image quality assessment method based
on degradation of structural information.

The XSIM index is composed of three components: luminance, contrast, and
structure; `xsim = 𝐿ᵅ * 𝐶ᵝ * 𝑆ᵞ`, where `W := (α, β, γ)` controls relative
importance of each components. By default `W = (1.0, 1.0, 1.0)`.

In practice, a mean version XSIM is used. At each pixel, XSIM is calculated
locally with neighborhoods weighted by `kernel`, returning a Xsim map;
`Xsim` is actaully `mean(Xsim_map)`.
By default `kernel = KernelFactors.gaussian(1.5, 11)`.

!!! tip

    The default parameters comes from [1]. For benchmark usage, it is recommended to not
    change the parameters, because most other XSIM implementations follows the same settings.
    Keyword `crop` controls whether the boundary of Xsim map should be dropped; you need
    to set it `true` to reproduce the result in [1].

# Example

`assess_xsim(img, ref)` should be sufficient to get a benchmark for algorithms. One
could also instantiate a customed XSIM, then pass it to `assess` or use it as a
function. For example:

```julia
iqi = XSIM(KernelFactors.gaussian(2.5, 17), (1.0, 1.0, 2.0))
assess(iqi, img, ref)
iqi(img, ref) # both usages are equivalent
```

!!! info

    XSIM is defined only for gray images. How `RGB` and other `Color3` images are handled may vary
    in different implementations. For `RGB` images, channels are handled seperately when calculating
    𝐿, 𝐶, and 𝑆. Generic `Color3` images are converted to `RGB` first before calculation.

# Reference

[1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: from error visibility to structural similarity. _IEEE transactions on image processing_, 13(4), 600-612.

[2] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2003). The XSIM Index for Image Quality Assessment. Retrived May 30, 2019, from http://www.cns.nyu.edu/~lcv/Xsim/
"""
struct XSIM{A<:AbstractVector} <: FullReferenceIQI
    kernel::A
    W::NTuple{3, Float64}
    crop::Bool
    function XSIM(kernel::AbstractVector=XSIM_KERNEL, W::NTuple=XSIM_W; crop=false)
        ndims(kernel) == 1 || throw(ArgumentError("only 1-d kernel is valid"))
        issymetric(kernel) || @warn "XSIM kernel is assumed to be symmetric"
        all(W .>= 0) || throw(ArgumentError("(α, β, γ) should be non-negative, instead it's $(W)"))
        kernel = OffsetArrays.centered(kernel)
        new{typeof(kernel)}(kernel, W, crop)
    end
end

# default values from [1]
# kernel generated from ImageFiltering.KernelFactors.gaussian(1.5, 3)
# don't use ImageFiltering directly here because we lazy-import that dependency
const XSIM_KERNEL = OffsetArray([0.30780132912346997
0.38439734175306
0.30780132912346997], -1:1)
const XSIM_W = (0.0, 1.0, 1.0) # (α, β, γ) -- PFUCHS for XSIM --

Base.:(==)(ia::XSIM, ib::XSIM) = ia.kernel == ib.kernel && ia.W == ib.W && ia.crop == ib.crop

# api
# By default we don't crop the padding boundary to match the Xsim result from
# MATLAB Image Processing Toolbox, which is used more broadly than the original
# implementaion [2] (written in MATLAB as well).
# -- Johnny Chen <johnnychen94@hotmail.com>
(iqi::XSIM)(x, ref) = mean(_Xsim_map(iqi, x, ref))

@doc (@doc XSIM)
assess_xsim(x, ref; crop=false) = XSIM(;crop=crop)(x, ref)

# Parameters `(K₁, K₂)` are used to avoid instability when denominator is very
# close to zero. Different from origianl implementation [2], we don't make it
# public since the Xsim result is insensitive to these parameters according to
# [1].
# -- Johnny Chen <johnnychen94@hotmail.com>
const XSIM_K = (0.01, 0.001) # -- PFUCHS for XSIM --

# XSIM is defined only for gray images,
# RGB images are treated as 3d gray images,
# other Color3 images are converted to RGB first.
function _Xsim_map(iqi::XSIM,
                   x::Union{GenericGrayImage, AbstractArray{<:AbstractRGB}},
                   ref::Union{GenericGrayImage, AbstractArray{<:AbstractRGB}},
                   peakval = 1.0,
                   K = XSIM_K)
    if size(x) ≠ size(ref)
        err = ArgumentError("images should be the same size, instead they're $(size(x))-$(size(ref))")
        throw(err)
    end
    if axes(x) ≠ axes(ref)
        x = OffsetArrays.no_offset_view(x)
        ref = OffsetArrays.no_offset_view(ref)
    end
    α, β, γ = iqi.W
    C₁, C₂ = @. (peakval * K)^2
    C₃ = C₂/2

    T = promote_type(float(eltype(ref)), float(eltype(x)))
    x = of_eltype(T, x)
    ref = of_eltype(T, ref)

    if [α, β, γ] ≈ [1.0, 1.0, 1.0]
        Xsim_map = __Xsim_map_fast(x, ref, iqi.kernel, C₁, C₂; crop=iqi.crop)
    else
        l, c, s = __Xsim_map_general(x, ref, iqi.kernel, C₁, C₂, C₃; crop=iqi.crop)
        # ensure that negative numbers in s are not being raised to powers less than 1
        # this is a non-standard implementation and it could vary from implementaion to implementaion
        γ < 1.0 && (s = max.(s, zero(eltype(s))))
        Xsim_map = @. l^α * c^β * s^γ # equation (12) in [1]
    end
    return Xsim_map
end

function _Xsim_map(iqi::XSIM,
          x::AbstractArray{<:Color3},
          ref::AbstractArray{<:Color3},
          peakval = 1.0, K = XSIM_K)
    C = ccolor(RGB, floattype(eltype(x)))
    _Xsim_map(iqi, of_eltype(C, x), of_eltype(C, ref), peakval, K)
end


# helpers
function issymetric(kernel)
    origin = first(axes(kernel, 1))
    center = (length(kernel)-1) ÷ 2 + origin
    kernel[origin:center] ≈ kernel[end:-1:center]
end


function __Xsim_map_fast(x, ref, window, C₁, C₂; crop)
    μx², μxy, μy², σx², σxy, σy² = _Xsim_statistics(x, ref, window; crop=crop)
    # this is a special case when [α, β, γ] ≈ [1.0, 1.0, 1.0]
    # equation (13) in [1]
    return @. ((2μxy + C₁)*(2σxy + C₂))/((μx²+μy²+C₁)*(σx² + σy² + C₂))
end

function __Xsim_map_general(x, ref, window, C₁, C₂, C₃; crop)
    μx², μxy, μy², σx², σxy, σy² = _Xsim_statistics(x, ref, window; crop = crop)

    T = eltype(σx²)
    σx_σy = @. sqrt( max(σx²*σy², zero(T) ))
    l = @. (2μxy + C₁)/(μx² + μy²) # equation (6) in [1]
    c = @. (2σx_σy + C₂)/(σx² + σy² + C₂) # equation (9) in [1]
    s = @. (σxy + C₃)/(σx_σy + C₃) # equation (10) in [1]

    # MS-XSIM needs these, so we don't multiply them together here
    return l, c, s
end

function _Xsim_statistics(x::GenericImage, ref::GenericImage, window; crop)
    # For RGB and other Color3 images, we don't slide the window at the color channel.
    # In other words, these characters will be calculated channelwisely

    region = map(axes(x)) do a
        o = length(window) ÷ 2
        # Even if crop=true, it crops only when image is larger than window
        length(a) > length(window) ? (first(a)+o:last(a)-o) : a
    end
    R = crop ? CartesianIndices(region) : CartesianIndices(x)

    window = ImageFiltering.kernelfactors(Tuple(repeated(window, ndims(ref))))

    # don't slide the window in the channel dimension
    μx = view(ImageFiltering.imfilter(x,   window, "symmetric"), R) # equation (14) in [1]
    μy = view(ImageFiltering.imfilter(ref, window, "symmetric"), R) # equation (14) in [1]
    μx² = _mul.(μx, μx)
    μy² = _mul.(μy, μy)
    μxy = _mul.(μx, μy)
    σx² = view(ImageFiltering.imfilter(_mul.(x,   x  ), window, "symmetric"), R) .- μx² # equation (15) in [1]
    σy² = view(ImageFiltering.imfilter(_mul.(ref, ref), window, "symmetric"), R) .- μy² # equation (15) in [1]
    σxy = view(ImageFiltering.imfilter(_mul.(x,   ref), window, "symmetric"), R) .- μxy # equation (16) in [1]

    # after that, channel dimension can be treated generically so we expand them here
    return channelview.((μx², μxy, μy², σx², σxy, σy²))
end
