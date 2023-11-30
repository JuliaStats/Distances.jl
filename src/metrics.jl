# Ordinary metrics

###########################################################
#
#   Abstract metric types
#
###########################################################

abstract type UnionPreMetric <: PreMetric end

abstract type UnionSemiMetric <: SemiMetric end

abstract type UnionMetric <: Metric end

###########################################################
#
#   Metric types
#
###########################################################

struct Euclidean <: UnionMetric
    thresh::Float64
end

"""
    Euclidean([thresh])

Create a euclidean metric.

When computing distances among large numbers of points, it can be much
more efficient to exploit the formula

    (x-y)^2 = x^2 - 2xy + y^2

However, this can introduce roundoff error. `thresh` (which defaults
to 0) specifies the relative square-distance tolerance on `2xy`
compared to `x^2 + y^2` to force recalculation of the distance using
the more precise direct (elementwise-subtraction) formula.

# Example:
```julia
julia> x = reshape([0.1, 0.3, -0.1], 3, 1);

julia> pairwise(Euclidean(), x, x)
1×1 Array{Float64,2}:
 7.45058e-9

julia> pairwise(Euclidean(1e-12), x, x)
1×1 Array{Float64,2}:
 0.0
```
"""
Euclidean() = Euclidean(0)

struct WeightedEuclidean{W} <: UnionMetric
    weights::W
end

"""
    PeriodicEuclidean(L)

Create a Euclidean metric on a rectangular periodic domain (i.e., a torus or
a cylinder). Periods per dimension are contained in the vector `L`:
```math
\\sqrt{\\sum_i(\\min\\mod(|x_i - y_i|, p), p - \\mod(|x_i - y_i|, p))^2}.
```
For dimensions without periodicity put `Inf` in the respective component.

# Example
```jldoctest
julia> x, y, L = [0.0, 0.0], [0.75, 0.0], [0.5, Inf];

julia> evaluate(PeriodicEuclidean(L), x, y)
0.25
```
"""
struct PeriodicEuclidean{W} <: UnionMetric
    periods::W
end

struct SqEuclidean <: UnionSemiMetric
    thresh::Float64
end

"""
    SqEuclidean([thresh])

Create a squared-euclidean semi-metric. For the meaning of `thresh`,
see [`Euclidean`](@ref).
"""
SqEuclidean() = SqEuclidean(0)

struct WeightedSqEuclidean{W} <: UnionSemiMetric
    weights::W
end

struct Chebyshev <: UnionMetric end

struct Cityblock <: UnionMetric end
struct WeightedCityblock{W} <: UnionMetric
    weights::W
end

struct TotalVariation <: UnionMetric end
struct Jaccard <: UnionMetric end
struct RogersTanimoto <: UnionMetric end

struct Minkowski{T <: Real} <: UnionMetric
    p::T
end
struct WeightedMinkowski{W,T <: Real} <: UnionMetric
    weights::W
    p::T
end

struct Hamming <: UnionMetric end
struct WeightedHamming{W} <: UnionMetric
    weights::W
end

struct CosineDist <: UnionSemiMetric end
# CorrDist is excluded from `UnionMetrics`
struct CorrDist <: SemiMetric end
struct BrayCurtis <: UnionSemiMetric end

struct ChiSqDist <: UnionSemiMetric end
struct KLDivergence <: UnionPreMetric end
struct GenKLDivergence <: UnionPreMetric end

"""
    RenyiDivergence(α::Real)
    renyi_divergence(P, Q, α::Real)

Create a Rényi premetric of order α.

Rényi defined a spectrum of divergence measures generalising the
Kullback–Leibler divergence (see `KLDivergence`). The divergence is
not a semimetric as it is not symmetric. It is parameterised by a
parameter α, and is equal to Kullback–Leibler divergence at α = 1:

At α = 0, ``R_0(P | Q) = -log(sum_{i: p_i > 0}(q_i))``

At α = 1, ``R_1(P | Q) = sum_{i: p_i > 0}(p_i log(p_i / q_i))``

At α = ∞, ``R_∞(P | Q) = log(sup_{i: p_i > 0}(p_i / q_i))``

Otherwise ``R_α(P | Q) = log(sum_{i: p_i > 0}((p_i ^ α) / (q_i ^ (α - 1))) / (α - 1)``

# Example:
```jldoctest
julia> x = reshape([0.1, 0.3, 0.4, 0.2], 2, 2);

julia> pairwise(RenyiDivergence(0), x, x)
2×2 Array{Float64,2}:
 0.0  0.0
 0.0  0.0

julia> pairwise(Euclidean(2), x, x)
2×2 Array{Float64,2}:
 0.0       0.577315
 0.655407  0.0
```
"""
struct RenyiDivergence{T <: Real} <: UnionPreMetric
    p::T # order of power mean (order of divergence - 1)
    is_normal::Bool
    is_zero::Bool
    is_one::Bool
    is_inf::Bool

    function RenyiDivergence{T}(q) where {T}
        # There are four different cases:
        #   simpler to separate them out now, not over and over in eval_op()
        is_zero = q ≈ zero(T)
        is_one = q ≈ one(T)
        is_inf = isinf(q)

        # Only positive Rényi divergences are defined
        !is_zero && q < zero(T) && throw(ArgumentError("Order of Rényi divergence not legal, $(q) < 0."))

        new{T}(q - 1, !(is_zero || is_one || is_inf), is_zero, is_one, is_inf)
    end
end
RenyiDivergence(q::T) where {T} = RenyiDivergence{T}(q)

struct JSDivergence <: UnionSemiMetric end

struct SpanNormDist <: UnionSemiMetric end

# Deviations are handled separately from the other distances/divergences and
# are excluded from `UnionMetrics`
struct MeanAbsDeviation <: Metric end
struct MeanSqDeviation <: SemiMetric end
struct RMSDeviation <: Metric end
struct NormRMSDeviation <: PreMetric end

# Union types
const metrics = (Euclidean,SqEuclidean,PeriodicEuclidean,Chebyshev,Cityblock,TotalVariation,Minkowski,Hamming,Jaccard,RogersTanimoto,CosineDist,ChiSqDist,KLDivergence,RenyiDivergence,BrayCurtis,JSDivergence,SpanNormDist,GenKLDivergence)
const weightedmetrics = (WeightedEuclidean,WeightedSqEuclidean,WeightedCityblock,WeightedMinkowski,WeightedHamming)
const UnionMetrics = Union{UnionPreMetric,UnionSemiMetric,UnionMetric}

###########################################################
#
#  Implementations
#
###########################################################

parameters(::UnionPreMetric) = nothing
parameters(::UnionSemiMetric) = nothing
parameters(::UnionMetric) = nothing
parameters(d::PeriodicEuclidean) = d.periods
for dist in weightedmetrics
    @eval parameters(d::$dist) = d.weights
end

result_type(dist::UnionMetrics, ::Type{Ta}, ::Type{Tb}) where {Ta,Tb} =
    result_type(dist, _eltype(Ta), _eltype(Tb), parameters(dist))
result_type(dist::UnionMetrics, ::Type{Ta}, ::Type{Tb}, ::Nothing) where {Ta,Tb} =
    typeof(_evaluate(dist, oneunit(Ta), oneunit(Tb)))
result_type(dist::UnionMetrics, ::Type{Ta}, ::Type{Tb}, p) where {Ta,Tb} =
    typeof(_evaluate(dist, oneunit(Ta), oneunit(Tb), oneunit(_eltype(p))))

Base.@propagate_inbounds function _evaluate(d::UnionMetrics, a, b)
    _evaluate(d, a, b, parameters(d))
end

# breaks the implementation into eval_start, eval_op, eval_reduce and eval_end

Base.@propagate_inbounds function _evaluate(d::UnionMetrics, a, b, ::Nothing)
    @boundscheck if length(a) != length(b)
        throw(DimensionMismatch("first collection has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end
    if length(a) == 0
        return zero(result_type(d, a, b))
    end
    s = eval_start(d, a, b)
    @inbounds for (ai, bi) in zip(a, b)
        s = eval_reduce(d, s, eval_op(d, ai, bi))
    end
    return eval_end(d, s)
end
Base.@propagate_inbounds function _evaluate(d::UnionMetrics, a::AbstractArray, b::AbstractArray, ::Nothing)
    @boundscheck if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end
    if length(a) == 0
        return zero(result_type(d, a, b))
    end
    @inbounds begin
        s = eval_start(d, a, b)
        if (IndexStyle(a, b) === IndexLinear() && eachindex(a) == eachindex(b)) || axes(a) == axes(b)
            @simd for I in eachindex(a, b)
                ai = a[I]
                bi = b[I]
                s = eval_reduce(d, s, eval_op(d, ai, bi))
            end
        else
            for (ai, bi) in zip(a, b)
                s = eval_reduce(d, s, eval_op(d, ai, bi))
            end
        end
        return eval_end(d, s)
    end
end

Base.@propagate_inbounds function _evaluate(d::UnionMetrics, a, b, p)
    @boundscheck if length(a) != length(b)
        throw(DimensionMismatch("first collection has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end
    @boundscheck if length(a) != length(p)
        throw(DimensionMismatch("data collections have length $(length(a)) but parameters have length $(length(p))."))
    end
    if length(a) == 0
        return zero(result_type(d, a, b))
    end
    s = eval_start(d, a, b)
    @inbounds for (ai, bi, pi) in zip(a, b, p)
        s = eval_reduce(d, s, eval_op(d, ai, bi, pi))
    end
    return eval_end(d, s)
end
Base.@propagate_inbounds function _evaluate(d::UnionMetrics, a::AbstractArray, b::AbstractArray, p::AbstractArray)
    @boundscheck if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end
    @boundscheck if length(a) != length(p)
        throw(DimensionMismatch("arrays have length $(length(a)) but parameters have length $(length(p))."))
    end
    if length(a) == 0
        return zero(result_type(d, a, b))
    end
    @inbounds begin
        s = eval_start(d, a, b)
        if (IndexStyle(a, b, p) === IndexLinear() && eachindex(a) == eachindex(b) == eachindex(p)) ||
                axes(a) == axes(b) == axes(p)
            @simd for I in eachindex(a, b, p)
                ai = a[I]
                bi = b[I]
                pi = p[I]
                s = eval_reduce(d, s, eval_op(d, ai, bi, pi))
            end
        else
            for (ai, bi, pi) in zip(a, b, p)
                s = eval_reduce(d, s, eval_op(d, ai, bi, pi))
            end
        end
        return eval_end(d, s)
    end
end

_evaluate(dist::UnionMetrics, a::Number, b::Number, ::Nothing) = eval_end(dist, eval_op(dist, a, b))
function _evaluate(dist::UnionMetrics, a::Number, b::Number, p)
    length(p) != 1 && throw(DimensionMismatch("inputs are scalars but parameters have length $(length(p))."))
    eval_end(dist, eval_op(dist, a, b, first(p)))
end

eval_start(d::UnionMetrics, a, b) = _eval_start(d, _eltype(a), _eltype(b))
_eval_start(d::UnionMetrics, ::Type{Ta}, ::Type{Tb}) where {Ta,Tb} =
    _eval_start(d, Ta, Tb, parameters(d))
_eval_start(d::UnionMetrics, ::Type{Ta}, ::Type{Tb}, ::Nothing) where {Ta,Tb} =
    zero(typeof(eval_op(d, oneunit(Ta), oneunit(Tb))))
_eval_start(d::UnionMetrics, ::Type{Ta}, ::Type{Tb}, p) where {Ta,Tb} =
    zero(typeof(eval_op(d, oneunit(Ta), oneunit(Tb), oneunit(_eltype(p)))))
eval_reduce(::UnionMetrics, s1, s2) = s1 + s2
eval_end(::UnionMetrics, s) = s

for M in (metrics..., weightedmetrics...)
    @eval @inline (dist::$M)(a, b) = _evaluate(dist, a, b, parameters(dist))
end

# Euclidean
@inline eval_op(::Euclidean, ai, bi) = abs2(ai - bi)
eval_end(::Euclidean, s) = sqrt(s)
const euclidean = Euclidean()

# Weighted Euclidean
@inline eval_op(::WeightedEuclidean, ai, bi, wi) = abs2(ai - bi) * wi
eval_end(::WeightedEuclidean, s) = sqrt(s)
weuclidean(a, b, w) = WeightedEuclidean(w)(a, b)

# PeriodicEuclidean
@inline function eval_op(::PeriodicEuclidean, ai, bi, p)
    s1 = abs(ai - bi)
    s2 = mod(s1, p)
    s3 = min(s2, p - s2)
    abs2(s3)
end
eval_end(::PeriodicEuclidean, s) = sqrt(s)
peuclidean(a, b, p) = PeriodicEuclidean(p)(a, b)

# SqEuclidean
@inline eval_op(::SqEuclidean, ai, bi) = abs2(ai - bi)
const sqeuclidean = SqEuclidean()

# Weighted Squared Euclidean
@inline eval_op(::WeightedSqEuclidean, ai, bi, wi) = abs2(ai - bi) * wi
wsqeuclidean(a, b, w) = WeightedSqEuclidean(w)(a, b)

# Cityblock
@inline eval_op(::Cityblock, ai, bi) = abs(ai - bi)
const cityblock = Cityblock()

# Weighted City Block
@inline eval_op(::WeightedCityblock, ai, bi, wi) = abs((ai - bi) * wi)
wcityblock(a, b, w) = WeightedCityblock(w)(a, b)

# Total variation
@inline eval_op(::TotalVariation, ai, bi) = abs(ai - bi)
eval_end(::TotalVariation, s) = s / 2
const totalvariation = TotalVariation()

# Chebyshev
@inline eval_op(::Chebyshev, ai, bi) = abs(ai - bi)
@inline eval_reduce(::Chebyshev, s1, s2) = max(s1, s2)
# if only NaN, will output NaN
Base.@propagate_inbounds eval_start(::Chebyshev, a, b) = abs(first(a) - first(b))
const chebyshev = Chebyshev()

# Minkowski
@inline eval_op(dist::Minkowski, ai, bi) = abs(ai - bi)^dist.p
@inline eval_end(dist::Minkowski, s) = s^(1 / dist.p)
minkowski(a, b, p::Real) = Minkowski(p)(a, b)

# Weighted Minkowski
@inline eval_op(dist::WeightedMinkowski, ai, bi, wi) = abs(ai - bi)^dist.p * wi
@inline eval_end(dist::WeightedMinkowski, s) = s^(1 / dist.p)
wminkowski(a, b, w, p::Real) = WeightedMinkowski(w, p)(a, b)

# Hamming
result_type(::Hamming, ::Type, ::Type) = Int # fallback for Hamming
eval_start(d::Hamming, a, b) = 0
@inline eval_op(::Hamming, ai, bi) = ai != bi ? 1 : 0
const hamming = Hamming()

# WeightedHamming
@inline eval_op(::WeightedHamming, ai, bi, wi) = ai != bi ? wi : zero(eltype(wi))
whamming(a, b, w) = WeightedHamming(w)(a, b)

# Cosine dist
@inline eval_start(dist::CosineDist, a, b) =
    zero.(typeof.(eval_op(dist, oneunit(_eltype(a)), oneunit(_eltype(b)))))
@inline eval_op(::CosineDist, ai, bi) = ai * bi, ai * ai, bi * bi
@inline function eval_reduce(::CosineDist, s1, s2)
    a1, b1, c1 = s1
    a2, b2, c2 = s2
    return a1 + a2, b1 + b2, c1 + c2
end
function eval_end(::CosineDist, s)
    ab, a2, b2 = s
    max(1 - ab / (sqrt(a2) * sqrt(b2)), 0)
end
const cosine_dist = CosineDist()

# CorrDist
_centralize(x) = x .- mean(x)
(::CorrDist)(a, b) = CosineDist()(_centralize(a), _centralize(b))
(::CorrDist)(a::Number, b::Number) = CosineDist()(zero(mean(a)), zero(mean(b)))
const corr_dist = CorrDist()

# ChiSqDist
@inline eval_op(::ChiSqDist, ai, bi) = (d = abs2(ai - bi) / (ai + bi); ifelse(ai != bi, d, zero(d)))
const chisq_dist = ChiSqDist()

# KLDivergence
@inline eval_op(dist::KLDivergence, ai, bi) =
    iszero(ai) ? zero(eval_op(dist, oneunit(ai), bi)) : ai * log(ai / bi)
const kl_divergence = KLDivergence()

# GenKLDivergence
@inline eval_op(dist::GenKLDivergence, ai, bi) =
    iszero(ai) ? oftype(eval_op(dist, oneunit(ai), bi), bi) : ai * log(ai / bi) - ai + bi
const gkl_divergence = GenKLDivergence()

# RenyiDivergence
Base.@propagate_inbounds function eval_start(::RenyiDivergence, a, b)
    T = promote_type(_eltype(a), _eltype(b))
    zero(T), zero(T), T(sum(a)), T(sum(b))
end

@inline function eval_op(dist::RenyiDivergence, ai::T, bi::T) where T
    if iszero(ai)
        return zero(T), zero(T), zero(T), zero(T)
    elseif dist.is_normal
        return ai, ai * ((ai / bi)^dist.p), zero(T), zero(T)
    elseif dist.is_zero
        return ai, bi, zero(T), zero(T)
    elseif dist.is_one
        return ai, ai * log(ai / bi), zero(T), zero(T)
    else # otherwise q = ∞
        return ai, ai / bi, zero(T), zero(T)
    end
end

@inline function eval_reduce(dist::RenyiDivergence,
                             s1::Tuple{T,T,T,T},
                             s2::Tuple{T,T,T,T}) where T
    if dist.is_inf
        if iszero(s1[1])
            return (s2[1], s2[2], s1[3], s1[4])
        elseif iszero(s2[1])
            return s1
        else
            return s1[2] > s2[2] ? s1 : (s2[1], s2[2], s1[3], s1[4])
        end
    else
        return s1[1] + s2[1], s1[2] + s2[2], s1[3], s1[4]
    end
end

function eval_end(dist::RenyiDivergence, s::Tuple{T,T,T,T}) where {T <: Real}
    if dist.is_zero || dist.is_normal
        log(s[2] / s[1]) / dist.p + log(s[4] / s[3])
    elseif dist.is_one
        return s[2] / s[1] + log(s[4] / s[3])
    else # q = ∞
        log(s[2]) + log(s[4] / s[3])
    end
end

renyi_divergence(a, b, q::Real) = RenyiDivergence(q)(a, b)
# Combine docs with RenyiDivergence. Fetching the docstring with @doc causes
# problems during package compilation; see
# https://github.com/JuliaLang/julia/issues/31640
let docstring = Base.Docs.getdoc(RenyiDivergence)
    @doc docstring renyi_divergence
end

# JSDivergence

@inline function eval_op(::JSDivergence, ai::T, bi::T) where T
    u = (ai + bi) / 2
    ta = iszero(ai) ? zero(log(one(T))) : ai * log(ai) / 2
    tb = iszero(bi) ? zero(log(one(T))) : bi * log(bi) / 2
    tu = iszero(u) ? zero(log(one(T))) : u * log(u)
    ta + tb - tu
end
const js_divergence = JSDivergence()

# SpanNormDist

result_type(dist::SpanNormDist, ::Type{Ta}, ::Type{Tb}) where {Ta,Tb} =
    typeof(eval_op(dist, oneunit(Ta), oneunit(Tb)))
Base.@propagate_inbounds function eval_start(::SpanNormDist, a, b)
    d = first(a) - first(b)
    return d, d
end
eval_op(::SpanNormDist, ai, bi)  = ai - bi
@inline function eval_reduce(::SpanNormDist, s1, s2)
    min_d, max_d = s1
    if s2 > max_d
        max_d = s2
    elseif s2 < min_d
        min_d = s2
    end
    return min_d, max_d
end

eval_end(::SpanNormDist, s) = s[2] - s[1]
(::SpanNormDist)(a::Number, b::Number) = zero(promote_type(typeof(a), typeof(b)))
const spannorm_dist = SpanNormDist()

# Jaccard

eval_start(dist::Jaccard, a, b) =
    zero.(typeof.(eval_op(dist, oneunit(_eltype(a)), oneunit(_eltype(b)))))
@inline function eval_op(::Jaccard, s1, s2)
    abs_m = abs(s1 - s2)
    abs_p = abs(s1 + s2)
    abs_p - abs_m, abs_p + abs_m
end
@inline function eval_reduce(::Jaccard, s1, s2)
    @inbounds a = s1[1] + s2[1]
    @inbounds b = s1[2] + s2[2]
    a, b
end
@inline function eval_end(::Jaccard, a)
    @inbounds v = 1 - (a[1] / a[2])
    return v
end
const jaccard = Jaccard()

# BrayCurtis

eval_start(dist::BrayCurtis, a, b) =
    zero.(typeof.(eval_op(dist, oneunit(_eltype(a)), oneunit(_eltype(b)))))
@inline function eval_op(::BrayCurtis, s1, s2)
    abs_m = abs(s1 - s2)
    abs_p = abs(s1 + s2)
    abs_m, abs_p
end
@inline function eval_reduce(::BrayCurtis, s1, s2)
    @inbounds a = s1[1] + s2[1]
    @inbounds b = s1[2] + s2[2]
    a, b
end
@inline function eval_end(::BrayCurtis, a)
    @inbounds v = a[1] / a[2]
    return v
end
const braycurtis = BrayCurtis()

# Tanimoto

@inline eval_start(::RogersTanimoto, _, _) = 0, 0, 0, 0
@inline function eval_op(::RogersTanimoto, s1, s2)
    tt = s1 && s2
    tf = s1 && !s2
    ft = !s1 && s2
    ff = !s1 && !s2
    tt, tf, ft, ff
end
@inline function eval_reduce(::RogersTanimoto, s1, s2)
    @inbounds begin
        a = s1[1] + s2[1]
        b = s1[2] + s2[2]
        c = s1[3] + s2[3]
        d = s1[4] + s2[4]
    end
    a, b, c, d
end
@inline function eval_end(::RogersTanimoto, a)
    @inbounds numerator = 2(a[2] + a[3])
    @inbounds denominator = a[1] + a[4] + 2(a[2] + a[3])
    numerator / denominator
end
const rogerstanimoto = RogersTanimoto()

# Deviations

(::MeanAbsDeviation)(a, b) = cityblock(a, b) / length(a)
const meanad = MeanAbsDeviation()

(::MeanSqDeviation)(a, b) = sqeuclidean(a, b) / length(a)
const msd = MeanSqDeviation()

(::RMSDeviation)(a, b) = sqrt(MeanSqDeviation()(a, b))
const rmsd = RMSDeviation()

function (::NormRMSDeviation)(a, b)
    amin, amax = extrema(a)
    return RMSDeviation()(a, b) / (amax - amin)
end
const nrmsd = NormRMSDeviation()


###########################################################
#
#  Special method
#
###########################################################

# SqEuclidean/Euclidean
function _pairwise!(dist::Union{SqEuclidean,Euclidean}, r::AbstractMatrix,
                    a::AbstractMatrix, b::AbstractMatrix)
    require_one_based_indexing(r, a, b)
    m, na, nb = get_pairwise_dims(r, a, b)
    # the following checks if a'*b can be stored in r directly, it fails for complex eltypes
    inplace = promote_type(eltype(r), typeof(oneunit(eltype(a))'oneunit(eltype(b)))) === eltype(r)
    R = inplace ? mul!(r, a', b) : a'b
    sa2 = sum(abs2, a, dims=1)
    sb2 = sum(abs2, b, dims=1)
    z² = zero(real(eltype(R)))
    @inbounds if dist.thresh <= 0
        # If there's no chance of triggering the threshold, we can use @simd
        for j = 1:nb
            sb = sb2[j]
            @simd for i = 1:na
                r[i, j] = eval_end(dist, (max(sa2[i] + sb - 2real(R[i, j]), z²)))
            end
        end
    else
        for j = 1:nb
            sb = sb2[j]
            for i = 1:na
                selfterms = sa2[i] + sb
                v = max(selfterms - 2real(R[i, j]), z²)
                if v < dist.thresh * selfterms
                    # The distance is likely to be inaccurate, recalculate directly
                    # This reflects the following:
                    #   while sqrt(x+ϵ) ≈ sqrt(x) + O(ϵ/sqrt(x)) when |x| >> ϵ,
                    #         sqrt(x+ϵ) ≈ O(sqrt(ϵ))             otherwise.
                    v = zero(v)
                    for k = 1:m
                        v += abs2(a[k, i] - b[k, j])
                    end
                end
                r[i, j] = eval_end(dist, v)
            end
        end
    end
    r
end

function _pairwise!(dist::Union{SqEuclidean,Euclidean}, r::AbstractMatrix, a::AbstractMatrix)
    require_one_based_indexing(r, a)
    m, n = get_pairwise_dims(r, a)
    inplace = promote_type(eltype(r), typeof(oneunit(eltype(a))'oneunit(eltype(a)))) === eltype(r)
    # the following checks if a'*b can be stored in r directly, it fails for complex eltypes
    R = inplace ? mul!(r, a', a) : a'a
    sa2 = sum(abs2, a, dims=1)
    safe = dist.thresh <= 0
    z² = zero(real(eltype(R)))
    @inbounds for j = 1:n
        for i = 1:(j - 1)
            r[i, j] = r[j, i]
        end
        r[j, j] = zero(eltype(r))
        sa2j = sa2[j]
        if safe
            @simd for i = (j + 1):n
                r[i, j] = eval_end(dist, (max(sa2[i] + sa2j - 2real(R[i, j]), z²)))
            end
        else
            for i = (j + 1):n
                selfterms = sa2[i] + sa2j
                v = max(selfterms - 2real(R[i, j]), z²)
                if v < dist.thresh * selfterms
                    v = zero(v)
                    for k = 1:m
                        v += abs2(a[k, i] - a[k, j])
                    end
                end
                r[i, j] = eval_end(dist, v)
            end
        end
    end
    r
end

# Weighted SqEuclidean/Euclidean
function _pairwise!(dist::Union{WeightedSqEuclidean,WeightedEuclidean}, r::AbstractMatrix,
                    a::AbstractMatrix, b::AbstractMatrix)
    require_one_based_indexing(r, a, b)
    w = dist.weights
    m, na, nb = get_pairwise_dims(length(w), r, a, b)

    sa2 = wsumsq_percol(w, a)
    sb2 = wsumsq_percol(w, b)
    # the following checks if a'*b can be stored in r directly, it fails for complex eltypes
    inplace = promote_type(eltype(r), typeof(oneunit(eltype(a))'oneunit(eltype(b)))) === eltype(r)
    R = inplace ? mul!(r, a', w .* b) : a'*Diagonal(w)*b
    z² = zero(real(eltype(R)))
    for j = 1:nb
        @simd for i = 1:na
            @inbounds r[i, j] = eval_end(dist, max(sa2[i] + sb2[j] - 2real(R[i, j]), z²))
        end
    end
    r
end
function _pairwise!(dist::Union{WeightedSqEuclidean,WeightedEuclidean}, r::AbstractMatrix,
                    a::AbstractMatrix)
    require_one_based_indexing(r, a)
    w = dist.weights
    m, n = get_pairwise_dims(length(w), r, a)

    sa2 = wsumsq_percol(w, a)
    # the following checks if a'*b can be stored in r directly, it fails for complex eltypes
    inplace = promote_type(eltype(r), typeof(oneunit(eltype(a))'oneunit(eltype(a)))) === eltype(r)
    R = inplace ? mul!(r, a', w .* a) : a'*Diagonal(w)*a
    z² = zero(real(eltype(R)))

    @inbounds for j = 1:n
        for i = 1:(j - 1)
            r[i, j] = r[j, i]
        end
        r[j, j] = zero(eltype(r))
        @simd for i = (j + 1):n
            r[i, j] = eval_end(dist, max(sa2[i] + sa2[j] - 2real(R[i, j]), z²))
        end
    end
    r
end

# MeanSqDeviation, RMSDeviation, NormRMSDeviation
function _pairwise!(dist::MeanSqDeviation, r::AbstractMatrix, a::AbstractMatrix, b::AbstractMatrix)
    _pairwise!(SqEuclidean(), r, a, b)
    # TODO: Replace by rdiv!(r, size(a, 1)) once julia compat ≥v1.2
    s = size(a, 1)
    @simd for I in eachindex(r)
        @inbounds r[I] /= s
    end
    return r
end
_pairwise!(dist::RMSDeviation, r::AbstractMatrix, a::AbstractMatrix, b::AbstractMatrix) =
    sqrt!(_pairwise!(MeanSqDeviation(), r, a, b))
function _pairwise!(dist::NormRMSDeviation, r::AbstractMatrix, a::AbstractMatrix, b::AbstractMatrix)
    _pairwise!(RMSDeviation(), r, a, b)
    @views for (i, j) in zip(axes(r, 1), axes(a, 2))
        amin, amax = extrema(a[:,j])
        r[i,:] ./= amax - amin
    end
    return r
end

function _pairwise!(dist::MeanSqDeviation, r::AbstractMatrix, a::AbstractMatrix)
    _pairwise!(SqEuclidean(), r, a)
    # TODO: Replace by rdiv!(r, size(a, 1)) once julia compat ≥v1.2
    s = size(a, 1)
    @simd for I in eachindex(r)
        @inbounds r[I] /= s
    end
    return r
end
_pairwise!(dist::RMSDeviation, r::AbstractMatrix, a::AbstractMatrix) =
    sqrt!(_pairwise!(MeanSqDeviation(), r, a))
function _pairwise!(dist::NormRMSDeviation, r::AbstractMatrix, a::AbstractMatrix)
    _pairwise!(RMSDeviation(), r, a)
    @views for (i, j) in zip(axes(r, 1), axes(a, 2))
        amin, amax = extrema(a[:,j])
        r[i,:] ./= amax - amin
    end
    return r
end

# CosineDist

function _pairwise!(::CosineDist, r::AbstractMatrix, a::AbstractMatrix, b::AbstractMatrix)
    require_one_based_indexing(r, a, b)
    m, na, nb = get_pairwise_dims(r, a, b)
    inplace = promote_type(eltype(r), typeof(oneunit(eltype(a))'oneunit(eltype(b)))) === eltype(r)
    R = inplace ? mul!(r, a', b) : a'b
    ra = norm_percol(a)
    rb = norm_percol(b)
    for j = 1:nb
        @simd for i = 1:na
            @inbounds r[i, j] = max(1 - R[i, j] / (ra[i] * rb[j]), 0)
        end
    end
    r
end
function _pairwise!(::CosineDist, r::AbstractMatrix, a::AbstractMatrix)
    require_one_based_indexing(r, a)
    m, n = get_pairwise_dims(r, a)
    inplace = promote_type(eltype(r), typeof(oneunit(eltype(a))'oneunit(eltype(a)))) === eltype(r)
    R = inplace ? mul!(r, a', a) : a'a
    ra = norm_percol(a)
    @inbounds for j = 1:n
        for i = 1:(j - 1)
            r[i, j] = r[j, i]
        end
        r[j, j] = zero(eltype(r))
        @simd for i = j + 1:n
            r[i, j] = max(1 - R[i, j] / (ra[i] * ra[j]), 0)
        end
    end
    r
end

# CorrDist
# This part of codes is accelerated because:
# 1. It calls the accelerated `_pairwise` specilization for CosineDist
# 2. pre-calculated `_centralize_colwise` avoids four times of redundant computations
#    of `_centralize` -- ~4x speed up
_centralize_colwise(x::AbstractMatrix) = x .- mean(x, dims=1)
_pairwise!(::CorrDist, r::AbstractMatrix, a::AbstractMatrix, b::AbstractMatrix) =
    _pairwise!(CosineDist(), r, _centralize_colwise(a), _centralize_colwise(b))
_pairwise!(::CorrDist, r::AbstractMatrix, a::AbstractMatrix) =
    _pairwise!(CosineDist(), r, _centralize_colwise(a))
