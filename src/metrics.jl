# Ordinary metrics

###########################################################
#
#   Metric types
#
###########################################################
const RealAbstractArray{T <: Real} =  AbstractArray{T}

struct Euclidean <: Metric
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

struct WeightedEuclidean{W <: RealAbstractArray} <: Metric
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
struct PeriodicEuclidean{W <: AbstractArray{<: Real}} <: Metric
    periods::W
end

struct SqEuclidean <: SemiMetric
    thresh::Float64
end

"""
    SqEuclidean([thresh])

Create a squared-euclidean semi-metric. For the meaning of `thresh`,
see [`Euclidean`](@ref).
"""
SqEuclidean() = SqEuclidean(0)

struct WeightedSqEuclidean{W <: RealAbstractArray} <: SemiMetric
    weights::W
end

struct Chebyshev <: Metric end

struct Cityblock <: Metric end
struct WeightedCityblock{W <: RealAbstractArray} <: Metric
    weights::W
end

struct TotalVariation <: Metric end
struct Jaccard <: Metric end
struct RogersTanimoto <: Metric end

struct Minkowski{T <: Real} <: Metric
    p::T
end
struct WeightedMinkowski{W <: RealAbstractArray,T <: Real} <: Metric
    weights::W
    p::T
end

struct Hamming <: Metric end
struct WeightedHamming{W <: RealAbstractArray} <: Metric
    weights::W
end

struct CosineDist <: SemiMetric end
# CorrDist is excluded from `UnionMetrics`
struct CorrDist <: SemiMetric end
struct BrayCurtis <: SemiMetric end

struct ChiSqDist <: SemiMetric end
struct KLDivergence <: PreMetric end
struct GenKLDivergence <: PreMetric end

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
struct RenyiDivergence{T <: Real} <: PreMetric
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

struct JSDivergence <: SemiMetric end

struct SpanNormDist <: SemiMetric end

# Deviations are handled separately from the other distances/divergences and
# are excluded from `UnionMetrics`
struct MeanAbsDeviation <: Metric end
struct MeanSqDeviation <: SemiMetric end
struct RMSDeviation <: Metric end
struct NormRMSDeviation <: PreMetric end

# Union types
const metrics = (Euclidean,SqEuclidean,PeriodicEuclidean,Chebyshev,Cityblock,TotalVariation,Minkowski,Hamming,Jaccard,RogersTanimoto,CosineDist,ChiSqDist,KLDivergence,RenyiDivergence,BrayCurtis,JSDivergence,SpanNormDist,GenKLDivergence)
const weightedmetrics = (WeightedEuclidean,WeightedSqEuclidean,WeightedCityblock,WeightedMinkowski,WeightedHamming)
const UnionWeightedMetrics{W} = Union{map(M->M{W}, weightedmetrics)...}
const UnionMetrics = Union{metrics...}

###########################################################
#
#  Implementations
#
###########################################################

const ArraySlice{T} = SubArray{T,1,Array{T,2},Tuple{Base.Slice{Base.OneTo{Int}},Int},true}

@inline parameters(::UnionMetrics) = nothing

Base.eltype(x::UnionWeightedMetrics) = eltype(x.weights)

# breaks the implementation into eval_start, eval_op, eval_reduce and eval_end

# Specialized for Arrays and avoids a branch on the size
@inline Base.@propagate_inbounds function _evaluate(d::UnionMetrics, a::Union{Array, ArraySlice}, b::Union{Array, ArraySlice})
    @boundscheck if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end
    p = parameters(d)
    @boundscheck if p !== nothing
        length(a) != length(p) && throw(DimensionMismatch("arrays have length $(length(a)) but parameters have length $(length(p))."))
    end
    if length(a) == 0
        return zero(result_type(d, a, b))
    end
    @inbounds begin
        s = eval_start(d, a, b)
        if p === nothing
            @simd for I in 1:length(a)
                ai = a[I]
                bi = b[I]
                s = eval_reduce(d, s, eval_op(d, ai, bi))
            end
        else
            @simd for I in 1:length(a)
                aI = a[I]
                bI = b[I]
                pI = p[I]
                s = eval_reduce(d, s, eval_op(d, aI, bI, pI))
            end
        end
        return eval_end(d, s)
    end
end

@inline function _evaluate(d::UnionMetrics, a::AbstractArray, b::AbstractArray)
    @boundscheck if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end
    p = parameters(d)
    @boundscheck if p !== nothing
        length(a) != length(p) && throw(DimensionMismatch("arrays have length $(length(a)) but parameters have length $(length(p))."))
    end
    if length(a) == 0
        return zero(result_type(d, a, b))
    end
    @inbounds begin
        s = eval_start(d, a, b)
        if size(a) == size(b)
            if p === nothing
                @simd for I in eachindex(a, b)
                    ai = a[I]
                    bi = b[I]
                    s = eval_reduce(d, s, eval_op(d, ai, bi))
                end
            else
                @simd for I in eachindex(a, b, p)
                    aI = a[I]
                    bI = b[I]
                    pI = p[I]
                    s = eval_reduce(d, s, eval_op(d, aI, bI, pI))
                end
            end
        else
            if p === nothing
                for (Ia, Ib) in zip(eachindex(a), eachindex(b))
                    ai = a[Ia]
                    bi = b[Ib]
                    s = eval_reduce(d, s, eval_op(d, ai, bi))
                end
            else
                for (Ia, Ib, Ip) in zip(eachindex(a), eachindex(b), eachindex(p))
                    aI = a[Ia]
                    bI = b[Ib]
                    pI = p[Ip]
                    s = eval_reduce(d, s, eval_op(d, aI, bI, pI))
                end
            end
        end
    end
    return eval_end(d, s)
end

@inline function _evaluate(d::UnionWeightedMetrics, a::AbstractArray, b::AbstractArray)
    @boundscheck if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end
    @boundscheck if length(a) != length(d.weights)
        throw(DimensionMismatch("arrays have length $(length(a)) but weights have length $(length(d.weights))."))
    end
    if length(a) == 0
        return zero(result_type(d, a, b))
    end
    @inbounds begin
        s = eval_start(d, a, b)
        if size(a) == size(b)
            @simd for I in eachindex(a, b, d.weights)
                ai = a[I]
                bi = b[I]
                wi = d.weights[I]
                s = eval_reduce(d, s, eval_op(d, ai, bi, wi))
            end
        else
            for (Ia, Ib, Iw) in zip(eachindex(a), eachindex(b), eachindex(d.weights))
                ai = a[Ia]
                bi = b[Ib]
                wi = d.weights[Iw]
                s = eval_reduce(d, s, eval_op(d, ai, bi, wi))
            end
        end
    end
    return eval_end(d, s)
end

result_type(dist::UnionMetrics, Ta::Type, Tb::Type) =
    typeof(evaluate(dist, oneunit(Ta), oneunit(Tb)))
eval_start(d::UnionMetrics, a::AbstractArray, b::AbstractArray) =
    zero(result_type(d, a, b))
eval_end(d::UnionMetrics, s) = s
result_type(dist::UnionWeightedMetrics, Ta::Type, Tb::Type) =
    typeof(evaluate(dist, oneunit(Ta), oneunit(Tb)))
@inline function eval_start(d::UnionWeightedMetrics, a::AbstractArray, b::AbstractArray)
    zero(result_type(d, a, b))
end
eval_end(d::UnionWeightedMetrics, s) = s

for M in metrics
    @eval @inline (dist::$M)(a::AbstractArray, b::AbstractArray) = _evaluate(dist, a, b)
    @eval @inline (dist::$M)(a::Number, b::Number) = eval_end(dist, eval_op(dist, a, b))
end
for M in weightedmetrics
    @eval (dist::$M)(a::AbstractArray, b::AbstractArray) = _evaluate(dist, a, b)
    @eval (dist::$M)(a::Number, b::Number) = eval_end(dist, eval_op(dist, a, b, oneunit(eltype(dist))))
end

# SqEuclidean
@inline eval_op(::SqEuclidean, ai, bi) = abs2(ai - bi)
@inline eval_reduce(::SqEuclidean, s1, s2) = s1 + s2

sqeuclidean(a::AbstractArray, b::AbstractArray) = SqEuclidean()(a, b)
sqeuclidean(a::Number, b::Number) = SqEuclidean()(a, b)

# Weighted Squared Euclidean
@inline eval_op(::WeightedSqEuclidean, ai, bi, wi) = abs2(ai - bi) * wi
@inline eval_reduce(::WeightedSqEuclidean, s1, s2) = s1 + s2
wsqeuclidean(a::AbstractArray, b::AbstractArray, w::AbstractArray) = WeightedSqEuclidean(w)(a, b)

# Euclidean
@inline eval_op(::Euclidean, ai, bi) = abs2(ai - bi)
@inline eval_reduce(::Euclidean, s1, s2) = s1 + s2
eval_end(::Euclidean, s) = sqrt(s)
euclidean(a::AbstractArray, b::AbstractArray) = Euclidean()(a, b)
euclidean(a::Number, b::Number) = Euclidean()(a, b)

# Weighted Euclidean
@inline eval_op(::WeightedEuclidean, ai, bi, wi) = abs2(ai - bi) * wi
@inline eval_reduce(::WeightedEuclidean, s1, s2) = s1 + s2
@inline eval_end(::WeightedEuclidean, s) = sqrt(s)
weuclidean(a::AbstractArray, b::AbstractArray, w::AbstractArray) = WeightedEuclidean(w)(a, b)

# PeriodicEuclidean
Base.eltype(d::PeriodicEuclidean) = eltype(d.periods)
@inline parameters(d::PeriodicEuclidean) = d.periods
@inline function eval_op(d::PeriodicEuclidean, ai, bi, p)
    s1 = abs(ai - bi)
    s2 = mod(s1, p)
    s3 = min(s2, p - s2)
    abs2(s3)
end
@inline function eval_op(d::PeriodicEuclidean, ai, bi)
    periods = d.periods
    p = isempty(periods) ? oneunit(eltype(periods)) : first(periods)
    eval_op(d, ai, bi, p)
end
@inline eval_reduce(::PeriodicEuclidean, s1, s2) = s1 + s2
@inline eval_end(::PeriodicEuclidean, s) = sqrt(s)
peuclidean(a::AbstractArray, b::AbstractArray, p::AbstractArray{<: Real}) =
    PeriodicEuclidean(p)(a, b)
peuclidean(a::Number, b::Number, p::Real) = PeriodicEuclidean([p])(a, b)

# Cityblock
@inline eval_op(::Cityblock, ai, bi) = abs(ai - bi)
@inline eval_reduce(::Cityblock, s1, s2) = s1 + s2
cityblock(a::AbstractArray, b::AbstractArray) = Cityblock()(a, b)
cityblock(a::Number, b::Number) = Cityblock()(a, b)

# City Block
@inline eval_op(::WeightedCityblock, ai, bi, wi) = abs((ai - bi) * wi)
@inline eval_reduce(::WeightedCityblock, s1, s2) = s1 + s2
wcityblock(a::AbstractArray, b::AbstractArray, w::AbstractArray) = WeightedCityblock(w)(a, b)

# Total variation
@inline eval_op(::TotalVariation, ai, bi) = abs(ai - bi)
@inline eval_reduce(::TotalVariation, s1, s2) = s1 + s2
eval_end(::TotalVariation, s) = s / 2
totalvariation(a::AbstractArray, b::AbstractArray) = TotalVariation()(a, b)
totalvariation(a::Number, b::Number) = TotalVariation()(a, b)

# Chebyshev
@inline eval_op(::Chebyshev, ai, bi) = abs(ai - bi)
@inline eval_reduce(::Chebyshev, s1, s2) = max(s1, s2)
# if only NaN, will output NaN
@inline Base.@propagate_inbounds eval_start(::Chebyshev, a::AbstractArray, b::AbstractArray) = abs(a[1] - b[1])
chebyshev(a::AbstractArray, b::AbstractArray) = Chebyshev()(a, b)
chebyshev(a::Number, b::Number) = Chebyshev()(a, b)

# Minkowski
@inline eval_op(dist::Minkowski, ai, bi) = abs(ai - bi).^dist.p
@inline eval_reduce(::Minkowski, s1, s2) = s1 + s2
eval_end(dist::Minkowski, s) = s.^(1 / dist.p)
minkowski(a::AbstractArray, b::AbstractArray, p::Real) = Minkowski(p)(a, b)
minkowski(a::Number, b::Number, p::Real) = Minkowski(p)(a, b)

# Weighted Minkowski
@inline eval_op(dist::WeightedMinkowski, ai, bi, wi) = abs(ai - bi).^dist.p * wi
@inline eval_reduce(::WeightedMinkowski, s1, s2) = s1 + s2
eval_end(dist::WeightedMinkowski, s) = s.^(1 / dist.p)
wminkowski(a::AbstractArray, b::AbstractArray, w::AbstractArray, p::Real) = WeightedMinkowski(w, p)(a, b)

# Hamming
@inline eval_op(::Hamming, ai, bi) = ai != bi ? 1 : 0
@inline eval_reduce(::Hamming, s1, s2) = s1 + s2
hamming(a::AbstractArray, b::AbstractArray) = Hamming()(a, b)
hamming(a::Number, b::Number) = Hamming()(a, b)

# WeightedHamming
@inline eval_op(::WeightedHamming, ai, bi, wi) = ai != bi ? wi : zero(eltype(wi))
@inline eval_reduce(::WeightedHamming, s1, s2) = s1 + s2
whamming(a::AbstractArray, b::AbstractArray, w::AbstractArray) = WeightedHamming(w)(a, b)

# Cosine dist
@inline function eval_start(dist::CosineDist, a::AbstractArray, b::AbstractArray)
    T = Base.promote_typeof(eval_op(dist, oneunit(eltype(a)), oneunit(eltype(b)))...)
    zero(T), zero(T), zero(T)
end
@inline eval_op(::CosineDist, ai, bi) = ai * bi, ai * ai, bi * bi
@inline function eval_reduce(::CosineDist, s1, s2)
    a1, b1, c1 = s1
    a2, b2, c2 = s2
    return a1 + a2, b1 + b2, c1 + c2
end
function eval_end(::CosineDist, s)
    ab, a2, b2 = s
    max(1 - ab / (sqrt(a2) * sqrt(b2)), zero(eltype(ab)))
end
cosine_dist(a::AbstractArray, b::AbstractArray) = CosineDist()(a, b)

# Correlation Dist
_centralize(x::AbstractArray) = x .- mean(x)
(dist::CorrDist)(a::AbstractArray, b::AbstractArray) = CosineDist()(_centralize(a), _centralize(b))
corr_dist(a::AbstractArray, b::AbstractArray) = CorrDist()(a, b)
result_type(::CorrDist, Ta::Type, Tb::Type) = result_type(CosineDist(), Ta, Tb)

# ChiSqDist
@inline eval_op(::ChiSqDist, ai, bi) = (d = abs2(ai - bi) / (ai + bi); ifelse(ai != bi, d, zero(d)))
@inline eval_reduce(::ChiSqDist, s1, s2) = s1 + s2
chisq_dist(a::AbstractArray, b::AbstractArray) = ChiSqDist()(a, b)

# KLDivergence
@inline eval_op(dist::KLDivergence, ai, bi) =
    ai > 0 ? ai * log(ai / bi) : zero(eval_op(dist, oneunit(ai), bi))
@inline eval_reduce(::KLDivergence, s1, s2) = s1 + s2
kl_divergence(a::AbstractArray, b::AbstractArray) = KLDivergence()(a, b)

# GenKLDivergence
@inline eval_op(dist::GenKLDivergence, ai, bi) =
    ai > 0 ? ai * log(ai / bi) - ai + bi : oftype(eval_op(dist, oneunit(ai), bi), bi)
@inline eval_reduce(::GenKLDivergence, s1, s2) = s1 + s2
gkl_divergence(a::AbstractArray, b::AbstractArray) = GenKLDivergence()(a, b)

# RenyiDivergence
@inline Base.@propagate_inbounds function eval_start(::RenyiDivergence, a::AbstractArray{T}, b::AbstractArray{T}) where {T <: Real}
    zero(T), zero(T), T(sum(a)), T(sum(b))
end

@inline function eval_op(dist::RenyiDivergence, ai::T, bi::T) where {T <: Real}
    if ai == zero(T)
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
                                               s2::Tuple{T,T,T,T}) where {T <: Real}
    if dist.is_inf
        if s1[1] == zero(T)
            return (s2[1], s2[2], s1[3], s1[4])
        elseif s2[1] == zero(T)
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

renyi_divergence(a::AbstractArray, b::AbstractArray, q::Real) = RenyiDivergence(q)(a, b)
# Combine docs with RenyiDivergence. Fetching the docstring with @doc causes
# problems during package compilation; see
# https://github.com/JuliaLang/julia/issues/31640
let docstring = Base.Docs.getdoc(RenyiDivergence)
    @doc docstring renyi_divergence
end

# JSDivergence
@inline function eval_op(::JSDivergence, ai::T, bi::T) where {T}
    u = (ai + bi) / 2
    ta = ai > 0 ? ai * log(ai) / 2 : zero(log(one(T)))
    tb = bi > 0 ? bi * log(bi) / 2 : zero(log(one(T)))
    tu = u > 0 ? u * log(u) : zero(log(one(T)))
    ta + tb - tu
end
@inline eval_reduce(::JSDivergence, s1, s2) = s1 + s2
js_divergence(a::AbstractArray, b::AbstractArray) = JSDivergence()(a, b)

# SpanNormDist
@inline Base.@propagate_inbounds function eval_start(::SpanNormDist, a::AbstractArray, b::AbstractArray)
    a[1] - b[1], a[1] - b[1]
end
@inline eval_op(::SpanNormDist, ai, bi)  = ai - bi
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
spannorm_dist(a::AbstractArray, b::AbstractArray) = SpanNormDist()(a, b)
result_type(dist::SpanNormDist, Ta::Type, Tb::Type) =
    typeof(eval_op(dist, oneunit(Ta), oneunit(Tb)))

# Jaccard

@inline eval_start(::Jaccard, a::AbstractArray{Bool}, b::AbstractArray{Bool}) = 0, 0
@inline function eval_start(dist::Jaccard, a::AbstractArray, b::AbstractArray)
    T = Base.promote_typeof(eval_op(dist, oneunit(eltype(a)), oneunit(eltype(b)))...)
    zero(T), zero(T)
end
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
jaccard(a::AbstractArray, b::AbstractArray) = Jaccard()(a, b)

# BrayCurtis

@inline eval_start(::BrayCurtis, a::AbstractArray{Bool}, b::AbstractArray{Bool}) = 0, 0
@inline function eval_start(dist::BrayCurtis, a::AbstractArray, b::AbstractArray)
    T = Base.promote_typeof(eval_op(dist, oneunit(eltype(a)), oneunit(eltype(b)))...)
    zero(T), zero(T)
end
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
braycurtis(a::AbstractArray, b::AbstractArray) = BrayCurtis()(a, b)


# Tanimoto

@inline eval_start(::RogersTanimoto, a::AbstractArray, b::AbstractArray) = 0, 0, 0, 0
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
        d = s1[4] + s1[4]
    end
    a, b, c, d
end
@inline function eval_end(::RogersTanimoto, a)
    @inbounds numerator = 2(a[2] + a[3])
    @inbounds denominator = a[1] + a[4] + 2(a[2] + a[3])
    numerator / denominator
end
rogerstanimoto(a::AbstractArray{T}, b::AbstractArray{T}) where {T <: Bool} = RogersTanimoto()(a, b)

# Deviations

(dist::MeanAbsDeviation)(a, b) = cityblock(a, b) / length(a)
meanad(a, b) = MeanAbsDeviation()(a, b)

(dist::MeanSqDeviation)(a, b) = sqeuclidean(a, b) / length(a)
msd(a, b) = MeanSqDeviation()(a, b)

(dist::RMSDeviation)(a, b) = sqrt(MeanSqDeviation()(a, b))
rmsd(a, b) = RMSDeviation()(a, b)

function (dist::NormRMSDeviation)(a, b)
    amin, amax = extrema(a)
    return RMSDeviation()(a, b) / (amax - amin)
end
nrmsd(a, b) = NormRMSDeviation()(a, b)


###########################################################
#
#  Special method
#
###########################################################

# SqEuclidean
function _pairwise!(r::AbstractMatrix, dist::SqEuclidean,
                    a::AbstractMatrix, b::AbstractMatrix)
    mul!(r, a', b)
    sa2 = sum(abs2, a, dims=1)
    sb2 = sum(abs2, b, dims=1)
    threshT = convert(eltype(r), dist.thresh)
    if threshT <= 0
        # If there's no chance of triggering the threshold, we can use @simd
        for j = 1:size(r, 2)
            sb = sb2[j]
            @simd for i = 1:size(r, 1)
                @inbounds r[i, j] = sa2[i] + sb - 2 * r[i, j]
            end
        end
    else
        for j = 1:size(r, 2)
            sb = sb2[j]
            for i = 1:size(r, 1)
                @inbounds selfterms = sa2[i] + sb
                @inbounds v = selfterms - 2 * r[i, j]
                if v < threshT * selfterms
                    # The distance is likely to be inaccurate, recalculate at higher prec.
                    # This reflects the following:
                    #   ((x+ϵ) - y)^2 ≈ x^2 - 2xy + y^2 + O(ϵ)    when |x-y| >> ϵ
                    #   ((x+ϵ) - y)^2 ≈ O(ϵ^2)                    otherwise
                    v = zero(v)
                    for k = 1:size(a, 1)
                        @inbounds v += (a[k, i] - b[k, j])^2
                    end
                end
                @inbounds r[i, j] = v
            end
        end
    end
    r
end

function _pairwise!(r::AbstractMatrix, dist::SqEuclidean, a::AbstractMatrix)
    m, n = get_pairwise_dims(r, a)
    mul!(r, a', a)
    sa2 = sumsq_percol(a)
    threshT = convert(eltype(r), dist.thresh)
    @inbounds for j = 1:n
        for i = 1:(j - 1)
            r[i, j] = r[j, i]
        end
        r[j, j] = 0
        sa2j = sa2[j]
        if threshT <= 0
            @simd for i = (j + 1):n
                r[i, j] = sa2[i] + sa2j - 2 * r[i, j]
            end
        else
            for i = (j + 1):n
                selfterms = sa2[i] + sa2j
                v = selfterms - 2 * r[i, j]
                if v < threshT * selfterms
                    v = zero(v)
                    for k = 1:size(a, 1)
                        v += (a[k, i] - a[k, j])^2
                    end
                end
                r[i, j] = v
            end
        end
    end
    r
end

# Weighted SqEuclidean
function _pairwise!(r::AbstractMatrix, dist::WeightedSqEuclidean,
                    a::AbstractMatrix, b::AbstractMatrix)
    w = dist.weights
    m, na, nb = get_pairwise_dims(length(w), r, a, b)

    sa2 = wsumsq_percol(w, a)
    sb2 = wsumsq_percol(w, b)
    mul!(r, a', b .* w)
    for j = 1:nb
        @simd for i = 1:na
            @inbounds r[i, j] = sa2[i] + sb2[j] - 2 * r[i, j]
        end
    end
    r
end
function _pairwise!(r::AbstractMatrix, dist::WeightedSqEuclidean,
                    a::AbstractMatrix)
    w = dist.weights
    m, n = get_pairwise_dims(length(w), r, a)

    sa2 = wsumsq_percol(w, a)
    mul!(r, a', a .* w)

    for j = 1:n
        for i = 1:(j - 1)
            @inbounds r[i, j] = r[j, i]
        end
        @inbounds r[j, j] = 0
        @simd for i = (j + 1):n
            @inbounds r[i, j] = sa2[i] + sa2[j] - 2 * r[i, j]
        end
    end
    r
end

# Euclidean
function _pairwise!(r::AbstractMatrix, dist::Euclidean,
                    a::AbstractMatrix, b::AbstractMatrix)
    m, na, nb = get_pairwise_dims(r, a, b)
    mul!(r, a', b)
    sa2 = sumsq_percol(a)
    sb2 = sumsq_percol(b)
    threshT = convert(eltype(r), dist.thresh)
    @inbounds for j = 1:nb
        sb = sb2[j]
        for i = 1:na
            selfterms = sa2[i] + sb
            v = selfterms - 2 * r[i, j]
            if v < threshT * selfterms
                # The distance is likely to be inaccurate, recalculate directly
                # This reflects the following:
                #   while sqrt(x+ϵ) ≈ sqrt(x) + O(ϵ/sqrt(x)) when |x| >> ϵ,
                #         sqrt(x+ϵ) ≈ O(sqrt(ϵ))             otherwise.
                v = zero(v)
                for k = 1:m
                    v += (a[k, i] - b[k, j])^2
                end
            end
            r[i, j] = sqrt(v)
        end
    end
    r
end

function _pairwise!(r::AbstractMatrix, dist::Euclidean, a::AbstractMatrix)
    m, n = get_pairwise_dims(r, a)
    mul!(r, a', a)
    sa2 = sumsq_percol(a)
    threshT = convert(eltype(r), dist.thresh)
    @inbounds for j = 1:n
        for i = 1:(j - 1)
            r[i, j] = r[j, i]
        end
        r[j, j] = 0
        sa2j = sa2[j]
        for i = (j + 1):n
            selfterms = sa2[i] + sa2j
            v = selfterms - 2 * r[i, j]
            if v < threshT * selfterms
                v = zero(v)
                for k = 1:m
                    v += (a[k, i] - a[k, j])^2
                end
            end
            r[i, j] = sqrt(v)
        end
    end
    r
end

# Weighted Euclidean
function colwise!(r::AbstractArray, dist::WeightedEuclidean, a::AbstractMatrix, b::AbstractMatrix)
    sqrt!(colwise!(r, WeightedSqEuclidean(dist.weights), a, b))
end
function colwise!(r::AbstractArray, dist::WeightedEuclidean, a::AbstractVector, b::AbstractMatrix)
    sqrt!(colwise!(r, WeightedSqEuclidean(dist.weights), a, b))
end
function _pairwise!(r::AbstractMatrix, dist::WeightedEuclidean,
                    a::AbstractMatrix, b::AbstractMatrix)
    sqrt!(_pairwise!(r, WeightedSqEuclidean(dist.weights), a, b))
end
function _pairwise!(r::AbstractMatrix, dist::WeightedEuclidean, a::AbstractMatrix)
    sqrt!(_pairwise!(r, WeightedSqEuclidean(dist.weights), a))
end

# CosineDist

function _pairwise!(r::AbstractMatrix, dist::CosineDist,
                    a::AbstractMatrix, b::AbstractMatrix)
    m, na, nb = get_pairwise_dims(r, a, b)
    mul!(r, a', b)
    ra = sqrt!(sumsq_percol(a))
    rb = sqrt!(sumsq_percol(b))
    for j = 1:nb
        @simd for i = 1:na
            @inbounds r[i, j] = max(1 - r[i, j] / (ra[i] * rb[j]), 0)
        end
    end
    r
end
function _pairwise!(r::AbstractMatrix, dist::CosineDist, a::AbstractMatrix)
    m, n = get_pairwise_dims(r, a)
    mul!(r, a', a)
    ra = sqrt!(sumsq_percol(a))
    @inbounds for j = 1:n
        @simd for i = j + 1:n
            r[i, j] = max(1 - r[i, j] / (ra[i] * ra[j]), 0)
        end
        r[j, j] = 0
        for i = 1:(j - 1)
            r[i, j] = r[j, i]
        end
    end
    r
end

# CorrDist
# This part of codes is accelerated because:
# 1. It calls the accelerated `_pairwise` specilization for CosineDist
# 2. pre-calculated `_centralize_colwise` avoids four times of redundant computations
#    of `_centralize` -- ~4x speed up
_centralize_colwise(x::AbstractVector) = x .- mean(x)
_centralize_colwise(x::AbstractMatrix) = x .- mean(x, dims=1)
function colwise!(r::AbstractVector, dist::CorrDist, a::AbstractMatrix, b::AbstractMatrix)
    colwise!(r, CosineDist(), _centralize_colwise(a), _centralize_colwise(b))
end
function colwise!(r::AbstractVector, dist::CorrDist, a::AbstractVector, b::AbstractMatrix)
    colwise!(r, CosineDist(), _centralize_colwise(a), _centralize_colwise(b))
end
function _pairwise!(r::AbstractMatrix, dist::CorrDist,
                    a::AbstractMatrix, b::AbstractMatrix)
    _pairwise!(r, CosineDist(), _centralize_colwise(a), _centralize_colwise(b))
end
function _pairwise!(r::AbstractMatrix, dist::CorrDist, a::AbstractMatrix)
    _pairwise!(r, CosineDist(), _centralize_colwise(a))
end
