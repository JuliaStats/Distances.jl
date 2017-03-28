# Ordinary metrics

###########################################################
#
#   Metric types
#
###########################################################

immutable Euclidean <: Metric
    thresh::Float64
end
immutable SqEuclidean <: SemiMetric
    thresh::Float64
end
type Chebyshev <: Metric end
type Cityblock <: Metric end
type Jaccard <: Metric end
type RogersTanimoto <: Metric end

immutable Minkowski{T <: Real} <: Metric
    p::T
end

type Hamming <: Metric end

type CosineDist <: SemiMetric end
type CorrDist <: SemiMetric end

type ChiSqDist <: SemiMetric end
type KLDivergence <: PreMetric end
type GenKLDivergence <: PreMetric end

immutable RenyiDivergence{T <: Real} <: PreMetric
    p::T # order of power mean (order of divergence - 1)
    is_normal::Bool
    is_zero::Bool
    is_one::Bool
    is_inf::Bool

    function (::Type{RenyiDivergence{T}}){T}(q)
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
RenyiDivergence{T}(q::T) = RenyiDivergence{T}(q)

type JSDivergence <: SemiMetric end

type SpanNormDist <: SemiMetric end

# Deviations are handled separately from the other distances/divergences and
# are excluded from `UnionMetrics`
type MeanAbsDeviation <: Metric end
type MeanSqDeviation <: SemiMetric end
type RMSDeviation <: Metric end
type NormRMSDeviation <: PreMetric end
type CVRMSDeviation <: PreMetric end


const UnionMetrics = Union{Euclidean, SqEuclidean, Chebyshev, Cityblock, Minkowski, Hamming, Jaccard, RogersTanimoto, CosineDist, CorrDist, ChiSqDist, KLDivergence, RenyiDivergence, JSDivergence, SpanNormDist, GenKLDivergence}

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

"""
    SqEuclidean([thresh])

Create a squared-euclidean semi-metric. For the meaning of `thresh`,
see [`Euclidean`](@ref).
"""
SqEuclidean() = SqEuclidean(0)

###########################################################
#
#  Define Evaluate
#
###########################################################

function evaluate(d::UnionMetrics, a::AbstractArray, b::AbstractArray)
    if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end
    if length(a) == 0
        return zero(result_type(d, a, b))
    end
    s = eval_start(d, a, b)
    if size(a) == size(b)
        @simd for I in eachindex(a, b)
            @inbounds ai = a[I]
            @inbounds bi = b[I]
            s = eval_reduce(d, s, eval_op(d, ai, bi))
        end
    else
        for (Ia, Ib) in zip(eachindex(a), eachindex(b))
            @inbounds ai = a[Ia]
            @inbounds bi = b[Ib]
            s = eval_reduce(d, s, eval_op(d, ai, bi))
        end
    end
    return eval_end(d, s)
end
result_type{T1, T2}(dist::UnionMetrics, ::AbstractArray{T1}, ::AbstractArray{T2}) =
    typeof(eval_end(dist, eval_op(dist, one(T1), one(T2))))
eval_start(d::UnionMetrics, a::AbstractArray, b::AbstractArray) =
    zero(result_type(d, a, b))
eval_end(d::UnionMetrics, s) = s

evaluate{T <: Number}(dist::UnionMetrics, a::T, b::T) = eval_end(dist, eval_op(dist, a, b))

# SqEuclidean
@inline eval_op(::SqEuclidean, ai, bi) = abs2(ai - bi)
@inline eval_reduce(::SqEuclidean, s1, s2) = s1 + s2

sqeuclidean(a::AbstractArray, b::AbstractArray) = evaluate(SqEuclidean(), a, b)
sqeuclidean{T <: Number}(a::T, b::T) = evaluate(SqEuclidean(), a, b)

# Euclidean
@inline eval_op(::Euclidean, ai, bi) = abs2(ai - bi)
@inline eval_reduce(::Euclidean, s1, s2) = s1 + s2
eval_end(::Euclidean, s) = sqrt(s)
euclidean(a::AbstractArray, b::AbstractArray) = evaluate(Euclidean(), a, b)
euclidean(a::Number, b::Number) = evaluate(Euclidean(), a, b)

# Cityblock
@inline eval_op(::Cityblock, ai, bi) = abs(ai - bi)
@inline eval_reduce(::Cityblock, s1, s2) = s1 + s2
cityblock(a::AbstractArray, b::AbstractArray) = evaluate(Cityblock(), a, b)
cityblock{T <: Number}(a::T, b::T) = evaluate(Cityblock(), a, b)

# Chebyshev
@inline eval_op(::Chebyshev, ai, bi) = abs(ai - bi)
@inline eval_reduce(::Chebyshev, s1, s2) = max(s1, s2)
# if only NaN, will output NaN
@inline eval_start(::Chebyshev, a::AbstractArray, b::AbstractArray) = abs(a[1] - b[1])
chebyshev(a::AbstractArray, b::AbstractArray) = evaluate(Chebyshev(), a, b)
chebyshev{T <: Number}(a::T, b::T) = evaluate(Chebyshev(), a, b)

# Minkowski
@inline eval_op(dist::Minkowski, ai, bi) = abs(ai - bi) .^ dist.p
@inline eval_reduce(::Minkowski, s1, s2) = s1 + s2
eval_end(dist::Minkowski, s) = s .^ (1/dist.p)
minkowski(a::AbstractArray, b::AbstractArray, p::Real) = evaluate(Minkowski(p), a, b)
minkowski{T <: Number}(a::T, b::T, p::Real) = evaluate(Minkowski(p), a, b)

# Hamming
@inline eval_op(::Hamming, ai, bi) = ai != bi ? 1 : 0
@inline eval_reduce(::Hamming, s1, s2) = s1 + s2
hamming(a::AbstractArray, b::AbstractArray) = evaluate(Hamming(), a, b)
hamming{T <: Number}(a::T, b::T) = evaluate(Hamming(), a, b)

# Cosine dist
function eval_start{T<:Real}(::CosineDist, a::AbstractArray{T}, b::AbstractArray{T})
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
cosine_dist(a::AbstractArray, b::AbstractArray) = evaluate(CosineDist(), a, b)

# Correlation Dist
_centralize(x::AbstractArray) = x .- mean(x)
evaluate(::CorrDist, a::AbstractArray, b::AbstractArray) = cosine_dist(_centralize(a), _centralize(b))
corr_dist(a::AbstractArray, b::AbstractArray) = evaluate(CorrDist(), a, b)
result_type(::CorrDist, a::AbstractArray, b::AbstractArray) = result_type(CosineDist(), a, b)

# ChiSqDist
@inline eval_op(::ChiSqDist, ai, bi) = abs2(ai - bi) / (ai + bi)
@inline eval_reduce(::ChiSqDist, s1, s2) = s1 + s2
chisq_dist(a::AbstractArray, b::AbstractArray) = evaluate(ChiSqDist(), a, b)

# KLDivergence
@inline eval_op(::KLDivergence, ai, bi) = ai > 0 ? ai * log(ai / bi) : zero(ai)
@inline eval_reduce(::KLDivergence, s1, s2) = s1 + s2
kl_divergence(a::AbstractArray, b::AbstractArray) = evaluate(KLDivergence(), a, b)

# GenKLDivergence
@inline eval_op(::GenKLDivergence, ai, bi) = ai > 0 ? ai * log(ai / bi) - ai + bi : bi
@inline eval_reduce(::GenKLDivergence, s1, s2) = s1 + s2
gkl_divergence(a::AbstractArray, b::AbstractArray) = evaluate(GenKLDivergence(), a, b)

# RenyiDivergence
function eval_start{T<:AbstractFloat}(::RenyiDivergence, a::AbstractArray{T}, b::AbstractArray{T})
    zero(T), zero(T)
end

@inline function eval_op{T<:AbstractFloat}(dist::RenyiDivergence, ai::T, bi::T)
    if ai == zero(T)
        return zero(T), zero(T)
    elseif dist.is_normal
        return ai, ai .* ((ai ./ bi) .^ dist.p)
    elseif dist.is_zero
        return ai, bi
    elseif dist.is_one
        return ai, ai * log(ai / bi)
    else # otherwise q = ∞
        return ai, ai / bi
    end
end

@inline function eval_reduce{T<:AbstractFloat}(dist::RenyiDivergence,
                                               s1::Tuple{T, T},
                                               s2::Tuple{T, T})
    if dist.is_inf
        if s1[1] == zero(T)
            return s2
        elseif s2[1] == zero(T)
            return s1
        else
            return s1[2] > s2[2] ? s1 : s2
        end
    else
        return s1[1] + s2[1], s1[2] + s2[2]
    end
end

function eval_end(dist::RenyiDivergence, s)
    if dist.is_zero || dist.is_normal
        log(s[2] / s[1]) / dist.p
    elseif dist.is_one
        return s[2] / s[1]
    else # q = ∞
        log(s[2])
    end
end

renyi_divergence(a::AbstractArray, b::AbstractArray, q::Real) = evaluate(RenyiDivergence(q), a, b)

# JSDivergence
@inline function eval_op{T}(::JSDivergence, ai::T, bi::T)
    u = (ai + bi) / 2
    ta = ai > 0 ? ai * log(ai) / 2 : zero(log(one(T)))
    tb = bi > 0 ? bi * log(bi) / 2 : zero(log(one(T)))
    tu = u > 0 ? u * log(u) : zero(log(one(T)))
    ta + tb - tu
end
@inline eval_reduce(::JSDivergence, s1, s2) = s1 + s2
js_divergence(a::AbstractArray, b::AbstractArray) = evaluate(JSDivergence(), a, b)

# SpanNormDist
function eval_start(::SpanNormDist, a::AbstractArray, b::AbstractArray)
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
spannorm_dist(a::AbstractArray, b::AbstractArray) = evaluate(SpanNormDist(), a, b)
function result_type{T1, T2}(dist::SpanNormDist, ::AbstractArray{T1}, ::AbstractArray{T2})
    typeof(eval_op(dist, one(T1), one(T2)))
end


# Jaccard

@inline eval_start(::Jaccard, a::AbstractArray{Bool}, b::AbstractArray{Bool}) = 0, 0
@inline eval_start{T}(::Jaccard, a::AbstractArray{T}, b::AbstractArray{T}) = zero(T), zero(T)
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
    @inbounds v = 1 - (a[1]/a[2])
    return v
end
jaccard(a::AbstractArray, b::AbstractArray) = evaluate(Jaccard(), a, b)

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
rogerstanimoto{T <: Bool}(a::AbstractArray{T}, b::AbstractArray{T}) = evaluate(RogersTanimoto(), a, b)

# Deviations

evaluate(::MeanAbsDeviation, a, b) = cityblock(a, b) / length(a)
meanad(a, b) = evaluate(MeanAbsDeviation(), a, b)

evaluate(::MeanSqDeviation, a, b) = sqeuclidean(a, b) / length(a)
msd(a, b) = evaluate(MeanSqDeviation(), a, b)

evaluate(::RMSDeviation, a, b) = sqrt(evaluate(MeanSqDeviation(), a, b))
rmsd(a, b) = evaluate(RMSDeviation(), a, b)

function evaluate(::NormRMSDeviation, a, b)
    amin, amax = extrema(a)
    return evaluate(RMSDeviation(), a, b) / (amax - amin)
end
nrmsd(a, b) = evaluate(NormRMSDeviation(), a, b)

function evaluate(::CVRMSDeviation, a, b)
    return evaluate(RMSDeviation(), a, b) / mean(a)
end
cvrmsd(a, b) = evaluate(CVRMSDeviation(), a, b)


###########################################################
#
#  Special method
#
###########################################################

# SqEuclidean
function pairwise!(r::AbstractMatrix, dist::SqEuclidean, a::AbstractMatrix, b::AbstractMatrix)
    At_mul_B!(r, a, b)
    sa2 = sum(abs2, a, 1)
    sb2 = sum(abs2, b, 1)
    threshT = convert(eltype(r), dist.thresh)
    if threshT <= 0
        # If there's no chance of triggering the threshold, we can use @simd
        for j = 1 : size(r,2)
            sb = sb2[j]
            @simd for i = 1 : size(r,1)
                @inbounds r[i,j] = sa2[i] + sb - 2 * r[i,j]
            end
        end
    else
        for j = 1 : size(r,2)
            sb = sb2[j]
            for i = 1 : size(r,1)
                @inbounds selfterms = sa2[i] + sb
                @inbounds v = selfterms - 2*r[i,j]
                if v < threshT*selfterms
                    # The distance is likely to be inaccurate, recalculate at higher prec.
                    # This reflects the following:
                    #   ((x+ϵ) - y)^2 ≈ x^2 - 2xy + y^2 + O(ϵ)    when |x-y| >> ϵ
                    #   ((x+ϵ) - y)^2 ≈ O(ϵ^2)                    otherwise
                    v = zero(v)
                    for k = 1:size(a,1)
                        @inbounds v += (a[k,i]-b[k,j])^2
                    end
                end
                @inbounds r[i,j] = v
            end
        end
    end
    r
end

function pairwise!(r::AbstractMatrix, dist::SqEuclidean, a::AbstractMatrix)
    m, n = get_pairwise_dims(r, a)
    At_mul_B!(r, a, a)
    sa2 = sumsq_percol(a)
    threshT = convert(eltype(r), dist.thresh)
    @inbounds for j = 1 : n
        for i = 1 : j-1
            r[i,j] = r[j,i]
        end
        r[j,j] = 0
        sa2j = sa2[j]
        if threshT <= 0
            @simd for i = j+1 : n
                r[i,j] = sa2[i] + sa2j - 2 * r[i,j]
            end
        else
            for i = j+1 : n
                selfterms = sa2[i] + sa2j
                v = selfterms - 2*r[i,j]
                if v < threshT*selfterms
                    v = zero(v)
                    for k = 1:size(a,1)
                        v += (a[k,i]-a[k,j])^2
                    end
                end
                r[i,j] = v
            end
        end
    end
    r
end

# Euclidean
function pairwise!(r::AbstractMatrix, dist::Euclidean, a::AbstractMatrix, b::AbstractMatrix)
    m, na, nb = get_pairwise_dims(r, a, b)
    At_mul_B!(r, a, b)
    sa2 = sumsq_percol(a)
    sb2 = sumsq_percol(b)
    threshT = convert(eltype(r), dist.thresh)
    @inbounds for j = 1 : nb
        sb = sb2[j]
        for i = 1 : na
            selfterms = sa2[i] + sb
            v = selfterms - 2*r[i,j]
            if v < threshT*selfterms
                # The distance is likely to be inaccurate, recalculate directly
                # This reflects the following:
                #   while sqrt(x+ϵ) ≈ sqrt(x) + O(ϵ/sqrt(x)) when |x| >> ϵ,
                #         sqrt(x+ϵ) ≈ O(sqrt(ϵ))             otherwise.
                v = zero(v)
                for k = 1:m
                    v += (a[k,i]-b[k,j])^2
                end
            end
            r[i,j] = sqrt(v)
        end
    end
    r
end

function pairwise!(r::AbstractMatrix, dist::Euclidean, a::AbstractMatrix)
    m, n = get_pairwise_dims(r, a)
    At_mul_B!(r, a, a)
    sa2 = sumsq_percol(a)
    threshT = convert(eltype(r), dist.thresh)
    @inbounds for j = 1 : n
        for i = 1 : j-1
            r[i,j] = r[j,i]
        end
        r[j,j] = 0
        sa2j = sa2[j]
        for i = j+1 : n
            selfterms = sa2[i] + sa2j
            v = selfterms - 2*r[i,j]
            if v < threshT*selfterms
                v = zero(v)
                for k = 1:m
                    v += (a[k,i]-a[k,j])^2
                end
            end
            r[i,j] = sqrt(v)
        end
    end
    r
end

# CosineDist

function pairwise!(r::AbstractMatrix, dist::CosineDist, a::AbstractMatrix, b::AbstractMatrix)
    m, na, nb = get_pairwise_dims(r, a, b)
    At_mul_B!(r, a, b)
    ra = sqrt!(sumsq_percol(a))
    rb = sqrt!(sumsq_percol(b))
    for j = 1 : nb
        @simd for i = 1 : na
            @inbounds r[i,j] = max(1 - r[i,j] / (ra[i] * rb[j]), 0)
        end
    end
    r
end
function pairwise!(r::AbstractMatrix, dist::CosineDist, a::AbstractMatrix)
    m, n = get_pairwise_dims(r, a)
    At_mul_B!(r, a, a)
    ra = sqrt!(sumsq_percol(a))
    @inbounds for j = 1 : n
        @simd for i = j+1 : n
            r[i,j] = max(1 - r[i,j] / (ra[i] * ra[j]), 0)
        end
        r[j,j] = 0
        for i = 1 : j-1
            r[i,j] = r[j,i]
        end
    end
    r
end

# CorrDist
_centralize_colwise(x::AbstractVector) = x .- mean(x)
_centralize_colwise(x::AbstractMatrix) = x .- mean(x, 1)
function colwise!(r::AbstractVector, dist::CorrDist, a::AbstractMatrix, b::AbstractMatrix)
    colwise!(r, CosineDist(), _centralize_colwise(a), _centralize_colwise(b))
end
function colwise!(r::AbstractVector, dist::CorrDist, a::AbstractVector, b::AbstractMatrix)
    colwise!(r, CosineDist(), _centralize_colwise(a), _centralize_colwise(b))
end
function pairwise!(r::AbstractMatrix, dist::CorrDist, a::AbstractMatrix, b::AbstractMatrix)
    pairwise!(r, CosineDist(), _centralize_colwise(a), _centralize_colwise(b))
end
function pairwise!(r::AbstractMatrix, dist::CorrDist, a::AbstractMatrix)
    pairwise!(r, CosineDist(), _centralize_colwise(a))
end
