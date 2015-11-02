# Ordinary metrics

###########################################################
#
#   Metric types
#
###########################################################

type Euclidean <: Metric end
type SqEuclidean <: SemiMetric end
type Chebyshev <: Metric end
type Cityblock <: Metric end

immutable Minkowski{T <: Real} <: Metric
    p::T
end

type Hamming <: Metric end

type CosineDist <: SemiMetric end
type CorrDist <: SemiMetric end

type ChiSqDist <: SemiMetric end
type KLDivergence <: PreMetric end
type JSDivergence <: SemiMetric end

type SpanNormDist <: SemiMetric end

typealias UnionMetrics @compat(Union{Euclidean, SqEuclidean, Chebyshev, Cityblock, Minkowski, Hamming, CosineDist, CorrDist, ChiSqDist, KLDivergence, JSDivergence, SpanNormDist})

###########################################################
#
#  Define Evaluate
#
###########################################################

if VERSION < v"0.4.0-dev+1624"
    eachindex(A::AbstractArray...) = 1:length(A[1])
end

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
@compat @inline eval_op(::SqEuclidean, ai, bi) = abs2(ai - bi)
@compat @inline eval_reduce(::SqEuclidean, s1, s2) = s1 + s2

sqeuclidean(a::AbstractArray, b::AbstractArray) = evaluate(SqEuclidean(), a, b)
sqeuclidean{T <: Number}(a::T, b::T) = evaluate(SqEuclidean(), a, b)

# Euclidean
@compat @inline eval_op(::Euclidean, ai, bi) = abs2(ai - bi)
@compat @inline eval_reduce(::Euclidean, s1, s2) = s1 + s2
eval_end(::Euclidean, s) = sqrt(s)
euclidean(a::AbstractArray, b::AbstractArray) = evaluate(Euclidean(), a, b)
euclidean(a::Number, b::Number) = evaluate(Euclidean(), a, b)

# Cityblock
@compat @inline eval_op(::Cityblock, ai, bi) = abs(ai - bi)
@compat @inline eval_reduce(::Cityblock, s1, s2) = s1 + s2
cityblock(a::AbstractArray, b::AbstractArray) = evaluate(Cityblock(), a, b)
cityblock{T <: Number}(a::T, b::T) = evaluate(Cityblock(), a, b)

# Chebyshev
@compat @inline eval_op(::Chebyshev, ai, bi) = abs(ai - bi)
@compat @inline eval_reduce(::Chebyshev, s1, s2) = max(s1, s2)
# if only NaN, will output NaN
@compat @inline eval_start(::Chebyshev, a::AbstractArray, b::AbstractArray) = abs(a[1] - b[1])
chebyshev(a::AbstractArray, b::AbstractArray) = evaluate(Chebyshev(), a, b)
chebyshev{T <: Number}(a::T, b::T) = evaluate(Chebyshev(), a, b)

# Minkowski
@compat @inline eval_op(dist::Minkowski, ai, bi) = abs(ai - bi) .^ dist.p
@compat @inline eval_reduce(::Minkowski, s1, s2) = s1 + s2
eval_end(dist::Minkowski, s) = s .^ (1/dist.p)
minkowski(a::AbstractArray, b::AbstractArray, p::Real) = evaluate(Minkowski(p), a, b)
minkowski{T <: Number}(a::T, b::T, p::Real) = evaluate(Minkowski(p), a, b)

# Hamming
@compat @inline eval_op(::Hamming, ai, bi) = ai != bi ? 1 : 0
@compat @inline eval_reduce(::Hamming, s1, s2) = s1 + s2
hamming(a::AbstractArray, b::AbstractArray) = evaluate(Hamming(), a, b)
hamming{T <: Number}(a::T, b::T) = evaluate(Hamming(), a, b)

# Cosine dist
function eval_start{T<:AbstractFloat}(::CosineDist, a::AbstractArray{T}, b::AbstractArray{T})
    zero(T), zero(T), zero(T)
end
@compat @inline eval_op(::CosineDist, ai, bi) = ai * bi, ai * ai, bi * bi
@compat @inline function eval_reduce(::CosineDist, s1, s2)
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

# ChiSqDist
@compat @inline eval_op(::ChiSqDist, ai, bi) = abs2(ai - bi) / (ai + bi)
@compat @inline eval_reduce(::ChiSqDist, s1, s2) = s1 + s2
chisq_dist(a::AbstractArray, b::AbstractArray) = evaluate(ChiSqDist(), a, b)

# KLDivergence
@compat @inline eval_op(::KLDivergence, ai, bi) = ai > 0 ? ai * log(ai / bi) : zero(ai)
@compat @inline eval_reduce(::KLDivergence, s1, s2) = s1 + s2
kl_divergence(a::AbstractArray, b::AbstractArray) = evaluate(KLDivergence(), a, b)

# JSDivergence
@compat @inline function eval_op{T}(::JSDivergence, ai::T, bi::T)
    u = (ai + bi) / 2
    ta = ai > 0 ? ai * log(ai) / 2 : zero(log(one(T)))
    tb = bi > 0 ? bi * log(bi) / 2 : zero(log(one(T)))
    tu = u > 0 ? u * log(u) : zero(log(one(T)))
    ta + tb - tu
end
@compat @inline eval_reduce(::JSDivergence, s1, s2) = s1 + s2
js_divergence(a::AbstractArray, b::AbstractArray) = evaluate(JSDivergence(), a, b)

# SpanNormDist
function eval_start(::SpanNormDist, a::AbstractArray, b::AbstractArray)
    a[1] - b[1], a[1]- b[1]
end
@compat @inline eval_op(::SpanNormDist, ai, bi)  = ai - bi
@compat @inline function eval_reduce(::SpanNormDist, s1, s2)
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
result_type(dist::SpanNormDist, T1::Type, T2::Type) = typeof(eval_op(dist, one(T1), one(T2)))


###########################################################
#
#  Special method
#
###########################################################

# SqEuclidean
function pairwise!(r::AbstractMatrix, dist::SqEuclidean, a::AbstractMatrix, b::AbstractMatrix)
    At_mul_B!(r, a, b)
    sa2 = sumabs2(a, 1)
    sb2 = sumabs2(b, 1)
    pdist!(r, sa2, sb2)
end
function pdist!(r, sa2, sb2)
    for j = 1 : size(r,2)
        sb = sb2[j]
        @simd for i = 1 : size(r,1)
            @inbounds r[i,j] = sa2[i] + sb - 2 * r[i,j]
        end
    end
    r
end
function pairwise!(r::AbstractMatrix, dist::SqEuclidean, a::AbstractMatrix)
    m::Int, n::Int = get_pairwise_dims(r, a)
    At_mul_B!(r, a, a)
    sa2 = sumsq_percol(a)
    for j = 1 : n
        for i = 1 : j-1
            @inbounds r[i,j] = r[j,i]
        end
        @inbounds r[j,j] = 0
        for i = j+1 : n
            @inbounds r[i,j] = sa2[i] + sa2[j] - 2 * r[i,j]
        end
    end
    r
end

# Euclidean
function pairwise!(r::AbstractMatrix, dist::Euclidean, a::AbstractMatrix, b::AbstractMatrix)
    m::Int, na::Int, nb::Int = get_pairwise_dims(r, a, b)
    At_mul_B!(r, a, b)
    sa2 = sumsq_percol(a)
    sb2 = sumsq_percol(b)
    for j = 1 : nb
        for i = 1 : na
            @inbounds v = sa2[i] + sb2[j] - 2 * r[i,j]
            @inbounds r[i,j] = isnan(v) ? NaN : sqrt(max(v, 0.))
        end
    end
    r
end
function pairwise!(r::AbstractMatrix, dist::Euclidean, a::AbstractMatrix)
    m::Int, n::Int = get_pairwise_dims(r, a)
    At_mul_B!(r, a, a)
    sa2 = sumsq_percol(a)
    for j = 1 : n
        for i = 1 : j-1
            @inbounds r[i,j] = r[j,i]
        end
        @inbounds r[j,j] = 0
        for i = j+1 : n
            @inbounds v = sa2[i] + sa2[j] - 2 * r[i,j]
            @inbounds r[i,j] = isnan(v) ? NaN : sqrt(max(v, 0.))
        end
    end
    r
end

# CosineDist
function pairwise!(r::AbstractMatrix, dist::CosineDist, a::AbstractMatrix, b::AbstractMatrix)
    m::Int, na::Int, nb::Int = get_pairwise_dims(r, a, b)
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
    for j = 1 : n
        @simd for i = j+1 : n
            @inbounds r[i,j] = max(1 - r[i,j] / (ra[i] * ra[j]), 0)
        end
        @inbounds r[j,j] = 0
        for i = 1 : j-1
            @inbounds r[i,j] = r[j,i]
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





