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

immutable RenyiDivergence{T <: Real} <: PreMetric
    p::T
    is_normal::Bool
    is_zero::Bool
    is_one::Bool
    is_inf::Bool
    function RenyiDivergence(q)
        # There are four different cases:
        #   simpler to separate them out now, not over and over in eval_op()
        is_zero = q ≈ 0
        is_one = q ≈ 1
        is_inf = isinf(q)
        
        # Only positive Rényi divergences are defined
        !is_zero && q < 0 && throw(ArgumentError("Order of Rényi divergence not legal, $(q) < 0."))
        
        new(q - 1, !(is_zero || is_one || is_inf), is_zero, is_one, is_inf)
    end
end
RenyiDivergence{T}(q::T) = RenyiDivergence{T}(q)

type JSDivergence <: SemiMetric end

type SpanNormDist <: SemiMetric end


typealias UnionMetrics Union{Euclidean, SqEuclidean, Chebyshev, Cityblock, Minkowski, Hamming, Jaccard, RogersTanimoto, CosineDist, CorrDist, ChiSqDist, KLDivergence, RenyiDivergence, JSDivergence, SpanNormDist}

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
function eval_start{T<:AbstractFloat}(::CosineDist, a::AbstractArray{T}, b::AbstractArray{T})
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

# RenyiDivergence
function eval_start{T<:AbstractFloat}(::RenyiDivergence, a::AbstractArray{T}, b::AbstractArray{T})
    zero(T), zero(T)
end

@inline function eval_op(dist::RenyiDivergence, ai, bi)
    if ai == zero(ai)
        return zero(ai), zero(ai)
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

@inline eval_reduce(dist::RenyiDivergence, s1, s2) =
    s1[1] + s2[1], (dist.is_inf ? max(s1[2], s2[2]) : s1[2] + s2[2])
eval_end(dist::RenyiDivergence, s) =
    dist.is_one ? s[2] / s[1] : (dist.is_inf ? log(s[2]) : log(s[2] / s[1]) / dist.p)
renyi_divergence(a::AbstractArray, b::AbstractArray, q::Real) = evaluate(RenyiDivergence(q), a, b)
renyi_divergence{T <: Number}(a::T, b::T, q::Real) = evaluate(RenyiDivergence(q), a, b)

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
    m, n = get_pairwise_dims(r, a)
    At_mul_B!(r, a, a)
    sa2 = sumsq_percol(a)
    @inbounds for j = 1 : n
        for i = 1 : j-1
            r[i,j] = r[j,i]
        end
        r[j,j] = 0
        for i = j+1 : n
            r[i,j] = sa2[i] + sa2[j] - 2 * r[i,j]
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
    @inbounds for j = 1 : nb
        for i = 1 : na
            v = sa2[i] + sb2[j] - 2 * r[i,j]
            r[i,j] = isnan(v) ? NaN : sqrt(max(v, 0.))
        end
    end
    r
end

function pairwise!(r::AbstractMatrix, dist::Euclidean, a::AbstractMatrix)
    m, n = get_pairwise_dims(r, a)
    At_mul_B!(r, a, a)
    sa2 = sumsq_percol(a)
    @inbounds for j = 1 : n
        for i = 1 : j-1
            r[i,j] = r[j,i]
        end
        @inbounds r[j,j] = 0
        for i = j+1 : n
            v = sa2[i] + sa2[j] - 2 * r[i,j]
            r[i,j] = isnan(v) ? NaN : sqrt(max(v, 0.))
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
