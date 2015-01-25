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

type Minkowski <: Metric
    p::Real
end

type Hamming <: Metric end
result_type(::Hamming, T1::Type, T2::Type) = Int

type CosineDist <: SemiMetric end
type CorrDist <: SemiMetric end

type ChiSqDist <: SemiMetric end
type KLDivergence <: PreMetric end
type JSDivergence <: SemiMetric end

type SpanNormDist <: SemiMetric end


###########################################################
#
#   Specialized distances
#
###########################################################

# SqEuclidean

function sumsqdiff(a::AbstractVector, b::AbstractVector)
    n = get_common_len(a, b)::Int
    s = 0.
    for i = 1:n
        @inbounds s += abs2(a[i] - b[i])
    end
    s
end

evaluate(dist::SqEuclidean, a::AbstractVector, b::AbstractVector) = sumsqdiff(a, b)
sqeuclidean(a::AbstractVector, b::AbstractVector) = evaluate(SqEuclidean(), a, b)
evaluate{T <: Number}(dist::SqEuclidean, a::T, b::T) = abs2(a-b)
sqeuclidean{T <: Number}(a::T, b::T) = evaluate(SqEuclidean(), a, b)

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

evaluate(dist::Euclidean, a::AbstractVector, b::AbstractVector) = sqrt(sumsqdiff(a, b))
euclidean(a::AbstractVector, b::AbstractVector) = evaluate(Euclidean(), a, b)
evaluate{T <: Number}(dist::Euclidean, a::T, b::T) = abs(a-b)
euclidean{T <: Number}(a::T, b::T) = evaluate(Euclidean(), a, b)

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


# Cityblock

function evaluate(dist::Cityblock, a::AbstractVector, b::AbstractVector)
    n = get_common_len(a, b)::Int
    s = 0.
    for i = 1:n
        @inbounds s += abs(a[i] - b[i])
    end
    s
end
cityblock(a::AbstractVector, b::AbstractVector) = evaluate(Cityblock(), a, b)
evaluate{T <: Number}(dist::Cityblock, a::T, b::T) = abs(a-b)
cityblock{T <: Number}(a::T, b::T) = evaluate(Cityblock(), a, b)


# Chebyshev

function evaluate(dist::Chebyshev, a::AbstractVector, b::AbstractVector)
    n = get_common_len(a, b)
    s = 0.
    for i = 1:n
        @inbounds ai = abs(a[i] - b[i])
        if ai > s
            s = ai
        end
    end
    s
end
chebyshev(a::AbstractVector, b::AbstractVector) = evaluate(Chebyshev(), a, b)
evaluate{T <: Number}(dist::Chebyshev, a::T, b::T) = abs(a-b)
chebyshev{T <: Number}(a::T, b::T) = evaluate(Chebyshev(), a, b)


# Minkowski

function evaluate(dist::Minkowski, a::AbstractVector, b::AbstractVector)
    n = get_common_len(a, b)
    s = 0.
    p = dist.p
    for i = 1:n
        @inbounds s += abs(a[i] - b[i]) .^ p
    end
    s .^ inv(p)
end
minkowski(a::AbstractVector, b::AbstractVector, p::Real) = evaluate(Minkowski(p), a, b)
evaluate{T <: Number}(dist::Minkowski, a::T, b::T) = abs(a-b)
minkowski{T <: Number}(a::T, b::T, p::Real) = evaluate(Minkowski(p), a, b)


# Hamming

function evaluate(dist::Hamming, a::AbstractVector, b::AbstractVector)
    n = length(a)
    r = 0
    for i = 1 : n
        @inbounds if a[i] != b[i]
            r += 1
        end
    end
    r
end

hamming(a::AbstractVector, b::AbstractVector) = evaluate(Hamming(), a, b)
evaluate{T <: Number}(dist::Hamming, a::T, b::T) = a != b ? 1 : 0
hamming{T <: Number}(a::T, b::T) = evaluate(Hamming(), a, b)


# Cosine dist

function evaluate{T<:FloatingPoint}(dist::CosineDist, a::AbstractVector{T}, b::AbstractVector{T})
    n = get_common_len(a, b)
    ab = zero(T)
    a2 = zero(T)
    b2 = zero(T)

    for i = 1:n
        @inbounds ai = a[i]
        @inbounds bi = b[i]
        ab += ai * bi
        a2 += ai * ai
        b2 += bi * bi
    end
    max(1 - ab / (sqrt(a2) * sqrt(b2)), 0)
end

cosine_dist(a::AbstractVector, b::AbstractVector) = evaluate(CosineDist(), a, b)

function pairwise!(r::AbstractMatrix, dist::CosineDist, a::AbstractMatrix, b::AbstractMatrix)
    m::Int, na::Int, nb::Int = get_pairwise_dims(r, a, b)
    At_mul_B!(r, a, b)
    ra = sqrt!(sumsq_percol(a))
    rb = sqrt!(sumsq_percol(b))
    for j = 1 : nb
        for i = 1 : na
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
        for i = j+1 : n
            @inbounds r[i,j] = max(1 - r[i,j] / (ra[i] * ra[j]), 0)
        end
        @inbounds r[j,j] = 0
        for i = 1 : j-1
            @inbounds r[i,j] = r[j,i]
        end
    end
    r
end


# Correlation Dist

_centralize(x::AbstractVector) = x .- mean(x)
_centralize(x::AbstractMatrix) = x .- mean(x, 1)

evaluate(dist::CorrDist, a::AbstractVector, b::AbstractVector) = cosine_dist(_centralize(a), _centralize(b))
corr_dist(a::AbstractVector, b::AbstractVector) = evaluate(CorrDist(), a, b)

function colwise!(r::AbstractArray, dist::CorrDist, a::AbstractMatrix, b::AbstractMatrix)
    colwise!(r, CosineDist(), _centralize(a), _centralize(b))
end

function colwise!(r::AbstractArray, dist::CorrDist, a::AbstractVector, b::AbstractMatrix)
    colwise!(r, CosineDist(), _centralize(a), _centralize(b))
end

function pairwise!(r::AbstractMatrix, dist::CorrDist, a::AbstractMatrix, b::AbstractMatrix)
    pairwise!(r, CosineDist(), _centralize(a), _centralize(b))
end

function pairwise!(r::AbstractMatrix, dist::CorrDist, a::AbstractMatrix)
    pairwise!(r, CosineDist(), _centralize(a))
end


# Chi-square distance

function evaluate{T<:FloatingPoint}(dist::ChiSqDist, a::AbstractVector{T}, b::AbstractVector{T})
    r = zero(T)
    for i = 1 : length(a)
        @inbounds ai = a[i]
        @inbounds bi = b[i]
        r += abs2(ai - bi) / (ai + bi)
    end
    r
end

chisq_dist(a::AbstractVector, b::AbstractVector) = evaluate(ChiSqDist(), a, b)


# KL divergence

function evaluate{T<:FloatingPoint}(dist::KLDivergence, a::AbstractVector{T}, b::AbstractVector{T})
    r = zero(T)
    for i = 1 : length(a)
        @inbounds ai = a[i]
        @inbounds bi = b[i]
        if ai > 0
            r += ai * log(ai / bi)
        end
    end
    r
end

kl_divergence(a::AbstractVector, b::AbstractVector) = evaluate(KLDivergence(), a, b)


# JS divergence

function evaluate{T<:FloatingPoint}(dist::JSDivergence, a::AbstractVector{T}, b::AbstractVector{T})
    r = zero(T)
    n = length(a)
    for i = 1 : n
        @inbounds ai = a[i]
        @inbounds bi = b[i]
        u = (ai + bi) / 2
        ta = ai > 0 ? ai * log(ai) / 2 : 0
        tb = bi > 0 ? bi * log(bi) / 2 : 0
        tu = u > 0 ? u * log(u) : 0
        r += (ta + tb - tu)
    end
    r
end

js_divergence(a::AbstractVector, b::AbstractVector) = evaluate(JSDivergence(), a, b)


# SpanNorm

function evaluate(dist::SpanNormDist, a::AbstractVector, b::AbstractVector)
    max_d = min_d = a[1] - b[1]
    for i = 1 : length(a)
        @inbounds di = a[i] - b[i]
        if di > max_d
            max_d = di
        elseif di < min_d
            min_d = di
        end
    end
    max_d - min_d
end

spannorm_dist(a::AbstractVector, b::AbstractVector) = evaluate(SpanNormDist(), a, b)
