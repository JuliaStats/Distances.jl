# Weighted metrics


###########################################################
#
#   Metric types
#
###########################################################

type WeightedEuclidean{T<:AbstractFloat} <: Metric
    weights::Vector{T}
end

type WeightedSqEuclidean{T<:AbstractFloat} <: SemiMetric
    weights::Vector{T}
end

type WeightedCityblock{T<:AbstractFloat} <: Metric
    weights::Vector{T}
end

type WeightedMinkowski{T<:AbstractFloat} <: Metric
    weights::Vector{T}
    p::Real
end

type WeightedHamming{T<:AbstractFloat} <: Metric
    weights::Vector{T}
end

result_type{T}(::WeightedEuclidean{T}, T1::Type, T2::Type) = T
result_type{T}(::WeightedSqEuclidean{T}, T1::Type, T2::Type) = T
result_type{T}(::WeightedCityblock{T}, T1::Type, T2::Type) = T
result_type{T}(::WeightedMinkowski{T}, T1::Type, T2::Type) = T
result_type{T}(::WeightedHamming{T}, T1::Type, T2::Type) = T


###########################################################
#
#   Specialized distances
#
###########################################################


# Weighted squared Euclidean

function wsumsqdiff(w::AbstractVector, a::AbstractVector, b::AbstractVector)
    n = get_common_len(w, a, b)::Int
    s = 0.
    for i = 1:n
        @inbounds s += abs2(a[i] - b[i]) * w[i]
    end
    s
end

evaluate{T<:AbstractFloat}(dist::WeightedSqEuclidean{T}, a::AbstractVector, b::AbstractVector) = wsumsqdiff(dist.weights, a, b)
wsqeuclidean(a::AbstractVector, b::AbstractVector, w::AbstractVector) = evaluate(WeightedSqEuclidean(w), a, b)

function pairwise!{T<:AbstractFloat}(r::AbstractMatrix, dist::WeightedSqEuclidean{T}, a::AbstractMatrix, b::AbstractMatrix)
    w = dist.weights
    m::Int, na::Int, nb::Int = get_pairwise_dims(length(w), r, a, b)

    sa2 = wsumsq_percol(w, a)
    sb2 = wsumsq_percol(w, b)
    At_mul_B!(r, a, b .* w)

    for j = 1 : nb
        for i = 1 : na
            @inbounds r[i,j] = sa2[i] + sb2[j] - 2 * r[i,j]
        end
    end
    r
end

function pairwise!{T<:AbstractFloat}(r::AbstractMatrix, dist::WeightedSqEuclidean{T}, a::AbstractMatrix)
    w = dist.weights
    m::Int, n::Int = get_pairwise_dims(length(w), r, a)

    sa2 = wsumsq_percol(w, a)
    At_mul_B!(r, a, a .* w)

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


# Weighted Euclidean

function evaluate{T<:AbstractFloat}(dist::WeightedEuclidean{T}, a::AbstractVector, b::AbstractVector)
    sqrt(evaluate(WeightedSqEuclidean(dist.weights), a, b))
end

weuclidean(a::AbstractVector, b::AbstractVector, w::AbstractVector) = evaluate(WeightedEuclidean(w), a, b)

function colwise!{T<:AbstractFloat}(r::AbstractArray, dist::WeightedEuclidean{T}, a::AbstractMatrix, b::AbstractMatrix)
    sqrt!(colwise!(r, WeightedSqEuclidean(dist.weights), a, b))
end

function colwise!{T<:AbstractFloat}(r::AbstractArray, dist::WeightedEuclidean{T}, a::AbstractVector, b::AbstractMatrix)
    sqrt!(colwise!(r, WeightedSqEuclidean(dist.weights), a, b))
end

function pairwise!{T<:AbstractFloat}(r::AbstractMatrix, dist::WeightedEuclidean{T}, a::AbstractMatrix, b::AbstractMatrix)
    sqrt!(pairwise!(r, WeightedSqEuclidean(dist.weights), a, b))
end

function pairwise!{T<:AbstractFloat}(r::AbstractMatrix, dist::WeightedEuclidean{T}, a::AbstractMatrix)
    sqrt!(pairwise!(r, WeightedSqEuclidean(dist.weights), a))
end


# Weighted Cityblock

function evaluate{T<:AbstractFloat}(dist::WeightedCityblock{T}, a::AbstractVector, b::AbstractVector) 
    w = dist.weights
    n = get_common_len(w, a, b)::Int
    s = 0.
    for i = 1:n
        @inbounds s += w[i] * abs(a[i] - b[i])
    end
    s
end
wcityblock(a::AbstractVector, b::AbstractVector, w::AbstractVector) = evaluate(WeightedCityblock(w), a, b)


# WeightedMinkowski

function evaluate{T<:AbstractFloat}(dist::WeightedMinkowski{T}, a::AbstractVector, b::AbstractVector) 
    w = dist.weights
    p = dist.p
    n = get_common_len(w, a, b)
    s = 0.
    for i = 1:n
        @inbounds s += w[i] * (abs(a[i] - b[i]) .^ p)
    end
    s .^ inv(p)
end

wminkowski(a::AbstractVector, b::AbstractVector, w::AbstractVector, p::Real) = evaluate(WeightedMinkowski(w, p), a, b)


# WeightedHamming

function evaluate{T<:AbstractFloat}(dist::WeightedHamming{T}, a::AbstractVector, b::AbstractVector)
    n = length(a)
    w = dist.weights

    r = zero(T)
    for i = 1 : n
        @inbounds if a[i] != b[i]
            r += w[i]
        end
    end
    return r
end

whamming(a::AbstractVector, b::AbstractVector, w::AbstractVector) = evaluate(WeightedHamming(w), a, b)
