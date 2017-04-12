# Mahalanobis distances

immutable Mahalanobis{T} <: Metric
    qmat::Matrix{T}
end

immutable SqMahalanobis{T} <: SemiMetric
    qmat::Matrix{T}
end

result_type{T}(::Mahalanobis{T}, ::AbstractArray, ::AbstractArray) = T
result_type{T}(::SqMahalanobis{T}, ::AbstractArray, ::AbstractArray) = T

# SqMahalanobis

function evaluate{T<:AbstractFloat}(dist::SqMahalanobis{T}, a::AbstractVector, b::AbstractVector)
    if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end
    
    Q = dist.qmat
    z = a - b
    return dot(z, Q * z)
end

sqmahalanobis(a::AbstractVector, b::AbstractVector, Q::AbstractMatrix) = evaluate(SqMahalanobis(Q), a, b)

function colwise!{T<:AbstractFloat}(r::AbstractArray, dist::SqMahalanobis{T}, a::AbstractMatrix, b::AbstractMatrix)
    Q = dist.qmat
    m, n = get_colwise_dims(size(Q, 1), r, a, b)
    z = a - b
    dot_percol!(r, Q * z, z)
end

function colwise!{T<:AbstractFloat}(r::AbstractArray, dist::SqMahalanobis{T}, a::AbstractVector, b::AbstractMatrix)
    Q = dist.qmat
    m, n = get_colwise_dims(size(Q, 1), r, a, b)
    z = a .- b
    Qz = Q * z
    dot_percol!(r, Q * z, z)
end

function pairwise!{T<:AbstractFloat}(r::AbstractMatrix, dist::SqMahalanobis{T}, a::AbstractMatrix, b::AbstractMatrix)
    Q = dist.qmat
    m, na, nb = get_pairwise_dims(size(Q, 1), r, a, b)

    Qa = Q * a
    Qb = Q * b
    sa2 = dot_percol(a, Qa)
    sb2 = dot_percol(b, Qb)
    At_mul_B!(r, a, Qb)

    for j = 1 : nb
        @simd for i = 1 : na
            @inbounds r[i,j] = sa2[i] + sb2[j] - 2 * r[i,j]
        end
    end
    r
end

function pairwise!{T<:AbstractFloat}(r::AbstractMatrix, dist::SqMahalanobis{T}, a::AbstractMatrix)
    Q = dist.qmat
    m, n = get_pairwise_dims(size(Q, 1), r, a)

    Qa = Q * a
    sa2 = dot_percol(a, Qa)
    At_mul_B!(r, a, Qa)

    for j = 1 : n
        for i = 1 : j-1
            @inbounds r[i,j] = r[j,i]
        end
        r[j,j] = 0
        for i = j+1 : n
            @inbounds r[i,j] = sa2[i] + sa2[j] - 2 * r[i,j]
        end
    end
    r
end


# Mahalanobis

function evaluate{T<:AbstractFloat}(dist::Mahalanobis{T}, a::AbstractVector, b::AbstractVector)
    sqrt(evaluate(SqMahalanobis(dist.qmat), a, b))
end

mahalanobis(a::AbstractVector, b::AbstractVector, Q::AbstractMatrix) = evaluate(Mahalanobis(Q), a, b)

function colwise!{T<:AbstractFloat}(r::AbstractArray, dist::Mahalanobis{T}, a::AbstractMatrix, b::AbstractMatrix)
    sqrt!(colwise!(r, SqMahalanobis(dist.qmat), a, b))
end

function colwise!{T<:AbstractFloat}(r::AbstractArray, dist::Mahalanobis{T}, a::AbstractVector, b::AbstractMatrix)
    sqrt!(colwise!(r, SqMahalanobis(dist.qmat), a, b))
end

function pairwise!{T<:AbstractFloat}(r::AbstractMatrix, dist::Mahalanobis{T}, a::AbstractMatrix, b::AbstractMatrix)
    sqrt!(pairwise!(r, SqMahalanobis(dist.qmat), a, b))
end

function pairwise!{T<:AbstractFloat}(r::AbstractMatrix, dist::Mahalanobis{T}, a::AbstractMatrix)
    sqrt!(pairwise!(r, SqMahalanobis(dist.qmat), a))
end
