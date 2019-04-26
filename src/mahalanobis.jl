# Mahalanobis distances

struct Mahalanobis{T} <: Metric
    qmat::Matrix{T}
end

struct SqMahalanobis{T} <: SemiMetric
    qmat::Matrix{T}
end

result_type(::Mahalanobis{T}, ::AbstractArray, ::AbstractArray) where {T} = T
result_type(::SqMahalanobis{T}, ::AbstractArray, ::AbstractArray) where {T} = T

# SqMahalanobis

function evaluate(dist::SqMahalanobis{T}, a::AbstractVector{<:Number}, b::AbstractVector{<:Number}) where {T <: Real}
    if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end

    Q = dist.qmat
    z = a - b
    return dot(z, Q * z)
end

sqmahalanobis(a::AbstractVector{<:Number}, b::AbstractVector{<:Number}, Q::AbstractMatrix{<:Number}) = evaluate(SqMahalanobis(Q), a, b)

function colwise!(r::AbstractArray{<:Number}, dist::SqMahalanobis{T}, a::AbstractMatrix{<:Number}, b::AbstractMatrix{<:Number}) where {T <: Real}
    Q = dist.qmat
    m, n = get_colwise_dims(size(Q, 1), r, a, b)
    z = a - b
    dot_percol!(r, Q * z, z)
end

function colwise!(r::AbstractArray{<:Number}, dist::SqMahalanobis{T}, a::AbstractVector{<:Number}, b::AbstractMatrix{<:Number}) where {T <: Real}
    Q = dist.qmat
    m, n = get_colwise_dims(size(Q, 1), r, a, b)
    z = a .- b
    Qz = Q * z
    dot_percol!(r, Q * z, z)
end

function _pairwise!(r::AbstractMatrix{<:Number}, dist::SqMahalanobis{T},
                    a::AbstractMatrix{<:Number}, b::AbstractMatrix{<:Number}) where {T <: Real}
    Q = dist.qmat
    m, na, nb = get_pairwise_dims(size(Q, 1), r, a, b)

    Qa = Q * a
    Qb = Q * b
    sa2 = dot_percol(a, Qa)
    sb2 = dot_percol(b, Qb)
    mul!(r, a', Qb)

    for j = 1:nb
        @simd for i = 1:na
            @inbounds r[i, j] = sa2[i] + sb2[j] - 2 * r[i, j]
        end
    end
    r
end

function _pairwise!(r::AbstractMatrix{<:Number}, dist::SqMahalanobis{T},
                    a::AbstractMatrix{<:Number}) where {T <: Real}
    Q = dist.qmat
    m, n = get_pairwise_dims(size(Q, 1), r, a)

    Qa = Q * a
    sa2 = dot_percol(a, Qa)
    mul!(r, a', Qa)

    for j = 1:n
        for i = 1:(j - 1)
            @inbounds r[i, j] = r[j, i]
        end
        r[j, j] = 0
        for i = (j + 1):n
            @inbounds r[i, j] = sa2[i] + sa2[j] - 2 * r[i, j]
        end
    end
    r
end


# Mahalanobis

function evaluate(dist::Mahalanobis{T}, a::AbstractVector{<:Number}, b::AbstractVector{<:Number}) where {T <: Real}
    sqrt(evaluate(SqMahalanobis(dist.qmat), a, b))
end

mahalanobis(a::AbstractVector{<:Number}, b::AbstractVector{<:Number}, Q::AbstractMatrix{<:Number}) = evaluate(Mahalanobis(Q), a, b)

function colwise!(r::AbstractArray{<:Number}, dist::Mahalanobis{T}, a::AbstractMatrix{<:Number}, b::AbstractMatrix{<:Number}) where {T <: Real}
    sqrt!(colwise!(r, SqMahalanobis(dist.qmat), a, b))
end

function colwise!(r::AbstractArray{<:Number}, dist::Mahalanobis{T}, a::AbstractVector{<:Number}, b::AbstractMatrix{<:Number}) where {T <: Real}
    sqrt!(colwise!(r, SqMahalanobis(dist.qmat), a, b))
end

function _pairwise!(r::AbstractMatrix{<:Number}, dist::Mahalanobis{T},
                    a::AbstractMatrix{<:Number}, b::AbstractMatrix{<:Number}) where {T <: Real}
    sqrt!(_pairwise!(r, SqMahalanobis(dist.qmat), a, b))
end

function _pairwise!(r::AbstractMatrix{<:Number}, dist::Mahalanobis{T},
                    a::AbstractMatrix{<:Number}) where {T <: Real}
    sqrt!(_pairwise!(r, SqMahalanobis(dist.qmat), a))
end
