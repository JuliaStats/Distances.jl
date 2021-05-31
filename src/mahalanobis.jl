# Mahalanobis distances

struct Mahalanobis{M<:AbstractMatrix} <: Metric
    qmat::M
    _isposdef::Bool
    function Mahalanobis{M}(Q::M, isposdef::Bool) where {M <: AbstractMatrix}
        if !isposdef
            ishermitian(Q) || throw(ArgumentError("bilinear form is not symmetric/Hermitian"))
            @warn "bilinear form is not positive definite"    
        end
        return new{M}(Q, isposdef)
    end
end
Mahalanobis(Q::AbstractMatrix; isposdef::Bool = isposdef(Q)) =
    Mahalanobis{typeof(Q)}(Q, isposdef)

struct SqMahalanobis{M<:AbstractMatrix} <: SemiMetric
    qmat::M
    _isposdef::Bool
    function SqMahalanobis{M}(Q::M, isposdef::Bool) where {M <: AbstractMatrix}
        if !isposdef
            ishermitian(Q) || throw(ArgumentError("bilinear form is not symmetric/Hermitian"))
            @warn "bilinear form is not positive definite"    
        end
        return new{M}(Q, isposdef)
    end
end
SqMahalanobis(Q::AbstractMatrix; isposdef::Bool = isposdef(Q)) =
    SqMahalanobis{typeof(Q)}(Q, isposdef)

function result_type(d::Mahalanobis, ::Type{T1}, ::Type{T2}) where {T1,T2}
    z = zero(T1) - zero(T2)
    return typeof(sqrt(z * zero(eltype(d.qmat)) * z))
end

function result_type(d::SqMahalanobis, ::Type{T1}, ::Type{T2}) where {T1,T2}
    z = zero(T1) - zero(T2)
    return typeof(z * zero(eltype(d.qmat)) * z)
end

# SqMahalanobis

function (dist::SqMahalanobis)(a::AbstractVector, b::AbstractVector)
    if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end

    Q = dist.qmat
    z = a - b
    return dot(z, Q * z)
end

sqmahalanobis(a::AbstractVector, b::AbstractVector, Q::AbstractMatrix, isposdef::Bool = isposdef(Q)) =
    SqMahalanobis(Q, isposdef=isposdef)(a, b)

function colwise!(r::AbstractArray, dist::SqMahalanobis, a::AbstractMatrix, b::AbstractMatrix)
    Q = dist.qmat
    get_colwise_dims(size(Q, 1), r, a, b)
    z = a - b
    dot_percol!(r, Q * z, z)
end

function colwise!(r::AbstractArray, dist::SqMahalanobis, a::AbstractVector, b::AbstractMatrix)
    Q = dist.qmat
    get_colwise_dims(size(Q, 1), r, a, b)
    z = a .- b
    dot_percol!(r, Q * z, z)
end

function _pairwise!(r::AbstractMatrix, dist::SqMahalanobis,
                    a::AbstractMatrix, b::AbstractMatrix)
    Q = dist.qmat
    m, na, nb = get_pairwise_dims(size(Q, 1), r, a, b)

    Qa = Q * a
    Qb = Q * b
    sa2 = dot_percol(a, Qa)
    sb2 = dot_percol(b, Qb)
    mul!(r, a', Qb)

    if dist._isposdef
        for j = 1:nb
            @simd for i = 1:na
                @inbounds r[i, j] = max(sa2[i] + sb2[j] - 2 * r[i, j], 0)
            end
        end
    else # indefinite
        for j = 1:nb
            @simd for i = 1:na
                @inbounds r[i, j] = sa2[i] + sb2[j] - 2 * r[i, j]
            end
        end
    end
    r
end

function _pairwise!(r::AbstractMatrix, dist::SqMahalanobis,
                    a::AbstractMatrix)
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
        if dist._isposdef
            for i = (j + 1):n
                @inbounds r[i, j] = max(sa2[i] + sa2[j] - 2 * r[i, j], 0)
            end
        else
            for i = (j + 1):n
                @inbounds r[i, j] = sa2[i] + sa2[j] - 2 * r[i, j]
            end
        end
    end
    r
end


# Mahalanobis

function (dist::Mahalanobis)(a::AbstractVector, b::AbstractVector)
    sqrt(SqMahalanobis(dist.qmat, isposdef=dist._isposdef)(a, b))
end

mahalanobis(a::AbstractVector, b::AbstractVector, Q::AbstractMatrix, isposdef::Bool = isposdef(Q)) =
    Mahalanobis(Q, isposdef=isposdef)(a, b)

function colwise!(r::AbstractArray, dist::Mahalanobis, a::AbstractMatrix, b::AbstractMatrix)
    sqrt!(colwise!(r, SqMahalanobis(dist.qmat, isposdef=dist._isposdef), a, b))
end

function colwise!(r::AbstractArray, dist::Mahalanobis, a::AbstractVector, b::AbstractMatrix)
    sqrt!(colwise!(r, SqMahalanobis(dist.qmat, isposdef=dist._isposdef), a, b))
end

function _pairwise!(r::AbstractMatrix, dist::Mahalanobis,
                    a::AbstractMatrix, b::AbstractMatrix)
    sqrt!(_pairwise!(r, SqMahalanobis(dist.qmat, isposdef=dist._isposdef), a, b))
end

function _pairwise!(r::AbstractMatrix, dist::Mahalanobis,
                    a::AbstractMatrix)
    sqrt!(_pairwise!(r, SqMahalanobis(dist.qmat, isposdef=dist._isposdef), a))
end
