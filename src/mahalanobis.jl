# Mahalanobis distances

struct SqMahalanobis{T} <: SemiMetric
    qmat::Matrix{T}
end

struct Mahalanobis{T} <: Metric
    qmat::Matrix{T}
end

result_type(::SqMahalanobis{T}, ::AbstractArray, ::AbstractArray) where {T} = T
result_type(::Mahalanobis{T}, ::AbstractArray, ::AbstractArray) where {T} = T

# SqMahalanobis

function evaluate(dist::SqMahalanobis{T}, a::AbstractVector, b::AbstractVector) where {T <: Real}
    if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end

    Q = dist.qmat
    z = a - b
    return dot(z, Q * z)
end

sqmahalanobis(a::AbstractVector, b::AbstractVector, Q::AbstractMatrix) = evaluate(SqMahalanobis(Q), a, b)

function colwise!(r::AbstractArray, dist::SqMahalanobis{T}, a::AbstractMatrix, b::AbstractMatrix) where {T <: Real}
    Q = dist.qmat
    m, n = get_colwise_dims(size(Q, 1), r, a, b)
    z = a - b
    dot_percol!(r, Q * z, z)
end

function colwise!(r::AbstractArray, dist::SqMahalanobis{T}, a::AbstractVector, b::AbstractMatrix) where {T <: Real}
    Q = dist.qmat
    m, n = get_colwise_dims(size(Q, 1), r, a, b)
    z = a .- b
    Qz = Q * z
    dot_percol!(r, Q * z, z)
end

function pairwise!(r::AbstractMatrix, dist::SqMahalanobis{T}, a::AbstractMatrix, b::AbstractMatrix) where {T <: Real}
    Q = dist.qmat
    m, na, nb = get_pairwise_dims(size(Q, 1), r, a, b)

    Qa = Q * a
    Qb = Q * b
    sa2 = dot_percol(a, Qa)
    sb2 = dot_percol(b, Qb)
    At_mul_B!(r, a, Qb)

    for j = 1:nb
        @simd for i = 1:na
            @inbounds r[i, j] = sa2[i] + sb2[j] - 2 * r[i, j]
        end
    end
    r
end

function pairwise!(r::AbstractMatrix, dist::SqMahalanobis{T}, a::AbstractMatrix) where {T <: Real}
    Q = dist.qmat
    m, n = get_pairwise_dims(size(Q, 1), r, a)

    Qa = Q * a
    sa2 = dot_percol(a, Qa)
    At_mul_B!(r, a, Qa)

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

function evaluate(dist::Mahalanobis{T}, a::AbstractVector, b::AbstractVector) where {T <: Real}
    sqrt(evaluate(SqMahalanobis(dist.qmat), a, b))
end

mahalanobis(a::AbstractVector, b::AbstractVector, Q::AbstractMatrix) = evaluate(Mahalanobis(Q), a, b)

function colwise!(r::AbstractArray, dist::Mahalanobis{T}, a::AbstractMatrix, b::AbstractMatrix) where {T <: Real}
    sqrt!(colwise!(r, SqMahalanobis(dist.qmat), a, b))
end

function colwise!(r::AbstractArray, dist::Mahalanobis{T}, a::AbstractVector, b::AbstractMatrix) where {T <: Real}
    sqrt!(colwise!(r, SqMahalanobis(dist.qmat), a, b))
end

function pairwise!(r::AbstractMatrix, dist::Mahalanobis{T}, a::AbstractMatrix, b::AbstractMatrix) where {T <: Real}
    sqrt!(pairwise!(r, SqMahalanobis(dist.qmat), a, b))
end

function pairwise!(r::AbstractMatrix, dist::Mahalanobis{T}, a::AbstractMatrix) where {T <: Real}
    sqrt!(pairwise!(r, SqMahalanobis(dist.qmat), a))
end


# Ellipsoidal

struct Ellipsoidal{N,T} <: Metric
    dist::Mahalanobis{T}

    function Ellipsoidal{N,T}(semiaxes, angles) where {N,T<:Real}
        @assert length(semiaxes) == N "number of semiaxes must match spatial dimension"
        @assert all(semiaxes .> zero(T)) "semiaxes must be positive"
        @assert N ∈ [2,3] "dimension must be either 2 or 3"

        # scaling matrix
        Λ = spdiagm(one(T)./semiaxes.^2)

        # rotation matrix
        if N == 2
            θ = angles[1]

            cosθ = cos(θ)
            sinθ = sin(θ)

            P = [cosθ -sinθ
                 sinθ  cosθ]
        end
        if N == 3
            θxy, θyz, θzx = angles

            cosxy = cos(θxy)
            sinxy = sin(θxy)
            cosyz = cos(θyz)
            sinyz = sin(θyz)
            coszx = cos(θzx)
            sinzx = sin(θzx)

            _1 = one(T)
            _0 = zero(T)

            Rxy = [cosxy -sinxy _0
                   sinxy  cosxy _0
                      _0     _0 _1]

            Ryz = [_1    _0     _0
                   _0 cosyz -sinyz
                   _0 sinyz  cosyz]

            Rzx = [ coszx _0 sinzx
                       _0 _1    _0
                   -sinzx _0 coszx]

            P = Rzx*Ryz*Rxy
        end

        # ellipsoid matrix
        Q = P*Λ*P'

        new(Mahalanobis(Q))
    end
end

Ellipsoidal(semiaxes::Vector{T}, angles::Vector{T}) where {T<:Real} =
  Ellipsoidal{length(semiaxes),T}(semiaxes, angles)

result_type(::Ellipsoidal{N,T}, ::AbstractArray, ::AbstractArray) where {N,T} = T

function evaluate(dist::Ellipsoidal{N,T}, a::AbstractVector, b::AbstractVector) where {N,T<:Real}
    evaluate(dist.dist, a, b)
end
