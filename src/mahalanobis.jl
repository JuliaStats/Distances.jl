# Mahalanobis distances

"""
    Mahalanobis(Q; skipchecks=false) <: Metric

Create a Mahalanobis distance (i.e., a bilinear form) with covariance matrix `Q`.
Upon construction, both symmetry/self-adjointness and positive semidefiniteness are checked,
where the latter check can be skipped by passing the keyword argument `skipchecks = true`.

# Example:
```julia
julia> A = collect(reshape(1:9, 3, 3)); Q = A'A;

julia> dist = Mahalanobis(Q)
Mahalanobis{Matrix{Int64}}([14 32 50; 32 77 122; 50 122 194])

julia> dist = Mahalanobis(A, skipchecks=true)
┌ Warning: matrix is not symmetric/Hermitian
└ @ Distances ...
Mahalanobis{Matrix{Int64}}([1 4 7; 2 5 8; 3 6 9])
"""
struct Mahalanobis{M<:AbstractMatrix} <: Metric
    qmat::M
    function Mahalanobis(Q::AbstractMatrix; skipchecks::Bool=false)
        # TODO: turn the warnings into errors in next breaking release
        ishermitian(Q) || @warn "matrix is not symmetric/Hermitian"
        if !skipchecks
            eigmin(Q) ≥ 0 || @warn "matrix is not positive semidefinite"
        end
        return new{typeof(Q)}(Q)
    end
end

"""
    SqMahalanobis(Q; skipchecks=false) <: Metric

Create a squared Mahalanobis distance (i.e., a bilinear form) with covariance matrix `Q`.
Upon construction, both symmetry/self-adjointness and positive semidefiniteness are checked,
where the latter check can be skipped by passing the keyword argument `skipchecks = true`.

# Example:
```julia
julia> A = collect(reshape(1:9, 3, 3)); Q = A'A;

julia> dist = SqMahalanobis(Q)
SqMahalanobis{Matrix{Int64}}([14 32 50; 32 77 122; 50 122 194])

julia> dist = SqMahalanobis(A, skipchecks=true)
┌ Warning: matrix is not symmetric/Hermitian
└ @ Distances ...
SqMahalanobis{Matrix{Int64}}([1 4 7; 2 5 8; 3 6 9])
"""
struct SqMahalanobis{M<:AbstractMatrix} <: SemiMetric
    qmat::M
    function SqMahalanobis(Q::AbstractMatrix; skipchecks::Bool=false)
        # TODO: turn the warnings into errors in next breaking release
        ishermitian(Q) || @warn "matrix is not symmetric/Hermitian"
        if !skipchecks
            eigmin(Q) ≥ 0 || @warn "matrix is not positive semidefinite"
        end
        return new{typeof(Q)}(Q)
    end
end

function result_type(d::Mahalanobis, ::Type{T1}, ::Type{T2}) where {T1,T2}
    z = zero(T1) - zero(T2)
    return typeof(sqrt(z * zero(eltype(d.qmat)) * z))
end

function result_type(d::SqMahalanobis, ::Type{T1}, ::Type{T2}) where {T1,T2}
    z = zero(T1) - zero(T2)
    return typeof(z * zero(eltype(d.qmat)) * z)
end

# TODO: merge the following two once we lift the lower bound for julia (above v1.4?)
function (dist::SqMahalanobis)(a::AbstractVector, b::AbstractVector)
    if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end

    Q = dist.qmat
    z = a - b
    return dot(z, Q * z)
end
function (dist::Mahalanobis)(a::AbstractVector, b::AbstractVector)
    if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end

    Q = dist.qmat
    z = a - b
    return sqrt(dot(z, Q * z))
end

sqmahalanobis(a::AbstractVector, b::AbstractVector, Q::AbstractMatrix) = SqMahalanobis(Q)(a, b)
mahalanobis(a::AbstractVector, b::AbstractVector, Q::AbstractMatrix) = Mahalanobis(Q)(a, b)

function _colwise!(dist, r, a, b)
    Q = dist.qmat
    get_colwise_dims(size(Q, 1), r, a, b)
    z = a .- b
    dot_percol!(r, Q * z, z)
end

function colwise!(dist::SqMahalanobis, r::AbstractArray, a::AbstractMatrix, b::AbstractMatrix)
    _colwise!(dist, r, a, b)
end
function colwise!(dist::SqMahalanobis, r::AbstractArray, a::AbstractVector, b::AbstractMatrix)
    _colwise!(dist, r, a, b)
end
function colwise!(dist::SqMahalanobis, r::AbstractArray, a::AbstractMatrix, b::AbstractVector)
    _colwise!(dist, r, a, b)
end

function colwise!(dist::Mahalanobis, r::AbstractArray, a::AbstractMatrix, b::AbstractMatrix)
    sqrt!(_colwise!(dist, r, a, b))
end
function colwise!(dist::Mahalanobis, r::AbstractArray, a::AbstractVector, b::AbstractMatrix)
    sqrt!(_colwise!(dist, r, a, b))
end
function colwise!(dist::Mahalanobis, r::AbstractArray, a::AbstractMatrix, b::AbstractVector)
    sqrt!(_colwise!(dist, r, a, b))
end

function _pairwise!(dist::Union{SqMahalanobis,Mahalanobis}, r::AbstractMatrix, a::AbstractMatrix, b::AbstractMatrix)
    Q = dist.qmat
    m, na, nb = get_pairwise_dims(size(Q, 1), r, a, b)

    Qa = Q * a
    Qb = Q * b
    sa2 = dot_percol(a, Qa)
    sb2 = dot_percol(b, Qb)
    mul!(r, a', Qb)

    for j = 1:nb
        @simd for i = 1:na
            @inbounds r[i, j] = eval_end(dist, max(sa2[i] + sb2[j] - 2 * r[i, j], 0))
        end
    end
    r
end

function _pairwise!(dist::Union{SqMahalanobis,Mahalanobis}, r::AbstractMatrix, a::AbstractMatrix)
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
            @inbounds r[i, j] = eval_end(dist, max(sa2[i] + sa2[j] - 2 * r[i, j], 0))
        end
    end
    r
end

eval_end(::SqMahalanobis, x) = x
eval_end(::Mahalanobis, x) = sqrt(x)
