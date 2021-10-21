# Common utilities

###########################################################
#
#   helper functions for dimension checking
#
###########################################################

function get_common_ncols(a::AbstractMatrix, b::AbstractMatrix)
    na = size(a, 2)
    size(b, 2) == na || throw(DimensionMismatch("The number of columns in a and b must match."))
    return na
end

function get_common_length(a, b)
    n = length(a)
    length(b) == n || throw(DimensionMismatch("The lengths of a and b must match."))
    return n
end

function get_pairwise_dims(r::AbstractMatrix, a::AbstractMatrix, b::AbstractMatrix)
    ma, na = size(a)
    mb, nb = size(b)
    ma == mb || throw(DimensionMismatch("The numbers of rows or columns in a and b must match."))
    size(r) == (na, nb) || throw(DimensionMismatch("Incorrect size of r."))
    return (ma, na, nb)
end

function get_pairwise_dims(r::AbstractMatrix, a::AbstractMatrix)
    m, n = size(a)
    size(r) == (n, n) || throw(DimensionMismatch("Incorrect size of r."))
    return (m, n)
end


# for metrics with fixed dimension (e.g. weighted metrics)

function get_colwise_dims(d::Int, r::AbstractArray, a::AbstractMatrix, b::AbstractMatrix)
    size(a, 1) == size(b, 1) == d ||
        throw(DimensionMismatch("Incorrect vector dimensions."))
    length(r) == size(a, 2) || throw(DimensionMismatch("Incorrect size of r."))
    return size(a)
end

function get_colwise_dims(d::Int, r::AbstractArray, a::AbstractVector, b::AbstractMatrix)
    length(a) == size(b, 1) == d ||
        throw(DimensionMismatch("Incorrect vector dimensions."))
    length(r) == size(b, 2) || throw(DimensionMismatch("Incorrect size of r."))
    return size(b)
end

function get_colwise_dims(d::Int, r::AbstractArray, a::AbstractMatrix, b::AbstractVector)
    size(a, 1) == length(b) == d ||
        throw(DimensionMismatch("Incorrect vector dimensions."))
    length(r) == size(a, 2) || throw(DimensionMismatch("Incorrect size of r."))
    return size(a)
end

function get_pairwise_dims(d::Int, r::AbstractMatrix, a::AbstractMatrix, b::AbstractMatrix)
    na = size(a, 2)
    nb = size(b, 2)
    size(a, 1) == size(b, 1) == d || throw(DimensionMismatch("Incorrect vector dimensions."))
    size(r) == (na, nb) || throw(DimensionMismatch("Incorrect size of r."))
    return (d, na, nb)
end

function get_pairwise_dims(d::Int, r::AbstractMatrix, a::AbstractMatrix)
    n = size(a, 2)
    size(a, 1) == d || throw(DimensionMismatch("Incorrect vector dimensions."))
    size(r) == (n, n) || throw(DimensionMismatch("Incorrect size of r."))
    return (d, n)
end

###########################################################
#
#   calculation
#
###########################################################

function sqrt!(a::AbstractArray)
    @simd for i in eachindex(a)
        @inbounds a[i] = sqrt(a[i])
    end
    a
end

function norm_percol(a::AbstractMatrix{T}) where {T}
    n = size(a, 2)
    r = Vector{float(T)}(undef, n)
    @simd for j in 1:n
        aj = view(a, :, j)
        r[j] = sqrt(dot(aj, aj))
    end
    return r
end

function wsumsq_percol(w::AbstractArray, a::AbstractMatrix)
    m = size(a, 1)
    n = size(a, 2)
    T = typeof(zero(eltype(w)) * abs2(zero(eltype(a))))
    r = Vector{T}(undef, n)
    for j = 1:n
        aj = view(a, :, j)
        s = zero(T)
        @simd for i = 1:m
            @inbounds s += w[i] * abs2(aj[i])
        end
        r[j] = s
    end
    return r
end

function dot_percol!(r::AbstractArray, a::AbstractMatrix, b::AbstractMatrix)
    m = size(a, 1)
    n = size(a, 2)
    size(b) == (m, n) && length(r) == n ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    for j = 1:n
        aj = view(a, :, j)
        bj = view(b, :, j)
        r[j] = dot(aj, bj)
    end
    return r
end

dot_percol(a::AbstractMatrix, b::AbstractMatrix) = dot_percol!(Vector{Float64}(undef, size(a, 2)), a, b)
