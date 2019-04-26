# Common utilities

###########################################################
#
#   helper functions for dimension checking
#
###########################################################

function get_common_ncols(a::AbstractMatrix{<:Number}, b::AbstractMatrix{<:Number})
    na = size(a, 2)
    size(b, 2) == na || throw(DimensionMismatch("The number of columns in a and b must match."))
    return na
end

function get_colwise_dims(r::AbstractArray{<:Number}, a::AbstractMatrix{<:Number}, b::AbstractMatrix{<:Number})
    size(a) == size(b) || throw(DimensionMismatch("The sizes of a and b must match."))
    length(r) == size(a, 2) || throw(DimensionMismatch("Incorrect size of r."))
    return size(a)
end

function get_colwise_dims(r::AbstractArray{<:Number}, a::AbstractVector{<:Number}, b::AbstractMatrix{<:Number})
    length(a) == size(b, 1) ||
        throw(DimensionMismatch("The length of a must match the number of rows in b."))
    length(r) == size(b, 2) || throw(DimensionMismatch("Incorrect size of r."))
    return size(b)
end

function get_colwise_dims(r::AbstractArray{<:Number}, a::AbstractMatrix{<:Number}, b::AbstractVector{<:Number})
    size(a, 1) == length(b) ||
        throw(DimensionMismatch("The length of b must match the number of rows in a."))
    length(r) == size(a, 2) || throw(DimensionMismatch("Incorrect size of r."))
    return size(a)
end

function get_pairwise_dims(r::AbstractMatrix{<:Number}, a::AbstractMatrix{<:Number}, b::AbstractMatrix{<:Number})
    ma, na = size(a)
    mb, nb = size(b)
    ma == mb || throw(DimensionMismatch("The numbers of rows or columns in a and b must match."))
    size(r) == (na, nb) || throw(DimensionMismatch("Incorrect size of r."))
    return (ma, na, nb)
end

function get_pairwise_dims(r::AbstractMatrix{<:Number}, a::AbstractMatrix{<:Number})
    m, n = size(a)
    size(r) == (n, n) || throw(DimensionMismatch("Incorrect size of r."))
    return (m, n)
end


# for metrics with fixed dimension (e.g. weighted metrics)

function get_colwise_dims(d::Int, r::AbstractArray{<:Number}, a::AbstractMatrix{<:Number}, b::AbstractMatrix{<:Number})
    size(a, 1) == size(b, 1) == d ||
        throw(DimensionMismatch("Incorrect vector dimensions."))
    length(r) == size(a, 2) || throw(DimensionMismatch("Incorrect size of r."))
    return size(a)
end

function get_colwise_dims(d::Int, r::AbstractArray{<:Number}, a::AbstractVector{<:Number}, b::AbstractMatrix{<:Number})
    length(a) == size(b, 1) == d ||
        throw(DimensionMismatch("Incorrect vector dimensions."))
    length(r) == size(b, 2) || throw(DimensionMismatch("Incorrect size of r."))
    return size(b)
end

function get_colwise_dims(d::Int, r::AbstractArray{<:Number}, a::AbstractMatrix{<:Number}, b::AbstractVector{<:Number})
    size(a, 1) == length(b) == d ||
        throw(DimensionMismatch("Incorrect vector dimensions."))
    length(r) == size(a, 2) || throw(DimensionMismatch("Incorrect size of r."))
    return size(a)
end

function get_pairwise_dims(d::Int, r::AbstractMatrix{<:Number}, a::AbstractMatrix{<:Number}, b::AbstractMatrix{<:Number})
    na = size(a, 2)
    nb = size(b, 2)
    size(a, 1) == size(b, 1) == d || throw(DimensionMismatch("Incorrect vector dimensions."))
    size(r) == (na, nb) || throw(DimensionMismatch("Incorrect size of r."))
    return (d, na, nb)
end

function get_pairwise_dims(d::Int, r::AbstractMatrix{<:Number}, a::AbstractMatrix{<:Number})
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

function sqrt!(a::AbstractArray{<:Number})
    @simd for i in eachindex(a)
        @inbounds a[i] = sqrt(a[i])
    end
    a
end

function sumsq_percol(a::AbstractMatrix{T}) where {T<:Number}
    m = size(a, 1)
    n = size(a, 2)
    r = Vector{T}(undef, n)
    for j = 1:n
        aj = view(a, :, j)
        r[j] = dot(aj, aj)
    end
    return r
end

function wsumsq_percol(w::AbstractArray{T1}, a::AbstractMatrix{T2}) where {T1<:Number, T2<:Number}
    m = size(a, 1)
    n = size(a, 2)
    T = typeof(one(T1) * one(T2))
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

function dot_percol!(r::AbstractArray{<:Number}, a::AbstractMatrix{<:Number}, b::AbstractMatrix{<:Number})
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

function dot_percol(a::AbstractMatrix{T1}, b::AbstractMatrix{T2}) where {T1<:Number, T2<:Number}
    T = promote_type(T1, T2) 
    dot_percol!(Vector{T}(undef, size(a, 2)), a, b)
end
