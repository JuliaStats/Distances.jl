# Generic concepts and functions

# a premetric is a function d that satisfies:
#
#   d(x, y) >= 0
#   d(x, x) = 0
#
abstract type PreMetric end

# a semimetric is a function d that satisfies:
#
#   d(x, y) >= 0
#   d(x, x) = 0
#   d(x, y) = d(y, x)
#
abstract type SemiMetric <: PreMetric end

# a metric is a semimetric that satisfies triangle inequality:
#
#   d(x, y) + d(y, z) >= d(x, z)
#
abstract type Metric <: SemiMetric end

evaluate(dist::PreMetric, a, b) = dist(a, b)

# Generic functions

"""
    result_type(dist::PreMetric, Ta::Type, Tb::Type) -> T
    result_type(dist::PreMetric, a::AbstractArray, b::AbstractArray) -> T

Infer the result type of metric `dist` with input type `Ta` and `Tb`, or input
data `a` and `b`.
"""
result_type(::PreMetric, ::Type, ::Type) = Float64 # fallback
result_type(dist::PreMetric, a::AbstractArray, b::AbstractArray) = result_type(dist, eltype(a), eltype(b))
result_type(dist::PreMetric, a::AbstractArray{<:AbstractArray}, b::AbstractArray{<:AbstractArray}) = result_type(dist, eltype(eltype(a)), eltype(eltype(b)))

# Generic column-wise evaluation

function colwise!(r::AbstractArray, metric::PreMetric, a::AbstractVector, b::AbstractMatrix)
    n = size(b, 2)
    length(r) == n || throw(DimensionMismatch("Incorrect size of r."))
    @inbounds for j = 1:n
        r[j] = metric(a, view(b, :, j))
    end
    r
end

function colwise!(r::AbstractArray, metric::PreMetric, a::AbstractMatrix, b::AbstractVector)
    n = size(a, 2)
    length(r) == n || throw(DimensionMismatch("Incorrect size of r."))
    @inbounds for j = 1:n
        r[j] = metric(view(a, :, j), b)
    end
    r
end

function colwise!(r::AbstractArray, metric::PreMetric, a::AbstractMatrix, b::AbstractMatrix)
    n = get_common_ncols(a, b)
    length(r) == n || throw(DimensionMismatch("Incorrect size of r."))
    @inbounds for j = 1:n
        r[j] = metric(view(a, :, j), view(b, :, j))
    end
    r
end

function colwise!(r::AbstractArray, metric::SemiMetric, a::AbstractMatrix, b::AbstractVector)
    colwise!(r, metric, b, a)
end

function colwise(metric::PreMetric, a::AbstractMatrix, b::AbstractMatrix)
    n = get_common_ncols(a, b)
    r = Vector{result_type(metric, a, b)}(undef, n)
    colwise!(r, metric, a, b)
end

function colwise(metric::PreMetric, a::AbstractVector, b::AbstractMatrix)
    n = size(b, 2)
    r = Vector{result_type(metric, a, b)}(undef, n)
    colwise!(r, metric, a, b)
end

function colwise(metric::PreMetric, a::AbstractMatrix, b::AbstractVector)
    n = size(a, 2)
    r = Vector{result_type(metric, a, b)}(undef, n)
    colwise!(r, metric, a, b)
end


# Generic pairwise evaluation

function _pairwise!(r::AbstractMatrix, metric::PreMetric,
                    a::AbstractMatrix, b::AbstractMatrix)
    na = size(a, 2)
    nb = size(b, 2)
    size(r) == (na, nb) || throw(DimensionMismatch("Incorrect size of r."))
    @inbounds for j = 1:size(b, 2)
        bj = view(b, :, j)
        for i = 1:size(a, 2)
            r[i, j] = metric(view(a, :, i), bj)
        end
    end
    r
end

function _pairwise!(r::AbstractMatrix, metric::SemiMetric, a::AbstractMatrix)
    n = size(a, 2)
    size(r) == (n, n) || throw(DimensionMismatch("Incorrect size of r."))
    @inbounds for j = 1:n
        aj = view(a, :, j)
        for i = 1:(j - 1)
            r[i, j] = r[j, i]   # leveraging the symmetry of SemiMetric
        end
        r[j, j] = 0
        for i = (j + 1):n
            r[i, j] = metric(view(a, :, i), aj)
        end
    end
    r
end

function _pairwise!(r::AbstractMatrix, metric::PreMetric,
                    a::AbstractVector, b::AbstractVector)
    require_one_based_indexing(r)
    na = length(a)
    nb = length(b)
    size(r) == (na, nb) || throw(DimensionMismatch("Incorrect size of r."))
    @inbounds for (j, bj) = enumerate(b)
        for (i, ai) = enumerate(a)
            r[i, j] = metric(ai, bj)
        end
    end
    r
end

function _pairwise!(r::AbstractMatrix, metric::SemiMetric, a::AbstractVector)
    require_one_based_indexing(a, r)
    n = length(a)
    size(r) == (n, n) || throw(DimensionMismatch("Incorrect size of r."))
    @inbounds for j = 1:n
        for i = 1:(j - 1)
            r[i, j] = r[j, i]   # leveraging the symmetry of SemiMetric
        end
        r[j, j] = 0
        for i = (j + 1):n
            r[i, j] = metric(a[i], a[j])
        end
    end
    r
end


function deprecated_dims(dims::Union{Nothing,Integer})
    if dims === nothing
        Base.depwarn("implicit `dims=2` argument now has to be passed explicitly " *
                     "to specify that distances between columns should be computed",
                     :pairwise!)
        return 2
    else
        return dims
    end
end

"""
    pairwise!(r::AbstractMatrix, metric::PreMetric,
              a::AbstractMatrix, b::AbstractMatrix=a; dims)
    pairwise!(r::AbstractMatrix, metric::PreMetric,
              a::AbstractVector, b::AbstractVector=a)

Compute distances between each pair of rows (if `dims=1`) or columns (if `dims=2`)
in `a` and `b` according to distance `metric`, and store the result in `r`.
If a single matrix `a` is provided, compute distances between its rows or columns.

`a` and `b` must have the same numbers of columns if `dims=1`, or of rows if `dims=2`.
`r` must be a square matrix with size `size(a, dims) == size(b, dims)`.
"""
function pairwise!(r::AbstractMatrix, metric::PreMetric,
                   a::AbstractMatrix, b::AbstractMatrix;
                   dims::Union{Nothing,Integer}=nothing)
    dims = deprecated_dims(dims)
    dims in (1, 2) || throw(ArgumentError("dims should be 1 or 2 (got $dims)"))
    if dims == 1
        na, ma = size(a)
        nb, mb = size(b)
        ma == mb || throw(DimensionMismatch("The numbers of columns in a and b " *
                                            "must match (got $ma and $mb)."))
    else
        ma, na = size(a)
        mb, nb = size(b)
        ma == mb || throw(DimensionMismatch("The numbers of rows in a and b " *
                                            "must match (got $ma and $mb)."))
    end
    size(r) == (na, nb) ||
        throw(DimensionMismatch("Incorrect size of r (got $(size(r)), expected $((na, nb)))."))
    if dims == 1
        _pairwise!(r, metric, transpose(a), transpose(b))
    else
        _pairwise!(r, metric, a, b)
    end
end

function pairwise!(r::AbstractMatrix, metric::PreMetric, a::AbstractMatrix;
                   dims::Union{Nothing,Integer}=nothing)
    dims = deprecated_dims(dims)
    dims in (1, 2) || throw(ArgumentError("dims should be 1 or 2 (got $dims)"))
    if dims == 1
        n, m = size(a)
    else
        m, n = size(a)
    end
    size(r) == (n, n) ||
        throw(DimensionMismatch("Incorrect size of r (got $(size(r)), expected $((n, n)))."))
    if dims == 1
        _pairwise!(r, metric, transpose(a))
    else
        _pairwise!(r, metric, a)
    end
end

function pairwise!(r::AbstractMatrix, metric::PreMetric,
                   a::AbstractVector, b::AbstractVector)
    na = length(a)
    nb = length(b)
    size(r) == (na, nb) ||
        throw(DimensionMismatch("Incorrect size of r (got $(size(r)), expected $((na, nb)))."))
    _pairwise!(r, metric, a, b)
end

function pairwise!(r::AbstractMatrix, metric::PreMetric, a::AbstractVector)
    n = length(a)
    size(r) == (n, n) ||
    throw(DimensionMismatch("Incorrect size of r (got $(size(r)), expected $((n, n)))."))
    _pairwise!(r, metric, a)
end

"""
    pairwise(metric::PreMetric, a::AbstractMatrix, b::AbstractMatrix=a; dims)
    pairwise(metric::PreMetric, a::AbstractVector, b::AbstractVector=a; dims)

Compute distances between each pair of rows (if `dims=1`) or columns (if `dims=2`)
in `a` and `b` according to distance `metric`. If a single matrix `a` is provided,
compute distances between its rows or columns.

`a` and `b` must have the same numbers of columns if `dims=1`, or of rows if `dims=2`.
"""
function pairwise(metric::PreMetric, a::AbstractMatrix, b::AbstractMatrix;
                  dims::Union{Nothing,Integer}=nothing)
    dims = deprecated_dims(dims)
    dims in (1, 2) || throw(ArgumentError("dims should be 1 or 2 (got $dims)"))
    m = size(a, dims)
    n = size(b, dims)
    r = Matrix{result_type(metric, a, b)}(undef, m, n)
    pairwise!(r, metric, a, b, dims=dims)
end

function pairwise(metric::PreMetric, a::AbstractMatrix;
                  dims::Union{Nothing,Integer}=nothing)
    dims = deprecated_dims(dims)
    dims in (1, 2) || throw(ArgumentError("dims should be 1 or 2 (got $dims)"))
    n = size(a, dims)
    r = Matrix{result_type(metric, a, a)}(undef, n, n)
    pairwise!(r, metric, a, dims=dims)
end


function pairwise(metric::PreMetric, a::AbstractVector, b::AbstractVector)
    m = length(a)
    n = length(b)
    r = Matrix{result_type(metric, a, b)}(undef, m, n)
    pairwise!(r, metric, a, b)
end


function pairwise(metric::PreMetric, a::AbstractVector)
    n = length(a)
    r = Matrix{result_type(metric, a, a)}(undef, n, n)
    pairwise!(r, metric, a)
end
