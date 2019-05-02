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


# Generic functions

result_type(::PreMetric, ::AbstractArray, ::AbstractArray) = Float64


# Generic column-wise evaluation

function colwise!(r::AbstractArray, metric::PreMetric, a::AbstractVector, b::AbstractMatrix)
    n = size(b, 2)
    length(r) == n || throw(DimensionMismatch("Incorrect size of r."))
    @inbounds for j = 1:n
        r[j] = evaluate(metric, a, view(b, :, j))
    end
    r
end

function colwise!(r::AbstractArray, metric::PreMetric, a::AbstractMatrix, b::AbstractVector)
    n = size(a, 2)
    length(r) == n || throw(DimensionMismatch("Incorrect size of r."))
    @inbounds for j = 1:n
        r[j] = evaluate(metric, view(a, :, j), b)
    end
    r
end

function colwise!(r::AbstractArray, metric::PreMetric, a::AbstractMatrix, b::AbstractMatrix)
    n = get_common_ncols(a, b)
    length(r) == n || throw(DimensionMismatch("Incorrect size of r."))
    @inbounds for j = 1:n
        r[j] = evaluate(metric, view(a, :, j), view(b, :, j))
    end
    r
end

function colwise!(r::AbstractArray, metric::SemiMetric, a::AbstractMatrix, b::AbstractVector)
    colwise!(r, metric, b, a)
end

function colwise!(r::AbstractArray, metric::PreMetric, a::AbstractVector, b::AbstractVector)
    length(r) == 1 || throw(DimensionMismatch("Incorrect size of r."))
    r[1] = evaluate(metric, a, b)
    r
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

colwise(metric::PreMetric, a::AbstractVector, b::AbstractVector) = [evaluate(metric, a, b)]


# Generic pairwise evaluation

function _pairwise!(r::AbstractMatrix, metric::PreMetric,
                    a::AbstractMatrix, b::AbstractMatrix=a)
    na = size(a, 2)
    nb = size(b, 2)
    size(r) == (na, nb) || throw(DimensionMismatch("Incorrect size of r."))
    @inbounds for j = 1:size(b, 2)
        bj = view(b, :, j)
        for i = 1:size(a, 2)
            r[i, j] = evaluate(metric, view(a, :, i), bj)
        end
    end
    r
end

function _pairwise!(r::AbstractMatrix, metric::SemiMetric, a::AbstractMatrix)
    n = size(a, 2)
    size(r) == (n, n) || throw(DimensionMismatch("Incorrect size of r."))
    @inbounds for j = 1:n
        aj = view(a, :, j)
        for i = (j + 1):n
            r[i, j] = evaluate(metric, view(a, :, i), aj)
        end
        r[j, j] = 0
        for i = 1:(j - 1)
            r[i, j] = r[j, i]   # leveraging the symmetry of SemiMetric
        end
    end
    r
end

function _pairwise!(r::AbstractMatrix, metric::PreMetric,
                    a::AbstractVector, b::AbstractVector)
    r[1,1] = evaluate(metric, a, b)
    r
end

function _pairwise!(r::AbstractMatrix, metric::PreMetric,
                    a::AbstractVector)
    _pairwise!(r, metric, a, a)
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
                   a::AbstractVector, b::AbstractVector;
                   dims::Union{Nothing,Integer}=nothing)
    dims = deprecated_dims(dims)
    dims in (1, 2) || throw(ArgumentError("dims should be 1 or 2 (got $dims)"))
    if dims == 1
        r_sz = (length(a), length(b))
    else
        r_sz = (1, 1)
    end
    size(r) == r_sz || throw(DimensionMismatch("Incorrect size of r (got $(size(r)), expected $(r_sz)."))
    if dims == 1
        _pairwise!(r, metric, transpose(a), transpose(b))
    else
        _pairwise!(r, metric, a, b)
    end
end

function pairwise!(r::AbstractMatrix, metric::PreMetric,
                   a::AbstractVector;
                   dims::Union{Nothing,Integer}=nothing)
    pairwise!(r, metric, a, a; dims=dims)
end

"""
    pairwise(metric::PreMetric, a::AbstractMatrix, b::AbstractMatrix=a; dims)

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

function pairwise(metric::PreMetric, a::AbstractVector, b::AbstractVector;
                  dims::Union{Nothing, Integer}=nothing)
    dims = deprecated_dims(dims)
    dims in (1, 2) || throw(ArgumentError("dims should be 1 or 2 (got $dims)"))
    if dims == 1
        m, n = (length(a), length(b))
    else
        m, n = (1, 1)
    end
    r = Matrix{result_type(metric, a, b)}(undef, m, n)
    pairwise!(r, metric, a, b, dims=dims)
end

function pairwise(metric::PreMetric, a::AbstractVector; dims::Union{Nothing, Integer}=nothing)
    pairwise(metric, a, a; dims=dims)
end
