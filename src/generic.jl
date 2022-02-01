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
    result_type(dist, Ta::Type, Tb::Type) -> T
    result_type(dist, a, b) -> T

Infer the result type of metric `dist` with input types `Ta` and `Tb`, or element types
of iterators `a` and `b`.
"""
result_type(dist, a, b) = result_type(dist, _eltype(a), _eltype(b))
result_type(f, a::Type, b::Type) = typeof(f(oneunit(a), oneunit(b))) # don't require `PreMetric` subtyping


_eltype(a) = __eltype(Base.IteratorEltype(a), a)
_eltype(::Type{T}) where {T} = eltype(T) === T ? T : _eltype(eltype(T))
_eltype(::Type{Union{Missing, T}}) where {T} = Union{Missing, T}

__eltype(::Base.HasEltype, a) = _eltype(eltype(a))
__eltype(::Base.EltypeUnknown, a) = _eltype(typeof(first(a)))

# Generic column-wise evaluation

"""
    colwise!(metric::PreMetric, r::AbstractArray, a, b)

Compute distances between corresponding elements of the iterable collections
`a` and `b` according to distance `metric`, and store the result in `r`.

`a` and `b` must have the same number of elements, `r` must be an array of length
`length(a) == length(b)`.
"""
function colwise!(metric::PreMetric, r::AbstractArray, a, b)
    require_one_based_indexing(r)
    n = length(a)
    length(b) == n || throw(DimensionMismatch("iterators have different lengths"))
    length(r) == n || throw(DimensionMismatch("Incorrect size of r."))
    @inbounds for (j, ab) in enumerate(zip(a, b))
        r[j] = metric(ab...)
    end
    r
end

function colwise!(metric::PreMetric, r::AbstractArray, a::AbstractVector, b::AbstractMatrix)
    require_one_based_indexing(r)
    n = size(b, 2)
    length(r) == n || throw(DimensionMismatch("Incorrect size of r."))
    @inbounds for (rj, bj) in enumerate(axes(b, 2))
        r[rj] = metric(a, view(b, :, bj))
    end
    r
end

function colwise!(metric::PreMetric, r::AbstractArray, a::AbstractMatrix, b::AbstractVector)
    require_one_based_indexing(r)
    n = size(a, 2)
    length(r) == n || throw(DimensionMismatch("Incorrect size of r."))
    @inbounds for (rj, aj) in enumerate(axes(a, 2))
        r[rj] = metric(view(a, :, aj), b)
    end
    r
end

"""
    colwise!(metric::PreMetric, r::AbstractArray,
             a::AbstractMatrix, b::AbstractMatrix)
    colwise!(metric::PreMetric, r::AbstractArray,
             a::AbstractVector, b::AbstractMatrix)
    colwise!(metric::PreMetric, r::AbstractArray,
             a::AbstractMatrix, b::AbstractVector)

Compute distances between each corresponding columns of `a` and `b` according
to distance `metric`, and store the result in `r`. Exactly one of `a` or `b`
can be a vector, in which case the distance between that vector and all columns
of the other matrix are computed.

`a` and `b` must have the same number of columns if neither of the two is a
vector. `r` must be an array of length `maximum(size(a, 2), size(b, 2))`.

!!! note
    If both `a` and `b` are vectors, the generic, iterator-based method of
    `colwise` applies.
"""
function colwise!(metric::PreMetric, r::AbstractArray, a::AbstractMatrix, b::AbstractMatrix)
    require_one_based_indexing(r, a, b)
    n = get_common_ncols(a, b)
    length(r) == n || throw(DimensionMismatch("Incorrect size of r."))
    @inbounds for j in 1:n
        r[j] = metric(view(a, :, j), view(b, :, j))
    end
    r
end

"""
    colwise(metric::PreMetric, a, b)

Compute distances between corresponding elements of the iterable collections
`a` and `b` according to distance `metric`.

`a` and `b` must have the same number of elements (`length(a) == length(b)`).
"""
function colwise(metric::PreMetric, a, b)
    n = get_common_length(a, b)
    r = Vector{result_type(metric, a, b)}(undef, n)
    colwise!(metric, r, a, b)
end

"""
    colwise(metric::PreMetric, a::AbstractMatrix, b::AbstractMatrix)
    colwise(metric::PreMetric, a::AbstractVector, b::AbstractMatrix)
    colwise(metric::PreMetric, a::AbstractMatrix, b::AbstractVector)

Compute distances between corresponding columns of `a` and `b` according to
distance `metric`. Exactly one of `a` or `b` can be a vector, in which case the
distance between that vector and all columns of the other matrix are computed.

`a` and `b` must have the same number of columns if neither of the two is a
vector.

!!! note
    If both `a` and `b` are vectors, the generic, iterator-based method of
    `colwise` applies.
"""
function colwise(metric::PreMetric, a::AbstractMatrix, b::AbstractMatrix)
    n = get_common_ncols(a, b)
    r = Vector{result_type(metric, a, b)}(undef, n)
    colwise!(metric, r, a, b)
end

function colwise(metric::PreMetric, a::AbstractVector, b::AbstractMatrix)
    n = size(b, 2)
    r = Vector{result_type(metric, a, b)}(undef, n)
    colwise!(metric, r, a, b)
end

function colwise(metric::PreMetric, a::AbstractMatrix, b::AbstractVector)
    n = size(a, 2)
    r = Vector{result_type(metric, a, b)}(undef, n)
    colwise!(metric, r, a, b)
end


# Generic pairwise evaluation

function _pairwise!(metric::PreMetric, r::AbstractMatrix, a, b=a)
    require_one_based_indexing(r)
    na = length(a)
    nb = length(b)
    size(r) == (na, nb) || throw(DimensionMismatch("Incorrect size of r."))
    @inbounds for (j, bj) in enumerate(b), (i, ai) in enumerate(a)
        r[i, j] = metric(ai, bj)
    end
    r
end

function _pairwise!(metric::PreMetric, r::AbstractMatrix,
                    a::AbstractMatrix, b::AbstractMatrix=a)
    require_one_based_indexing(r, a, b)
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

function _pairwise!(metric::SemiMetric, r::AbstractMatrix, a)
    require_one_based_indexing(r)
    n = length(a)
    size(r) == (n, n) || throw(DimensionMismatch("Incorrect size of r."))
    @inbounds for (j, aj) in enumerate(a), (i, ai) in enumerate(a)
        r[i, j] = if i > j
            metric(ai, aj)
        elseif i == j
            zero(eltype(r))
        else
            r[j, i]
        end
    end
    r
end

function _pairwise!(metric::SemiMetric, r::AbstractMatrix, a::AbstractMatrix)
    require_one_based_indexing(r)
    n = size(a, 2)
    size(r) == (n, n) || throw(DimensionMismatch("Incorrect size of r."))
    @inbounds for j = 1:n
        for i = 1:(j - 1)
            r[i, j] = r[j, i]   # leveraging the symmetry of SemiMetric
        end
        r[j, j] = zero(eltype(r))
        aj = view(a, :, j)
        for i = (j + 1):n
            r[i, j] = metric(view(a, :, i), aj)
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
    pairwise!(metric::PreMetric, r::AbstractMatrix,
              a::AbstractMatrix, b::AbstractMatrix=a; dims)

Compute distances between each pair of rows (if `dims=1`) or columns (if `dims=2`)
in `a` and `b` according to distance `metric`, and store the result in `r`.
If a single matrix `a` is provided, compute distances between its rows or columns.

`a` and `b` must have the same numbers of columns if `dims=1`, or of rows if `dims=2`.
`r` must be a matrix with size `size(a, dims) × size(b, dims)`.
"""
function pairwise!(metric::PreMetric, r::AbstractMatrix,
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
        _pairwise!(metric, r, permutedims(a), permutedims(b))
    else
        _pairwise!(metric, r, a, b)
    end
end

function pairwise!(metric::PreMetric, r::AbstractMatrix, a::AbstractMatrix;
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
        _pairwise!(metric, r, permutedims(a))
    else
        _pairwise!(metric, r, a)
    end
end

"""
    pairwise!(metric::PreMetric, r::AbstractMatrix, a, b=a)

Compute distances between each element of collection `a` and each element of
collection `b` according to distance `metric`, and store the result in `r`.
If a single iterable `a` is provided, compute distances between its elements.

`r` must be a matrix with size `length(a) × length(b)`.
"""
pairwise!(metric::PreMetric, r::AbstractMatrix, a, b) = _pairwise!(metric, r, a, b)
pairwise!(metric::PreMetric, r::AbstractMatrix, a) = _pairwise!(metric, r, a)

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
    pairwise!(metric, r, a, b, dims=dims)
end

function pairwise(metric::PreMetric, a::AbstractMatrix;
                  dims::Union{Nothing,Integer}=nothing)
    dims = deprecated_dims(dims)
    dims in (1, 2) || throw(ArgumentError("dims should be 1 or 2 (got $dims)"))
    n = size(a, dims)
    r = Matrix{result_type(metric, a, a)}(undef, n, n)
    pairwise!(metric, r, a, dims=dims)
end

"""
    pairwise(metric::PreMetric, a, b=a)

Compute distances between each element of collection `a` and each element of
collection `b` according to distance `metric`. If a single iterable `a` is
provided, compute distances between its elements.
"""
function pairwise(metric::PreMetric, a, b)
    m = length(a)
    n = length(b)
    r = Matrix{result_type(metric, a, b)}(undef, m, n)
    _pairwise!(metric, r, a, b)
end

function pairwise(metric::PreMetric, a)
    n = length(a)
    r = Matrix{result_type(metric, a, a)}(undef, n, n)
    _pairwise!(metric, r, a)
end
