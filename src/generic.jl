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

__eltype(::Base.HasEltype, a) = _eltype(eltype(a))
__eltype(::Base.EltypeUnknown, a) = _eltype(typeof(first(a)))

# Generic column-wise evaluation

@deprecate colwise!(r::AbstractArray, metric::PreMetric, a, b) zipwise!(r, metric, a, b)

"""
    zipwise!(r::AbstractArray, metric::PreMetric, a, b)

Compute distances between corresponding elements of the iterable collections
`a` and `b` according to distance `metric`, and store the result in `r`. Exactly
one of `a` or `b` can be a length-1 iterator, in which case the distance between
that object and all objects of the respective other iterator are computed.

`r` must be an array of length `maximum(length(a), length(b))`.
"""
function zipwise!(r::AbstractArray, metric::PreMetric, a, b)
    na = length(a)
    nb = length(b)
    if na == nb
        length(r) == na || throw(DimensionMismatch("Incorrect size of r."))
        @inbounds for (j, ab) in enumerate(zip(a, b))
            r[j] = metric(ab...)
        end
    elseif na == 1
        length(r) == nb || throw(DimensionMismatch("Incorrect size of r."))
        @inbounds for (j, ab) in enumerate(zip(Iterators.repeated(a, nb), b))
            r[j] = metric(ab...)
        end
    elseif nb == 1
        length(r) == nb || throw(DimensionMismatch("Incorrect size of r."))
        @inbounds for (j, ab) in enumerate(zip(a, Iterators.repeated(b, na)))
            r[j] = metric(ab...)
        end
    else
        throw(DimensionMismatch("The lengths of a and b must match."))
    end
    return r
end

# legacy methods, would be better to enforce iterators more strongly (colwise vs rowwise)
function zipwise!(r::AbstractArray, metric::PreMetric, a::AbstractVector, b::AbstractMatrix)
    zipwise!(r, metric, Ref(a), eachcol(b))
end
function zipwise!(r::AbstractArray, metric::PreMetric, a::AbstractMatrix, b::AbstractVector)
    zipwise!(r, metric, eachcol(a), Ref(b))
end
function zipwise!(r::AbstractArray, metric::PreMetric, a::AbstractMatrix, b::AbstractMatrix)
    zipwise!(r, metric, eachcol(a), eachcol(b))
end

@deprecate colwise(metric::PreMetric, a, b) zipwise(metric, a, b)

"""
    zipwise(metric::PreMetric, a, b)

Compute distances between corresponding elements of the iterable collections
`a` and `b` according to distance `metric`. Exactly one of `a` or `b` can be a
length-1 iterator, in which case the distance between that object and all objects
of the respective other iterator are computed.
"""
function zipwise(metric::PreMetric, a, b)
    na = length(a)
    nb = length(b)
    if na == nb
        r = Vector{result_type(metric, a, b)}(undef, na)
        return zipwise!(r, metric, a, b)
    elseif na == 1
        r = Vector{result_type(metric, a, b)}(undef, nb)
        return zipwise!(r, metric, Iterators.repeated(a, nb), b)
    elseif nb == 1
        r = Vector{result_type(metric, a, b)}(undef, na)
        return zipwise!(r, metric, a, Iterators.repeated(b, na))
    else
        throw(DimensionMismatch("The lengths of a and b must match."))
    end
end

# legacy methods, would be better to enforce iterators more strongly (colwise vs rowwise)
"""
    zipwise(metric::PreMetric, a::AbstractMatrix, b::AbstractMatrix)
    zipwise(metric::PreMetric, a::AbstractVector, b::AbstractMatrix)
    zipwise(metric::PreMetric, a::AbstractMatrix, b::AbstractVector)

Compute distances between corresponding columns of `a` and `b` according to
distance `metric`. Exactly one of `a` or `b` can be a vector, in which case the
distance between that vector and all columns of the other matrix are computed.

`a` and `b` must have the same number of columns if neither of the two is a
vector.

!!! note
    If both `a` and `b` are vectors, the generic, iterator-based method of
    `zipwise` applies.
"""
function zipwise(metric::PreMetric, a::AbstractMatrix, b::AbstractMatrix)
    zipwise(metric, eachcol(a), eachcol(b))
end
function zipwise(metric::PreMetric, a::AbstractMatrix, b::AbstractVector)
    zipwise(metric, eachcol(a), Iterators.repeated(b, size(a, 2)))
end
function zipwise(metric::PreMetric, a::AbstractVector, b::AbstractMatrix)
    zipwise(metric, Iterators.repeated(a, size(b, 2)), eachcol(b))
end


# Generic pairwise evaluation

function _pairwise!(r::AbstractMatrix, metric::PreMetric, a, b=a)
    require_one_based_indexing(r)
    na = length(a)
    nb = length(b)
    size(r) == (na, nb) || throw(DimensionMismatch("Incorrect size of r."))
    @inbounds for (j, bj) in enumerate(b), (i, ai) in enumerate(a)
        r[i, j] = metric(ai, bj)
    end
    r
end

function _pairwise!(r::AbstractMatrix, metric::SemiMetric, a)
    require_one_based_indexing(r)
    n = length(a)
    size(r) == (n, n) || throw(DimensionMismatch("Incorrect size of r."))
    @inbounds for (j, aj) in enumerate(a), (i, ai) in enumerate(a)
        r[i, j] = if i > j
            metric(ai, aj)
        elseif i == j
            0
        else
            r[j, i]
        end
    end
    r
end

"""
    pairwise!(r::AbstractMatrix, metric::PreMetric, a, b=a)

Compute distances between each element of collection `a` and each element of
collection `b` according to distance `metric`, and store the result in `r`.
If a single iterable `a` is provided, compute distances between its elements.

`r` must be a matrix with size `length(a) × length(b)`.
"""
pairwise!(r::AbstractMatrix, metric::PreMetric, a, b) = _pairwise!(r, metric, a, b)
pairwise!(r::AbstractMatrix, metric::PreMetric, a) = _pairwise!(r, metric, a)

"""
    pairwise!(r::AbstractMatrix, metric::PreMetric,
              a::AbstractMatrix, b::AbstractMatrix=a; dims)

Compute distances between each pair of rows (if `dims=1`) or columns (if `dims=2`)
in `a` and `b` according to distance `metric`, and store the result in `r`.
If a single matrix `a` is provided, compute distances between its rows or columns.

`a` and `b` must have the same numbers of columns if `dims=1`, or of rows if `dims=2`.
`r` must be a matrix with size `size(a, dims) × size(b, dims)`.
"""
function pairwise!(r::AbstractMatrix, metric::PreMetric, a::AbstractMatrix, b::AbstractMatrix; dims)
    Base.depwarn("Replace calls with matrices by eachrow (dims=1) and eachcol (dims=2), respectively", :pairwise!)
    if dims == 1
        return pairwise!(r, metric, eachrow(a), eachrow(b))
    elseif dims == 1
        return pairwise!(r, metric, eachcol(a), eachcol(b))
    else
        throw(ArgumentError("dims should be 1 or 2 (got $dims)"))
    end
end

function pairwise!(r::AbstractMatrix, metric::PreMetric, a::AbstractMatrix; dims)
    Base.depwarn("Replace calls with matrices by eachrow (dims=1) and eachcol (dims=2), respectively", :pairwise!)
    if dims == 1
        return pairwise!(r, metric, eachrow(a))
    elseif dims == 2
        return pairwise!(r, metric, eachcol(a))
    else
        throw(ArgumentError("dims should be 1 or 2 (got $dims)"))
    end
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
    _pairwise!(r, metric, a, b)
end

function pairwise(metric::PreMetric, a)
    n = length(a)
    r = Matrix{result_type(metric, a, a)}(undef, n, n)
    _pairwise!(r, metric, a)
end

# legacy methods
"""
    pairwise(metric::PreMetric, a::AbstractMatrix, b::AbstractMatrix=a; dims)

Compute distances between each pair of rows (if `dims=1`) or columns (if `dims=2`)
in `a` and `b` according to distance `metric`. If a single matrix `a` is provided,
compute distances between its rows or columns.

`a` and `b` must have the same numbers of columns if `dims=1`, or of rows if `dims=2`.
"""
function pairwise(metric::PreMetric, a::AbstractMatrix, b::AbstractMatrix; dims)
    Base.depwarn("Replace calls with matrices by eachrow (dims=1) and eachcol (dims=2), respectively", :pairwise)
    if dims == 1
        return pairwise(metric, eachrow(a), eachrow(b))
    elseif dims == 2
        return pairwise(metric, eachcol(a), eachcol(b))
    else
        throw(ArgumentError("dims should be 1 or 2 (got $dims)"))
    end
end
function pairwise(metric::PreMetric, a::AbstractMatrix; dims)
    Base.depwarn("Replace calls with a matrix by eachrow (dims=1) and eachcol (dims=2), respectively", :pairwise)
    if dims == 1
        return pairwise(metric, eachrow(a))
    elseif dims == 2
        return pairwise(metric, eachcol(a))
    else
        throw(ArgumentError("dims should be 1 or 2 (got $dims)"))
    end
end
