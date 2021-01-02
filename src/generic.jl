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

@deprecate colwise!(r::AbstractArray, metric::PreMetric, a::AbstractVector, b::AbstractMatrix) broadcast!(metric, r, Ref(a), eachcol(b))
@deprecate colwise!(r::AbstractArray, metric::PreMetric, a::AbstractMatrix, b::AbstractVector) broadcast!(metric, r, eachcol(a), Ref(b))
@deprecate colwise!(r::AbstractArray, metric::PreMetric, a::AbstractMatrix, b::AbstractMatrix) broadcast!(metric, r, eachcol(a), eachcol(b))
@deprecate colwise!(r::AbstractArray, metric::PreMetric, a, b) broadcast!(metric, r, a, b)

@deprecate colwise(metric::PreMetric, a::AbstractVector, b::AbstractMatrix) broadcast(metric, Ref(a), eachcol(b))
@deprecate colwise(metric::PreMetric, a::AbstractMatrix, b::AbstractVector) broadcast(metric, eachcol(a), Ref(b))
@deprecate colwise(metric::PreMetric, a::AbstractMatrix, b::AbstractMatrix) broadcast(metric, eachcol(a), eachcol(b))
@deprecate colwise(metric::PreMetric, a, b) broadcast(metric, a, b)


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
