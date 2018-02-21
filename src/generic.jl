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

function colwise(metric::PreMetric, a::AbstractMatrix, b::AbstractMatrix)
    n = get_common_ncols(a, b)
    r = Vector{result_type(metric, a, b)}(uninitialized, n)
    colwise!(r, metric, a, b)
end

function colwise(metric::PreMetric, a::AbstractVector, b::AbstractMatrix)
    n = size(b, 2)
    r = Vector{result_type(metric, a, b)}(uninitialized, n)
    colwise!(r, metric, a, b)
end

function colwise(metric::PreMetric, a::AbstractMatrix, b::AbstractVector)
    n = size(a, 2)
    r = Vector{result_type(metric, a, b)}(uninitialized, n)
    colwise!(r, metric, a, b)
end


# Generic pairwise evaluation

function pairwise!(r::AbstractMatrix, metric::PreMetric, a::AbstractMatrix, b::AbstractMatrix)
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

function pairwise!(r::AbstractMatrix, metric::PreMetric, a::AbstractMatrix)
    pairwise!(r, metric, a, a)
end

function pairwise!(r::AbstractMatrix, metric::SemiMetric, a::AbstractMatrix)
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

function pairwise(metric::PreMetric, a::AbstractMatrix, b::AbstractMatrix)
    m = size(a, 2)
    n = size(b, 2)
    r = Matrix{result_type(metric, a, b)}(uninitialized, m, n)
    pairwise!(r, metric, a, b)
end

function pairwise(metric::PreMetric, a::AbstractMatrix)
    n = size(a, 2)
    r = Matrix{result_type(metric, a, a)}(uninitialized, n, n)
    pairwise!(r, metric, a)
end
