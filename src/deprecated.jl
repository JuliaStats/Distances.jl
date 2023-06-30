Base.@deprecate pairwise!(r::AbstractMatrix, dist::PreMetric, a) pairwise!(dist, r, a)
Base.@deprecate pairwise!(r::AbstractMatrix, dist::PreMetric, a, b) pairwise!(dist, r, a, b)

Base.@deprecate pairwise!(
    r::AbstractMatrix, dist::PreMetric, a::AbstractMatrix;
    dims::Union{Nothing,Integer}=nothing
) pairwise!(dist, r, a; dims=dims)
Base.@deprecate pairwise!(
    r::AbstractMatrix, dist::PreMetric, a::AbstractMatrix, b::AbstractMatrix;
    dims::Union{Nothing,Integer}=nothing
) pairwise!(dist, r, a, b; dims=dims)

Base.@deprecate colwise!(r::AbstractArray, dist::PreMetric, a, b) colwise!(dist, r, a, b)

# docstrings for deprecated methods
@doc """
    pairwise!(r::AbstractMatrix, dist::PreMetric, a)

Same as `pairwise!(dist, r, a)`.

!!! warning
    Since this alternative syntax is deprecated and will be removed in a future release of
    Distances.jl, its use is discouraged. Please call `pairwise!(dist, r, a)` instead.
""" pairwise!(r::AbstractMatrix, dist::PreMetric, a)
@doc """
    pairwise!(r::AbstractMatrix, dist::PreMetric, a, b)

Same as `pairwise!(dist, r, a, b)`.

!!! warning
    Since this alternative syntax is deprecated and will be removed in a future release of
    Distances.jl, its use is discouraged. Please call `pairwise!(dist, r, a, b)` instead.
""" pairwise!(r::AbstractMatrix, dist::PreMetric, a, b)

@doc """
    pairwise!(r::AbstractMatrix, dist::PreMetric, a::AbstractMatrix; dims)

Same as `pairwise!(dist, r, a; dims)`.

    !!! warning
    Since this alternative syntax is deprecated and will be removed in a future release of
    Distances.jl, its use is discouraged. Please call `pairwise!(dist, r, a; dims)` instead.
""" pairwise!(
    r::AbstractMatrix, dist::PreMetric, a::AbstractMatrix;
    dims::Union{Nothing,Integer}
)
@doc """
    pairwise!(r::AbstractMatrix, dist::PreMetric, a::AbstractMatrix, b::AbstractMatrix; dims)

Same as `pairwise!(dist, r, a, b; dims)`.

!!! warning
    Since this alternative syntax is deprecated and will be removed in a future release of
    Distances.jl, its use is discouraged. Please call `pairwise!(dist, r, a, b; dims)`
    instead.
""" pairwise!(
    r::AbstractMatrix, dist::PreMetric, a::AbstractMatrix, b::AbstractMatrix;
    dims::Union{Nothing,Integer}
)

@doc """
    colwise!(r::AbstractArray, dist::PreMetric, a, b)

Same as `colwise!(dist, r, a, b)`.

!!! warning
    Since this alternative syntax is deprecated and will be removed in a future release of
    Distances.jl, its use is discouraged. Please call `colwise!(dist, r, a, b)` instead.
""" colwise!(r::AbstractArray, dist::PreMetric, a, b)
