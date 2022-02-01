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
