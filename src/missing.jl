"""
Exclude any missing indices from being included the wrappped distance metric.
"""
struct SkipMissing{D<:PreMetric} <: PreMetric
    d::D
end

result_type(dist::SkipMissing, a::Type, b::Type) = result_type(dist.d, a, b)

# Always fallback to the internal metric behaviour
(dist::SkipMissing)(a, b) = dist.d(a, b)

# Special case vector arguments where we can mask out incomplete cases
function (dist::SkipMissing)(a::AbstractVector, b::AbstractVector)
    require_one_based_indexing(a)
    require_one_based_indexing(b)
    n = length(a)
    length(b) == n || throw(DimensionMismatch("a and b have different lengths"))

    mask = BitVector(undef, n)
    @inbounds for i in 1:n
        mask[i] = !(ismissing(a[i]) || ismissing(b[i]))
    end

    # Calling `_evaluate` allows us to also mask metric parameters like weights or periods
    # I don't think this can be generalized to user defined metric types though without knowing
    # what the parameters mean.
    # NTOE: We call disallowmissings to avoid downstream type promotion issues.
    if dist.d isa UnionMetrics
        params = parameters(dist.d)

        return _evaluate(
            dist.d,
            disallowmissing(view(a, mask)),
            disallowmissing(view(b, mask)),
            isnothing(params) ? params : view(params, mask),
        )
    else
        return dist.d(
            disallowmissing(view(a, mask)),
            disallowmissing(view(b, mask)),
        )
    end
end

# Convenience function
skipmissing(dist::PreMetric, args...) = SkipMissing(dist)(args...)