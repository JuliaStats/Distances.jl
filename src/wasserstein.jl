#####
##### Wasserstein distance
#####

export Wasserstein

# TODO: Make concrete
struct Wasserstein{T<:AbstractFloat} <: PreMetric
    u_weights::Union{AbstractArray{T}, Nothing}
    v_weights::Union{AbstractArray{T}, Nothing}
end

Wasserstein(u_weights, v_weights) = Wasserstein{eltype(u_weights)}(u_weights, v_weights)

(w::Wasserstein)(u, v) = wasserstein(u, v, w.u_weights, w.v_weights)

evaluate(dist::Wasserstein, u, v) = dist(u,v)

abstract type Side end
struct Left <: Side end
struct Right <: Side end

"""
    pysearchsorted(a,b;side="left")

Based on accepted answer in:
    https://stackoverflow.com/questions/55339848/julia-vectorized-version-of-searchsorted
"""
pysearchsorted(a,b,::Left) = searchsortedfirst.(Ref(a),b) .- 1
pysearchsorted(a,b,::Right) = searchsortedlast.(Ref(a),b)

function compute_integral(u_cdf, v_cdf, deltas, p)
    if p == 1
        return sum(abs.(u_cdf - v_cdf) .* deltas)
    end
    if p == 2
        return sqrt(sum((u_cdf - v_cdf).^2 .* deltas))
    end
    return sum(abs.(u_cdf - v_cdf).^p .* deltas)^(1/p)
end

function _cdf_distance(p, u_values, v_values, u_weights=nothing, v_weights=nothing)
    _validate_distribution(u_values, u_weights)
    _validate_distribution(v_values, v_weights)

    u_sorter = sortperm(u_values)
    v_sorter = sortperm(v_values)

    all_values = vcat(u_values, v_values)
    sort!(all_values)

    # Compute the differences between pairs of successive values of u and v.
    deltas = diff(all_values)

    # Get the respective positions of the values of u and v among the values of
    # both distributions.
    u_cdf_indices = pysearchsorted(u_values[u_sorter],all_values[1:end-1], Right())
    v_cdf_indices = pysearchsorted(v_values[v_sorter],all_values[1:end-1], Right())

    # Calculate the CDFs of u and v using their weights, if specified.
    if u_weights == nothing
        u_cdf = (u_cdf_indices) / length(u_values)
    else
        u_sorted_cumweights = vcat([0], cumsum(u_weights[u_sorter]))
        u_cdf = u_sorted_cumweights[u_cdf_indices.+1] / u_sorted_cumweights[end]
    end

    if v_weights == nothing
        v_cdf = (v_cdf_indices) / length(v_values)
    else
        v_sorted_cumweights = vcat([0], cumsum(v_weights[v_sorter]))
        v_cdf = v_sorted_cumweights[v_cdf_indices.+1] / v_sorted_cumweights[end]
    end

    # Compute the value of the integral based on the CDFs.
    return compute_integral(u_cdf, v_cdf, deltas, p)
end

function _validate_distribution(vals, weights)
    # Validate the value array.
    length(vals) == 0 && throw(ArgumentError("Distribution can't be empty."))
    # Validate the weight array, if specified.
    if weights ≠ nothing
        if length(weights) != length(vals)
            throw(DimensionMismatch("Value and weight array-likes for the same empirical distribution must be of the same size."))
        end
        any(weights .< 0) && throw(ArgumentError("All weights must be non-negative."))
        if !(0 < sum(weights) < Inf)
            throw(ArgumentError("Weight array-like sum must be positive and finite. Set as None for an equal distribution of weight."))
        end
    end
    return nothing
end

"""
    wasserstein(u_values, v_values, u_weights=nothing, v_weights=nothing)

Compute the first Wasserstein distance between two 1D distributions.
This distance is also known as the earth mover's distance, since it can be
seen as the minimum amount of "work" required to transform ``u`` into
``v``, where "work" is measured as the amount of distribution weight
that must be moved, multiplied by the distance it has to be moved.

 - `u_values` Values observed in the (empirical) distribution.
 - `v_values` Values observed in the (empirical) distribution.

 - `u_weights` Weight for each value.
 - `v_weights` Weight for each value.

If the weight sum differs from 1, it must still be positive
and finite so that the weights can be normalized to sum to 1.
"""
function wasserstein(u_values, v_values, u_weights=nothing, v_weights=nothing)
    return _cdf_distance(1, u_values, v_values, u_weights, v_weights)
end