module DistancesChainRulesCoreExt

using Distances

import ChainRulesCore

const CRC = ChainRulesCore

## SqEuclidean

function CRC.rrule(
    ::CRC.RuleConfig{>:CRC.HasReverseMode},
    dist::SqEuclidean,
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real}
)
    Ω = dist(x, y)

    function SqEuclidean_pullback(ΔΩ)
        x̄ = (2 * CRC.unthunk(ΔΩ)) .* (x .- y)
        return CRC.NoTangent(), x̄, -x̄
    end

    return Ω, SqEuclidean_pullback
end

function CRC.rrule(::CRC.RuleConfig{>:CRC.HasReverseMode}, ::typeof(colwise), dist::SqEuclidean, X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real})
    Ω = colwise(dist, X, Y)

    function colwise_SqEuclidean_pullback(ΔΩ)
        X̄ = 2 .* CRC.unthunk(ΔΩ)' .* (X .- Y)
        return CRC.NoTangent(), CRC.NoTangent(), X̄, -X̄
    end

    return Ω, colwise_SqEuclidean_pullback
end

function CRC.rrule(::CRC.RuleConfig{>:CRC.HasReverseMode}, ::typeof(pairwise), dist::SqEuclidean, X::AbstractMatrix{<:Real}; dims::Union{Nothing,Integer}=nothing)
    dims = Distances.deprecated_dims(dims)
    dims in (1, 2) || throw(ArgumentError("dims should be 1 or 2 (got $dims)"))
    Ω = pairwise(dist, X; dims=dims)

    function pairwise_SqEuclidean_X_pullback(ΔΩ)
        Δ = CRC.unthunk(ΔΩ)
        A = Δ .+ transpose(Δ)
        X̄ = if dims == 1
            2 .* (sum(A; dims=2) .* X .- A * X)
        else
            2 .* (X .* sum(A; dims=1) .- X * A)
        end
        return CRC.NoTangent(), CRC.NoTangent(), X̄
    end

    return Ω, pairwise_SqEuclidean_X_pullback
end

function CRC.rrule(::CRC.RuleConfig{>:CRC.HasReverseMode}, ::typeof(pairwise), dist::SqEuclidean, X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}; dims::Union{Nothing,Integer}=nothing)
    dims = Distances.deprecated_dims(dims)
    dims in (1, 2) || throw(ArgumentError("dims should be 1 or 2 (got $dims)"))
    Ω = pairwise(dist, X, Y; dims=dims)

    function pairwise_SqEuclidean_X_Y_pullback(ΔΩ)
        Δ = CRC.unthunk(ΔΩ)
        Δt = transpose(Δ)
        X̄ = if dims == 1
            2 .* (sum(Δ; dims=2) .* X .- Δ * Y)
        else
            2 .* (X .* sum(Δt; dims=1) .- Y * Δt)
        end
        Ȳ = if dims == 1
            2 .* (sum(Δt; dims=2) .* Y .- Δt * X)
        else
            2 .* (Y .* sum(Δ; dims=1) .- X * Δ)
        end
        return CRC.NoTangent(), CRC.NoTangent(), X̄, Ȳ
    end

    return Ω, pairwise_SqEuclidean_X_Y_pullback
end

## Euclidean

_normalize(x::Real, nrm::Real) = iszero(nrm) && !isnan(x) ? one(x / nrm) : x / nrm

function CRC.rrule(::CRC.RuleConfig{>:CRC.HasReverseMode}, dist::Euclidean, x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    Ω = dist(x, y)

    function Euclidean_pullback(ΔΩ)
        x̄ = _normalize(CRC.unthunk(ΔΩ), Ω) .* (x .- y)
        return CRC.NoTangent(), x̄, -x̄
    end

    return Ω, Euclidean_pullback
end

function CRC.rrule(::CRC.RuleConfig{>:CRC.HasReverseMode}, ::typeof(colwise), dist::Euclidean, X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real})
    Ω = colwise(dist, X, Y)

    function colwise_Euclidean_pullback(ΔΩ)
        X̄ = _normalize.(CRC.unthunk(ΔΩ)', Ω') .* (X .- Y)
        return CRC.NoTangent(), CRC.NoTangent(), X̄, -X̄
    end

    return Ω, colwise_Euclidean_pullback
end

function CRC.rrule(::CRC.RuleConfig{>:CRC.HasReverseMode}, ::typeof(pairwise), dist::Euclidean, X::AbstractMatrix{<:Real}; dims::Union{Nothing,Integer}=nothing)
    dims = Distances.deprecated_dims(dims)
    dims in (1, 2) || throw(ArgumentError("dims should be 1 or 2 (got $dims)"))
    Ω = pairwise(dist, X; dims=dims)

    function pairwise_Euclidean_X_pullback(ΔΩ)
        Δ = CRC.unthunk(ΔΩ)
        A = _normalize.(Δ .+ transpose(Δ), Ω)
        X̄ = if dims == 1
            sum(A; dims=2) .* X .- A * X
        else
            X .* sum(A; dims=1) .- X * A
        end
        return CRC.NoTangent(), CRC.NoTangent(), X̄
    end

    return Ω, pairwise_Euclidean_X_pullback
end

function CRC.rrule(::CRC.RuleConfig{>:CRC.HasReverseMode}, ::typeof(pairwise), dist::Euclidean, X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}; dims::Union{Nothing,Integer}=nothing)
    dims = Distances.deprecated_dims(dims)
    dims in (1, 2) || throw(ArgumentError("dims should be 1 or 2 (got $dims)"))
    Ω = pairwise(dist, X, Y; dims=dims)

    function pairwise_Euclidean_X_Y_pullback(ΔΩ)
        Δ = _normalize.(CRC.unthunk(ΔΩ), Ω)
        Δt = transpose(Δ)
        X̄ = if dims == 1
            sum(Δ; dims=2) .* X .- Δ * Y
        else
            X .* sum(Δt; dims=1) .- Y * Δt
        end
        Ȳ = if dims == 1
            sum(Δt; dims=2) .* Y .- Δt * X
        else
            Y .* sum(Δ; dims=1) .- X * Δ
        end
        return CRC.NoTangent(), CRC.NoTangent(), X̄, Ȳ
    end

    return Ω, pairwise_Euclidean_X_Y_pullback
end

end # module