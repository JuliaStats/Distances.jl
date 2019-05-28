# Weighted metrics


###########################################################
#
#   Metric types
#
###########################################################
const RealAbstractArray{T <: Real} =  AbstractArray{T}


struct WeightedEuclidean{W <: RealAbstractArray} <: Metric
    weights::W
end

struct WeightedSqEuclidean{W <: RealAbstractArray} <: SemiMetric
    weights::W
end

struct WeightedCityblock{W <: RealAbstractArray} <: Metric
    weights::W
end

struct WeightedMinkowski{W <: RealAbstractArray,T <: Real} <: Metric
    weights::W
    p::T
end

struct WeightedHamming{W <: RealAbstractArray} <: Metric
    weights::W
end


const UnionWeightedMetrics{W} = Union{WeightedEuclidean{W},WeightedSqEuclidean{W},WeightedCityblock{W},WeightedMinkowski{W},WeightedHamming{W}}
Base.eltype(x::UnionWeightedMetrics) = eltype(x.weights)
###########################################################
#
# Evaluate
#
###########################################################

function evaluate(dist::UnionWeightedMetrics, a::Number, b::Number)
    eval_end(dist, eval_op(dist, a, b, oneunit(eltype(dist))))
end
result_type(dist::UnionWeightedMetrics, a::AbstractArray, b::AbstractArray) =
    typeof(evaluate(dist, oneunit(eltype(a)), oneunit(eltype(b))))

@inline function eval_start(d::UnionWeightedMetrics, a::AbstractArray, b::AbstractArray)
    zero(result_type(d, a, b))
end
eval_end(d::UnionWeightedMetrics, s) = s



@inline function evaluate(d::UnionWeightedMetrics, a::AbstractArray, b::AbstractArray)
    @boundscheck if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end
    @boundscheck if length(a) != length(d.weights)
        throw(DimensionMismatch("arrays have length $(length(a)) but weights have length $(length(d.weights))."))
    end
    if length(a) == 0
        return zero(result_type(d, a, b))
    end
    @inbounds begin
        s = eval_start(d, a, b)
        if size(a) == size(b)
            @simd for I in eachindex(a, b, d.weights)
                ai = a[I]
                bi = b[I]
                wi = d.weights[I]
                s = eval_reduce(d, s, eval_op(d, ai, bi, wi))
            end
        else
            for (Ia, Ib, Iw) in zip(eachindex(a), eachindex(b), eachindex(d.weights))
                ai = a[Ia]
                bi = b[Ib]
                wi = d.weights[Iw]
                s = eval_reduce(d, s, eval_op(d, ai, bi, wi))
            end
        end
    end
    return eval_end(d, s)
end

# Squared Euclidean
@inline eval_op(::WeightedSqEuclidean, ai, bi, wi) = abs2(ai - bi) * wi
@inline eval_reduce(::WeightedSqEuclidean, s1, s2) = s1 + s2
wsqeuclidean(a::AbstractArray, b::AbstractArray, w::AbstractArray) = evaluate(WeightedSqEuclidean(w), a, b)

# Weighted Euclidean
@inline eval_op(::WeightedEuclidean, ai, bi, wi) = abs2(ai - bi) * wi
@inline eval_reduce(::WeightedEuclidean, s1, s2) = s1 + s2
@inline eval_end(::WeightedEuclidean, s) = sqrt(s)
weuclidean(a::AbstractArray, b::AbstractArray, w::AbstractArray) = evaluate(WeightedEuclidean(w), a, b)

# City Block
@inline eval_op(::WeightedCityblock, ai, bi, wi) = abs((ai - bi) * wi)
@inline eval_reduce(::WeightedCityblock, s1, s2) = s1 + s2
wcityblock(a::AbstractArray, b::AbstractArray, w::AbstractArray) = evaluate(WeightedCityblock(w), a, b)

# Minkowski
@inline eval_op(dist::WeightedMinkowski, ai, bi, wi) = abs(ai - bi).^dist.p * wi
@inline eval_reduce(::WeightedMinkowski, s1, s2) = s1 + s2
eval_end(dist::WeightedMinkowski, s) = s.^(1 / dist.p)
wminkowski(a::AbstractArray, b::AbstractArray, w::AbstractArray, p::Real) = evaluate(WeightedMinkowski(w, p), a, b)

# WeightedHamming
@inline eval_op(::WeightedHamming, ai, bi, wi) = ai != bi ? wi : zero(eltype(wi))
@inline eval_reduce(::WeightedHamming, s1, s2) = s1 + s2
whamming(a::AbstractArray, b::AbstractArray, w::AbstractArray) = evaluate(WeightedHamming(w), a, b)

###########################################################
#
#   Specialized
#
###########################################################

# SqEuclidean
function _pairwise!(r::AbstractMatrix, dist::WeightedSqEuclidean,
                    a::AbstractMatrix, b::AbstractMatrix)
    w = dist.weights
    m, na, nb = get_pairwise_dims(length(w), r, a, b)

    sa2 = wsumsq_percol(w, a)
    sb2 = wsumsq_percol(w, b)
    mul!(r, a', b .* w)
    for j = 1:nb
        @simd for i = 1:na
            @inbounds r[i, j] = sa2[i] + sb2[j] - 2 * r[i, j]
        end
    end
    r
end
function _pairwise!(r::AbstractMatrix, dist::WeightedSqEuclidean,
                    a::AbstractMatrix)
    w = dist.weights
    m, n = get_pairwise_dims(length(w), r, a)

    sa2 = wsumsq_percol(w, a)
    mul!(r, a', a .* w)

    for j = 1:n
        for i = 1:(j - 1)
            @inbounds r[i, j] = r[j, i]
        end
        @inbounds r[j, j] = 0
        @simd for i = (j + 1):n
            @inbounds r[i, j] = sa2[i] + sa2[j] - 2 * r[i, j]
        end
    end
    r
end

# Euclidean
function colwise!(r::AbstractArray, dist::WeightedEuclidean, a::AbstractMatrix, b::AbstractMatrix)
    sqrt!(colwise!(r, WeightedSqEuclidean(dist.weights), a, b))
end
function colwise!(r::AbstractArray, dist::WeightedEuclidean, a::AbstractVector, b::AbstractMatrix)
    sqrt!(colwise!(r, WeightedSqEuclidean(dist.weights), a, b))
end
function _pairwise!(r::AbstractMatrix, dist::WeightedEuclidean,
                    a::AbstractMatrix, b::AbstractMatrix)
    sqrt!(_pairwise!(r, WeightedSqEuclidean(dist.weights), a, b))
end
function _pairwise!(r::AbstractMatrix, dist::WeightedEuclidean, a::AbstractMatrix)
    sqrt!(_pairwise!(r, WeightedSqEuclidean(dist.weights), a))
end
