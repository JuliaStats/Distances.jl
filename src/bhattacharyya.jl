# Bhattacharyya distances. Much like for KLDivergence we assume the vectors to
# be compared are probability distributions, frequencies or counts rather than
# vectors of samples. Pre-calc accordingly if you have samples.

type BhattacharyyaDist <: SemiMetric end

type HellingerDist <: Metric end


# Bhattacharyya coefficient

function bhattacharyya_coeff{T<:Number}(a::AbstractVector{T}, b::AbstractVector{T})
    if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end

    n = length(a)
    sqab = zero(T)
    # We must normalize since we cannot assume that the vectors are normalized to probability vectors.
    asum = zero(T)
    bsum = zero(T)

    @simd for i = 1:n
        @inbounds ai = a[i]
        @inbounds bi = b[i]
        sqab += sqrt(ai * bi)
        asum += ai
        bsum += bi
    end

    sqab / sqrt(asum * bsum)
end

bhattacharyya_coeff{T <: Number}(a::T, b::T) = throw("Bhattacharyya coefficient cannot be calculated for scalars")

# Faster pair- and column-wise versions TBD...


# Bhattacharyya distance
evaluate{T<:Number}(dist::BhattacharyyaDist, a::AbstractVector{T}, b::AbstractVector{T}) = -log(bhattacharyya_coeff(a, b))
bhattacharyya(a::AbstractVector, b::AbstractVector) = evaluate(BhattacharyyaDist(), a, b)
evaluate{T <: Number}(dist::BhattacharyyaDist, a::T, b::T) = throw("Bhattacharyya distance cannot be calculated for scalars")
bhattacharyya{T <: Number}(a::T, b::T) = evaluate(BhattacharyyaDist(), a, b)

# Hellinger distance
evaluate{T<:Number}(dist::HellingerDist, a::AbstractVector{T}, b::AbstractVector{T}) = sqrt(1 - bhattacharyya_coeff(a, b))
hellinger(a::AbstractVector, b::AbstractVector) = evaluate(HellingerDist(), a, b)
evaluate{T <: Number}(dist::HellingerDist, a::T, b::T) = throw("Hellinger distance cannot be calculated for scalars")
hellinger{T <: Number}(a::T, b::T) = evaluate(HellingerDist(), a, b)
