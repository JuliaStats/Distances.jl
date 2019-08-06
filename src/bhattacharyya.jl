# Bhattacharyya distances. Much like for KLDivergence we assume the vectors to
# be compared are probability distributions, frequencies or counts rather than
# vectors of samples. Pre-calc accordingly if you have samples.

struct BhattacharyyaDist <: SemiMetric end

struct HellingerDist <: Metric end

# Bhattacharyya coefficient

function bhattacharyya_coeff(a::AbstractVector{T}, b::AbstractVector{T}) where {T <: Number}
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

bhattacharyya_coeff(a::T, b::T) where {T <: Number} = throw("Bhattacharyya coefficient cannot be calculated for scalars")

# Faster pair- and column-wise versions TBD...


# Bhattacharyya distance
(::BhattacharyyaDist)(a::AbstractVector{T}, b::AbstractVector{T}) where {T <: Number} = -log(bhattacharyya_coeff(a, b))
(::BhattacharyyaDist)(a::T, b::T) where {T <: Number} = throw("Bhattacharyya distance cannot be calculated for scalars")
bhattacharyya(a, b) = BhattacharyyaDist()(a, b)

# Hellinger distance
(::HellingerDist)(a::AbstractVector{T}, b::AbstractVector{T}) where {T <: Number} = sqrt(1 - bhattacharyya_coeff(a, b))
(::HellingerDist)(a::T, b::T) where {T <: Number} = throw("Hellinger distance cannot be calculated for scalars")
hellinger(a, b) = HellingerDist()(a, b)
