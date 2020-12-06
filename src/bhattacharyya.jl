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
    sqab = zero(typeof(sqrt(zero(T))))
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
function bhattacharyya_coeff(a, b)
    T = typeof(sqrt(zero(promote_type(_eltype(a), _eltype(b)))))
    n = length(a)
    if n != length(b)
        throw(DimensionMismatch("first argument has length $n which does not match the length of the second, $(length(b))."))
    end

    sqab = zero(T)
    # We must normalize since we cannot assume that the vectors are normalized to probability vectors.
    asum = zero(T)
    bsum = zero(T)

    for (ai, bi) in zip(a, b)
        sqab += sqrt(ai * bi)
        asum += ai
        bsum += bi
    end

    return sqab / sqrt(asum * bsum)
end

# Faster pair- and column-wise versions TBD...


# Bhattacharyya distance
(::BhattacharyyaDist)(a, b) = -log(bhattacharyya_coeff(a, b))
(::BhattacharyyaDist)(::Number, ::Number) = throw("Bhattacharyya distance cannot be calculated for scalars")
bhattacharyya(a, b) = BhattacharyyaDist()(a, b)

# Hellinger distance
(::HellingerDist)(a, b) = sqrt(1 - bhattacharyya_coeff(a, b))
(::HellingerDist)(::Number, ::Number) = throw("Hellinger distance cannot be calculated for scalars")
hellinger(a, b) = HellingerDist()(a, b)
