# Bhattacharyya distances. Much like for KLDivergence we assume the vectors to
# be compared are probability distributions, frequencies or counts rather than
# vectors of samples. Pre-calc accordingly if you have samples.

struct BhattacharyyaDist <: SemiMetric end

struct HellingerDist <: Metric end


# Bhattacharyya coefficient

function bhattacharyya_coeff(a, b)
    n = length(a)
    if n != length(b)
        throw(DimensionMismatch("first argument has length $n which does not match the length of the second, $(length(b))."))
    end
    sqab, asum, bsum = _bhattacharyya_coeff(a, b)
    # We must normalize since we cannot assume that the vectors are normalized to probability vectors.
    return sqab / sqrt(asum * bsum)
end

@inline function _bhattacharyya_coeff(a, b)
    Ta = _eltype(a)
    Tb = _eltype(b)
    T = typeof(sqrt(zero(promote_type(Ta, Tb))))
    sqab = zero(T)
    asum = zero(Ta)
    bsum = zero(Tb)

    for (ai, bi) in zip(a, b)
        sqab += sqrt(ai * bi)
        asum += ai
        bsum += bi
    end
    return sqab, asum, bsum
end
@inline function _bhattacharyya_coeff(a::AbstractVector{Ta}, b::AbstractVector{Tb}) where {Ta<:Number,Tb<:Number}
    T = typeof(sqrt(oneunit(Ta)*oneunit(Tb)))
    sqab = zero(T)
    asum = zero(Ta)
    bsum = zero(Tb)

    @simd for i = 1:length(a)
        @inbounds ai = a[i]
        @inbounds bi = b[i]
        sqab += sqrt(ai * bi)
        asum += ai
        bsum += bi
    end
    return sqab, asum, bsum
end

# Faster pair- and column-wise versions TBD...


# Bhattacharyya distance
(::BhattacharyyaDist)(a, b) = -log(bhattacharyya_coeff(a, b))
bhattacharyya(a, b) = BhattacharyyaDist()(a, b)

# Hellinger distance
(::HellingerDist)(a, b) = sqrt(1 - bhattacharyya_coeff(a, b))
hellinger(a, b) = HellingerDist()(a, b)
