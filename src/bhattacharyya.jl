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

    @simd for i in eachindex(a, b)
        @inbounds ai = a[i]
        @inbounds bi = b[i]
        sqab += sqrt(ai * bi)
        asum += ai
        bsum += bi
    end
    return sqab, asum, bsum
end

@inline function _bhattacharyya_coeff(a::SparseVectorUnion{<:Number}, b::SparseVectorUnion{<:Number})
    anzind = nonzeroinds(a)
    bnzind = nonzeroinds(b)
    anzval = nonzeros(a)
    bnzval = nonzeros(b)
    ma = nnz(a)
    mb = nnz(b)

    ia = 1; ib = 1
    s = zero(typeof(sqrt(oneunit(eltype(a))*oneunit(eltype(b)))))
    @inbounds while ia <= ma && ib <= mb
        ja = anzind[ia]
        jb = bnzind[ib]
        if ja == jb
            s += sqrt(anzval[ia] * bnzval[ib])
            ia += 1; ib += 1
        elseif ja < jb
            ia += 1
        else
            ib += 1
        end
    end
    # efficient method for sum for SparseVectorView is missing
    return s, sum(anzval), sum(bnzval)
end

# Faster pair- and column-wise versions TBD...


# Bhattacharyya distance
(::BhattacharyyaDist)(a, b) = -log(bhattacharyya_coeff(a, b))
const bhattacharyya = BhattacharyyaDist()

# Hellinger distance
(::HellingerDist)(a, b) = sqrt(1 - bhattacharyya_coeff(a, b))
const hellinger = HellingerDist()
