module DistancesStatisticsExt

using Statistics: mean

using Distances

# export CorrDist, corr_dist

# CorrDist is excluded from `UnionMetrics`
# struct CorrDist <: SemiMetric end

# CorrDist
_centralize(x) = x .- mean(x)
(::CorrDist)(a, b) = CosineDist()(_centralize(a), _centralize(b))
(::CorrDist)(a::Number, b::Number) = CosineDist()(zero(mean(a)), zero(mean(b)))

# CorrDist
# This part of codes is accelerated because:
# 1. It calls the accelerated `_pairwise` specilization for CosineDist
# 2. pre-calculated `_centralize_colwise` avoids four times of redundant computations
#    of `_centralize` -- ~4x speed up
_centralize_colwise(x::AbstractMatrix) = x .- mean(x, dims=1)
_pairwise!(r::AbstractMatrix, ::CorrDist, a::AbstractMatrix, b::AbstractMatrix) =
    _pairwise!(r, CosineDist(), _centralize_colwise(a), _centralize_colwise(b))
_pairwise!(r::AbstractMatrix, ::CorrDist, a::AbstractMatrix) =
    _pairwise!(r, CosineDist(), _centralize_colwise(a))

end # module Statistics
