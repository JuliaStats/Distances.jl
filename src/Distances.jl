module Distances

using LinearAlgebra
using Statistics: mean
import StatsAPI: pairwise, pairwise!

export
    # generic types/functions
    PreMetric,
    SemiMetric,
    Metric,

    # generic functions
    result_type,
    colwise,
    pairwise,
    colwise!,
    pairwise!,
    evaluate,

    # distance classes
    Euclidean,
    SqEuclidean,
    PeriodicEuclidean,
    Cityblock,
    TotalVariation,
    Chebyshev,
    Minkowski,
    Jaccard,
    BrayCurtis,
    RogersTanimoto,

    Hamming,
    CosineDist,
    CorrDist,
    ChiSqDist,
    KLDivergence,
    GenKLDivergence,
    JSDivergence,
    RenyiDivergence,
    SpanNormDist,

    WeightedEuclidean,
    WeightedSqEuclidean,
    WeightedCityblock,
    WeightedMinkowski,
    WeightedHamming,
    SqMahalanobis,
    Mahalanobis,
    BhattacharyyaDist,
    HellingerDist,

    Haversine,
    SphericalAngle,

    MeanAbsDeviation,
    MeanSqDeviation,
    RMSDeviation,
    NormRMSDeviation,
    Bregman,

    # convenient functions
    euclidean,
    sqeuclidean,
    peuclidean,
    cityblock,
    totalvariation,
    jaccard,
    braycurtis,
    rogerstanimoto,
    chebyshev,
    minkowski,

    hamming,
    cosine_dist,
    corr_dist,
    chisq_dist,
    kl_divergence,
    gkl_divergence,
    js_divergence,
    renyi_divergence,
    spannorm_dist,

    weuclidean,
    wsqeuclidean,
    wcityblock,
    wminkowski,
    whamming,
    sqmahalanobis,
    mahalanobis,
    bhattacharyya,
    hellinger,
    bregman,

    haversine,
    spherical_angle,

    meanad,
    msd,
    rmsd,
    nrmsd

if VERSION < v"1.2-"
    import Base: has_offset_axes
    require_one_based_indexing(A...) =
        !has_offset_axes(A...) ||
            throw(ArgumentError("offset arrays are not supported but got an array with index other than 1"))
else
    import Base: require_one_based_indexing
end

include("common.jl")
include("generic.jl")
include("metrics.jl")
include("haversine.jl")
include("mahalanobis.jl")
include("bhattacharyya.jl")
include("bregman.jl")

include("deprecated.jl")

@static if !isdefined(Base, :get_extension)
    include("../ext/DistancesSparseArraysExt.jl")
end

end # module end
