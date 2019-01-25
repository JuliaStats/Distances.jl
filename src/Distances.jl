module Distances

using LinearAlgebra
using Statistics

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
    PeriodicEuclidean,

    Haversine,

    MeanAbsDeviation,
    MeanSqDeviation,
    RMSDeviation,
    NormRMSDeviation,
    Bregman,

    # convenient functions
    euclidean,
    sqeuclidean,
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
    peuclidean,

    haversine,

    meanad,
    msd,
    rmsd,
    nrmsd

include("common.jl")
include("generic.jl")
include("metrics.jl")
include("wmetrics.jl")
include("haversine.jl")
include("mahalanobis.jl")
include("bhattacharyya.jl")
include("bregman.jl")

end # module end
