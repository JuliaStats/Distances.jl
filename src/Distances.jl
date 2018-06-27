__precompile__()

module Distances

using LinearAlgebra

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

    MeanAbsDeviation,
    MeanSqDeviation,
    RMSDeviation,
    NormRMSDeviation,

    # convenient functions
    euclidean,
    sqeuclidean,
    cityblock,
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

end # module end
