__precompile__()

module Distances

using Compat

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

    MeanAbsDeviation,
    MeanSqDeviation,
    RMSDeviation,
    NormRMSDeviation,
    CVRMSDeviation,

    # convenient functions
    euclidean,
    sqeuclidean,
    cityblock,
    jaccard,
    rogerstanimoto,
    chebyshev,
    minkowski,
    mahalanobis,

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

    meanad,
    msd,
    rmsd,
    nrmsd,
    cvrmsd

include("common.jl")
include("generic.jl")
include("metrics.jl")
include("wmetrics.jl")
include("mahalanobis.jl")
include("bhattacharyya.jl")

end # module end
