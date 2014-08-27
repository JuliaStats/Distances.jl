module Distances

using ArrayViews

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

    Hamming,
    CosineDist,
    CorrDist,
    ChiSqDist,
    KLDivergence,
    JSDivergence,
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
    JSMetric,

    # convenient functions
    euclidean,
    sqeuclidean,
    cityblock,
    chebyshev,
    minkowski,
    mahalanobis,

    hamming,
    cosine_dist,
    corr_dist,
    chisq_dist,
    kl_divergence,
    js_divergence,
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
    js_metric

include("common.jl")
include("generic.jl")
include("metrics.jl")
include("wmetrics.jl")
include("mahalanobis.jl")
include("bhattacharyya.jl")
include("jsmetric.jl")

end # module end


