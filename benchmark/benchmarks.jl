using BenchmarkTools
using Distances

const SUITE = BenchmarkGroup()

function create_distances(w, Q)
    dists = [
        SqEuclidean(),
        Euclidean(),
        Cityblock(),
        Chebyshev(),
        Minkowski(3.0),
        Hamming(),

        CosineDist(),
        CorrDist(),
        ChiSqDist(),

        BhattacharyyaDist(),
        HellingerDist(),

        WeightedSqEuclidean(w),
        WeightedEuclidean(w),
        WeightedCityblock(w),
        WeightedMinkowski(w, 3.0),
        WeightedHamming(w),

        SqMahalanobis(Q),
        Mahalanobis(Q),
    ]

    divs = [
        KLDivergence(),
        RenyiDivergence(0),
        RenyiDivergence(1),
        RenyiDivergence(2),
        RenyiDivergence(Inf),
        JSDivergence(),
    ]
    return dists, divs
end

function create_2D_distances()
    dists = [
        Haversine(6371.),
        Ellipsoidal([1.0, 0.5], [π/2])
    ]

    dists
end

function create_3D_distances()
    dists = [
        Ellipsoidal([1.0,0.5,0.5], [π/2,0.0,0.0])
    ]

    dists
end

###########
# Colwise #
###########

SUITE["colwise"] = BenchmarkGroup()

function evaluate_colwise(dist, x, y)
    n = size(x, 2)
    T = typeof(evaluate(dist, x[:, 1], y[:, 1]))
    r = Vector{T}(n)
    for j = 1:n
        r[j] = evaluate(dist, x[:, j], y[:, j])
    end
    return r
end

function add_colwise_benchmarks!(SUITE)

    m = 200
    n = 10000

    x = rand(m, n)
    y = rand(m, n)

    p = x
    q = y
    for i = 1:n
        p[:, i] /= sum(x[:, i])
        q[:, i] /= sum(y[:, i])
    end

    w = rand(m)

    Q = rand(m, m)
    Q = Q' * Q

    _dists, divs = create_distances(w, Q)

    for (dists, (a, b)) in [(_dists, (x,y)), (divs, (p,q))]
        for dist in (dists)
            Tdist = typeof(dist)
            SUITE["colwise"][Tdist] = BenchmarkGroup()
            SUITE["colwise"][Tdist]["loop"]        = @benchmarkable evaluate_colwise($dist, $a, $b)
            SUITE["colwise"][Tdist]["specialized"] = @benchmarkable colwise($dist, $a, $b)
        end
    end

    ####################################
    # Distances defined for 2D vectors #
    ####################################

    x2 = rand(2, n)
    y2 = rand(2, n)

    _dists = create_2D_distances()

    for dist in _dists
        Tdist = typeof(dist)
        SUITE["colwise"][Tdist] = BenchmarkGroup()
        SUITE["colwise"][Tdist]["loop"]        = @benchmarkable evaluate_colwise($dist, $x2, $y2)
        SUITE["colwise"][Tdist]["specialized"] = @benchmarkable colwise($dist, $x2, $y2)
    end

    ####################################
    # Distances defined for 3D vectors #
    ####################################

    x3 = rand(3, n)
    y3 = rand(3, n)

    _dists = create_3D_distances()

    for dist in _dists
        Tdist = typeof(dist)
        SUITE["colwise"][Tdist] = BenchmarkGroup()
        SUITE["colwise"][Tdist]["loop"]        = @benchmarkable evaluate_colwise($dist, $x3, $y3)
        SUITE["colwise"][Tdist]["specialized"] = @benchmarkable colwise($dist, $x3, $y3)
    end
end

add_colwise_benchmarks!(SUITE)


############
# Pairwise #
############

SUITE["pairwise"] = BenchmarkGroup()

function evaluate_pairwise(dist, x, y)
    nx = size(x, 2)
    ny = size(y, 2)
    T = typeof(evaluate(dist, x[:, 1], y[:, 1]))
    r = Matrix{T}(nx, ny)
    for j = 1:ny
        for i = 1:nx
            r[i, j] = evaluate(dist, x[:, i], y[:, j])
        end
    end
    return r
end

function add_pairwise_benchmarks!(SUITE)
    m = 100
    nx = 200
    ny = 250

    x = rand(m, nx)
    y = rand(m, ny)

    p = x
    for i = 1:nx
        p[:, i] /= sum(x[:, i])
    end

    q = y
    for i = 1:ny
        q[:, i] /= sum(y[:, i])
    end

    w = rand(m)
    Q = rand(m, m)
    Q = Q' * Q

    _dists, divs = create_distances(w, Q)

    for (dists, (a, b)) in [(_dists, (x,y)), (divs, (p,q))]
        for dist in (dists)
            Tdist = typeof(dist)
            SUITE["pairwise"][Tdist] = BenchmarkGroup()
            SUITE["pairwise"][Tdist]["loop"]        = @benchmarkable evaluate_pairwise($dist, $a, $b)
            SUITE["pairwise"][Tdist]["specialized"] = @benchmarkable pairwise($dist, $a, $b)
        end
    end

    ####################################
    # Distances defined for 2D vectors #
    ####################################

    x2 = rand(2, nx)
    y2 = rand(2, ny)

    _dists = create_2D_distances()

    for dist in _dists
        Tdist = typeof(dist)
        SUITE["colwise"][Tdist] = BenchmarkGroup()
        SUITE["colwise"][Tdist]["loop"]        = @benchmarkable evaluate_pairwise($dist, $x2, $y2)
        SUITE["colwise"][Tdist]["specialized"] = @benchmarkable pairwise($dist, $x2, $y2)
    end

    ####################################
    # Distances defined for 3D vectors #
    ####################################

    x3 = rand(3, nx)
    y3 = rand(3, ny)

    _dists = create_3D_distances()

    for dist in _dists
        Tdist = typeof(dist)
        SUITE["colwise"][Tdist] = BenchmarkGroup()
        SUITE["colwise"][Tdist]["loop"]        = @benchmarkable evaluate_pairwise($dist, $x3, $y3)
        SUITE["colwise"][Tdist]["specialized"] = @benchmarkable pairwise($dist, $x3, $y3)
    end
end

add_pairwise_benchmarks!(SUITE)
