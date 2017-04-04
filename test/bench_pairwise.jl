
# Benchmark on pairwise distance evaluation

using Distances
using BenchmarkTools

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1.0

function evaluate_pairwise{T}(::Type{T}, dist, x, y)
    nx = size(x, 2)
    ny = size(y, 2)
    r = Matrix{T}(nx, ny)
    for j = 1:ny
        for i = 1:nx
            r[i, j] = evaluate(dist, x[:, i], y[:, j])
        end
    end
    return r
end

function bench_pairwise_distance(dist, x, y)
    r1 = evaluate(dist, x[:,1], y[:,1])

    # timing
    t0 = @belapsed evaluate_pairwise($(typeof(r1)), $dist, $x, $y)
    t1 = @belapsed pairwise($dist, $x, $y)

    print("| ", typeof(dist).name.name, " |")
    @printf("%9.6fs | %9.6fs | %7.4f |\n", t0, t1, (t0 / t1))
end

m = 100
nx = 200
ny = 250

x = rand(m, nx)
y = rand(m, ny)

p = x
for i = 1:nx
    p[:,i] /= sum(x[:,i])
end

q = y
for i = 1:ny
    q[:,i] /= sum(y[:,i])
end

w = rand(m)
Q = rand(m, m)
Q = Q' * Q

println("|  distance  |  loop  |  pairwise |  gain  |")
println("|----------- | -------| ----------| -------|")

bench_pairwise_distance(SqEuclidean(), x, y)
bench_pairwise_distance(Euclidean(), x, y)
bench_pairwise_distance(Cityblock(), x, y)
bench_pairwise_distance(Chebyshev(), x, y)
bench_pairwise_distance(Minkowski(3.0), x, y)
bench_pairwise_distance(Hamming(), x, y)

bench_pairwise_distance(CosineDist(), x, y)
bench_pairwise_distance(CorrDist(), x, y)
bench_pairwise_distance(ChiSqDist(), x, y)
bench_pairwise_distance(KLDivergence(), p, q)
bench_pairwise_distance(RenyiDivergence(0), p, q)
bench_pairwise_distance(RenyiDivergence(1), p, q)
bench_pairwise_distance(RenyiDivergence(2), p, q)
bench_pairwise_distance(RenyiDivergence(Inf), p, q)
bench_pairwise_distance(JSDivergence(), p, q)

bench_pairwise_distance(BhattacharyyaDist(), x, y)
bench_pairwise_distance(HellingerDist(), x, y)

bench_pairwise_distance(WeightedSqEuclidean(w), x, y)
bench_pairwise_distance(WeightedEuclidean(w), x, y)
bench_pairwise_distance(WeightedCityblock(w), x, y)
bench_pairwise_distance(WeightedMinkowski(w, 3.0), x, y)
bench_pairwise_distance(WeightedHamming(w), x, y)

bench_pairwise_distance(SqMahalanobis(Q), x, y)
bench_pairwise_distance(Mahalanobis(Q), x, y)
