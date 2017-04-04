
# Benchmark on column-wise distance evaluation

using Distances
using BenchmarkTools

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1.0

function evaluate_colwise{T}(::Type{T}, dist, x, y)
    n = size(x, 2)
    r = Vector{T}(n)
    for j = 1:n
        r[j] = evaluate(dist, x[:, j], y[:, j])
    end
    return r
end

function bench_colwise_distance(dist, x, y)
    r1 = evaluate(dist, x[:,1], y[:,1])

    # timing
    t0 = @belapsed evaluate_colwise($(typeof(r1)), $dist, $x, $y)
    t1 = @belapsed colwise($dist, $x, $y)
    print("| ", typeof(dist).name.name, " |")
    @printf("%9.6fs | %9.6fs | %7.4f |\n", t0, t1, (t0 / t1))
end

m = 200
n = 10000

x = rand(m, n)
y = rand(m, n)

p = x
q = y
for i = 1:n
    p[:,i] /= sum(x[:,i])
    q[:,i] /= sum(y[:,i])
end

w = rand(m)

Q = rand(m, m)
Q = Q' * Q

println("|  distance  |  loop  |  colwise  |  gain  |")
println("|----------- | -------| ----------| -------|")

bench_colwise_distance(SqEuclidean(), x, y)
bench_colwise_distance(Euclidean(), x, y)
bench_colwise_distance(Cityblock(), x, y)
bench_colwise_distance(Chebyshev(), x, y)
bench_colwise_distance(Minkowski(3.0), x, y)
bench_colwise_distance(Hamming(), x, y)

bench_colwise_distance(CosineDist(), x, y)
bench_colwise_distance(CorrDist(), x, y)
bench_colwise_distance(ChiSqDist(), x, y)
bench_colwise_distance(KLDivergence(), p, q)
bench_colwise_distance(RenyiDivergence(0), p, q)
bench_colwise_distance(RenyiDivergence(1), p, q)
bench_colwise_distance(RenyiDivergence(2), p, q)
bench_colwise_distance(RenyiDivergence(Inf), p, q)
bench_colwise_distance(JSDivergence(), p, q)

bench_colwise_distance(BhattacharyyaDist(), x, y)
bench_colwise_distance(HellingerDist(), x, y)

bench_colwise_distance(WeightedSqEuclidean(w), x, y)
bench_colwise_distance(WeightedEuclidean(w), x, y)
bench_colwise_distance(WeightedCityblock(w), x, y)
bench_colwise_distance(WeightedMinkowski(w, 3.0), x, y)
bench_colwise_distance(WeightedHamming(w), x, y)

bench_colwise_distance(SqMahalanobis(Q), x, y)
bench_colwise_distance(Mahalanobis(Q), x, y)
