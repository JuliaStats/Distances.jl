
# Benchmark on column-wise distance evaluation

using Distances

macro bench_colwise_dist(repeat, dist, x, y)
    quote
        println("bench ", typeof($dist))

        # warming up
        r1 = evaluate($dist, ($x)[:,1], ($y)[:,1])
        colwise($dist, $x, $y)

        # timing

        t0 = @elapsed for k = 1 : $repeat
            n = size($x, 2)
            r = Array(typeof(r1), n)
            for j = 1 : n
                r[j] = evaluate($dist, ($x)[:,j], ($y)[:,j])
            end
        end
        @printf "    loop:     t = %9.6fs\n" (t0 / $repeat)

        t1 = @elapsed for k = 1 : $repeat
            r = colwise($dist, $x, $y)
        end
        @printf "    colwise:  t = %9.6fs  |  gain = %7.4fx\n" (t1 / $repeat) (t0 / t1)
        println()
    end
end


m = 200
n = 10000

x = rand(m, n)
y = rand(m, n)
w = rand(m)

Q = rand(m, m)
Q = Q' * Q

@bench_colwise_dist 20 SqEuclidean() x y
@bench_colwise_dist 20 Euclidean() x y
@bench_colwise_dist 20 Cityblock() x y
@bench_colwise_dist 20 Chebyshev() x y
@bench_colwise_dist  5 Minkowski(3.0) x y
@bench_colwise_dist 20 Hamming() x y

@bench_colwise_dist 20 CosineDist() x y
@bench_colwise_dist 10 CorrDist() x y
@bench_colwise_dist 20 ChiSqDist() x y
@bench_colwise_dist 10 KLDivergence() x y
@bench_colwise_dist  5 JSDivergence() x y

@bench_colwise_dist 20 WeightedSqEuclidean(w) x y
@bench_colwise_dist 20 WeightedEuclidean(w) x y
@bench_colwise_dist 20 WeightedCityblock(w) x y
@bench_colwise_dist  5 WeightedMinkowski(w, 3.0) x y
@bench_colwise_dist 20 WeightedHamming(w) x y

@bench_colwise_dist 10 SqMahalanobis(Q) x y
@bench_colwise_dist 10 Mahalanobis(Q) x y
