
# Benchmark on pairwise distance evaluation

using Distances

macro bench_pairwise_dist(_repeat, _dist, _x, _y)
    quote
        repeat = $(esc(_repeat))
        dist = $(esc(_dist))
        x = $(esc(_x))
        y = $(esc(_y))
        println("bench ", typeof(dist))

        # warming up
        r1 = evaluate(dist, x[:,1], ($y)[:,1])
        pairwise(dist, x, y)

        # timing

        t0 = @elapsed for k = 1 : $repeat
            nx = size(x, 2)
            ny = size(y, 2)
            r = Matrix{typeof(r1)}(nx, ny)
            for j = 1 : ny
                for i = 1 : nx
                    r[i, j] = evaluate(dist, x[:,i], (y)[:,j])
                end
            end
        end
        @printf "    loop:      t = %9.6fs\n" (t0 / $repeat)

        t1 = @elapsed for k = 1 : $repeat
            r = pairwise(dist, x, y)
        end
        @printf "    pairwise:  t = %9.6fs  |  gain = %7.4fx\n" (t1 / $repeat) (t0 / t1)
        println()
    end
end


m = 100
nx = 200
ny = 250

x = rand(m, nx)
y = rand(m, ny)

w = rand(m)
Q = rand(m, m)
Q = Q' * Q

@bench_pairwise_dist 20 SqEuclidean() x y
@bench_pairwise_dist 20 Euclidean() x y
@bench_pairwise_dist 20 Cityblock() x y
@bench_pairwise_dist 20 Chebyshev() x y
@bench_pairwise_dist  5 Minkowski(3.0) x y
@bench_pairwise_dist 20 Hamming() x y

@bench_pairwise_dist 20 CosineDist() x y
@bench_pairwise_dist 10 CorrDist() x y
@bench_pairwise_dist 20 ChiSqDist() x y
@bench_pairwise_dist 10 KLDivergence() x y
@bench_pairwise_dist  5 JSDivergence() x y

@bench_pairwise_dist 10 BhattacharyyaDist() x y
@bench_pairwise_dist 10 HellingerDist() x y

@bench_pairwise_dist 20 WeightedSqEuclidean(w) x y
@bench_pairwise_dist 20 WeightedEuclidean(w) x y
@bench_pairwise_dist 20 WeightedCityblock(w) x y
@bench_pairwise_dist  5 WeightedMinkowski(w, 3.0) x y
@bench_pairwise_dist 20 WeightedHamming(w) x y

@bench_pairwise_dist 10 SqMahalanobis(Q) x y
@bench_pairwise_dist 10 Mahalanobis(Q) x y
