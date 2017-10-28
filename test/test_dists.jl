function test_metricity(dist, x, y, z)
    quote
        dxy = evaluate(dist, x, y)
        dxz = evaluate(dist, x, z)
        dyz = evaluate(dist, y, z)
        if isa(dist, PreMetric)
            # Unfortunately small non-zero numbers (~10^-16) are appearing
            # in our tests due to accumulating floating point rounding errors.
            # We either need to allow small errors in our tests or change the
            # way we do accumulations...
            @test evaluate(dist, x, x) + one(eltype(x)) ≈ one(eltype(x))
            @test evaluate(dist, y, y) + one(eltype(y)) ≈ one(eltype(y))
            @test evaluate(dist, z, z) + one(eltype(z)) ≈ one(eltype(z))
            @test dxy ≥ zero(eltype(x))
            @test dxz ≥ zero(eltype(x))
            @test dyz ≥ zero(eltype(x))
        end
        if isa(dist, SemiMetric)
            @test dxy ≈ evaluate(dist, y, x)
            @test dxz ≈ evaluate(dist, z, x)
            @test dyz ≈ evaluate(dist, y, z)
        else # Not symmetric, so more PreMetric tests
            @test evaluate(dist, y, x) ≥ zero(eltype(x))
            @test evaluate(dist, z, x) ≥ zero(eltype(x))
            @test evaluate(dist, z, y) ≥ zero(eltype(x))
        end
        if isa(dist, Metric)
            # Again we have small rounding errors in accumulations
            @test dxz ≤ dxy + dyz || dxz ≈ dxy + dyz
            dyx = evaluate(dist, y, x)
            @test dyz ≤ dyx + dxz || dyz ≈ dyx + dxz
            dzy = evaluate(dist, z, y)
            @test dxy ≤ dxz + dzy || dxy ≈ dxz + dzy
        end
    end
end

@testset "PreMetric, SemiMetric, Metric" begin
    for T in (Float64, F64)
        n = 10
        x = rand(T, n)
        y = rand(T, n)
        z = rand(T, n)

        test_metricity(SqEuclidean(), x, y, z)
        test_metricity(Euclidean(), x, y, z)
        test_metricity(Cityblock(), x, y, z)
        test_metricity(Chebyshev(), x, y, z)
        test_metricity(Minkowski(2.5), x, y, z)

        test_metricity(CosineDist(), x, y, z)
        test_metricity(CorrDist(), x, y, z)

        test_metricity(ChiSqDist(), x, y, z)

        test_metricity(Jaccard(), x, y, z)
        test_metricity(SpanNormDist(), x, y, z)

        test_metricity(BhattacharyyaDist(), x, y, z)
        test_metricity(HellingerDist(), x, y, z)

        x₁ = rand(T, 2)
        x₂ = rand(T, 2)
        x₃ = rand(T, 2)

        test_metricity(Haversine(6371.), x₁, x₂, x₃)

        k = rand(1:3, n)
        l = rand(1:3, n)
        m = rand(1:3, n)

        test_metricity(Hamming(), k, l, m)

        a = rand(Bool, n)
        b = rand(Bool, n)
        c = rand(Bool, n)

        test_metricity(RogersTanimoto(), a, b, c)
        test_metricity(Jaccard(), a, b, c)

        w = rand(T, n)

        test_metricity(WeightedSqEuclidean(w), x, y, z)
        test_metricity(WeightedEuclidean(w), x, y, z)
        test_metricity(WeightedCityblock(w), x, y, z)
        test_metricity(WeightedMinkowski(w, 2.5), x, y, z)
        test_metricity(WeightedHamming(w), a, b, c)

        Q = rand(T, n, n)
        Q = Q * Q'  # make sure Q is positive-definite

        test_metricity(SqMahalanobis(Q), x, y, z)
        test_metricity(Mahalanobis(Q), x, y, z)

        #semiaxes = [T(1/2), T(1), T(2)]
        #angles = [T(π/2), T(π/2), T(π/2)]

        #test_metricity(Ellipsoidal(semiaxes, angles), x, y, z)

        p = rand(T, n)
        q = rand(T, n)
        r = rand(T, n)
        p[p .< median(p)] = 0
        p /= sum(p)
        q /= sum(q)
        r /= sum(r)

        test_metricity(KLDivergence(), p, q, r)
        test_metricity(RenyiDivergence(0.0), p, q, r)
        test_metricity(RenyiDivergence(1.0), p, q, r)
        test_metricity(RenyiDivergence(Inf), p, q, r)
        test_metricity(RenyiDivergence(0.5), p, q, r)
        test_metricity(RenyiDivergence(2), p, q, r)
        test_metricity(RenyiDivergence(10), p, q, r)
        test_metricity(JSDivergence(), p, q, r)
    end
end

@testset "individual metrics" begin
    a = 1
    b = 2
    @test sqeuclidean(a, b) == 1.0

    @test euclidean(a, b) == 1.0
    @test cityblock(a, b) == 1.0
    @test chebyshev(a, b) == 1.0
    @test minkowski(a, b, 2) == 1.0
    @test hamming(a, b) == 1

    bt = [true, false, true]
    bf = [false, true, true]
    @test rogerstanimoto(bt, bf) == 4.0 / 5.0
    for T in (Float64, F64)

        for (_x, _y) in (([4.0, 5.0, 6.0, 7.0], [3.0, 9.0, 8.0, 1.0]),
                         ([4.0, 5.0, 6.0, 7.0], [3. 8.; 9. 1.0]))
            x, y = T.(_x), T.(_y)
            @test sqeuclidean(x, y) == 57.0
            @test euclidean(x, y) == sqrt(57.0)
            @test jaccard(x, y) == 13.0 / 28
            @test cityblock(x, y) == 13.0
            @test chebyshev(x, y) == 6.0
            @test minkowski(x, y, 2) == sqrt(57.0)
            @test_throws DimensionMismatch cosine_dist(1.0:2, 1.0:3)
            @test cosine_dist(x, y) ≈ (1.0 - 112. / sqrt(19530.0))
            x_int, y_int = Int64.(x), Int64.(y)
            @test cosine_dist(x_int, y_int) == (1.0 - 112.0 / sqrt(19530.0))
            @test corr_dist(x, y) ≈ cosine_dist(x .- mean(x), vec(y) .- mean(y))
            @test chisq_dist(x, y) == sum((x - vec(y)).^2 ./ (x + vec(y)))
            @test spannorm_dist(x, y) == maximum(x - vec(y)) - minimum(x - vec(y))

            @test gkl_divergence(x, y) ≈ sum(i -> x[i] * log(x[i] / y[i]) - x[i] + y[i], 1:length(x))

            @test meanad(x, y) ≈ mean(Float64[abs(x[i] - y[i]) for i in 1:length(x)])
            @test msd(x, y) ≈ mean(Float64[abs2(x[i] - y[i]) for i in 1:length(x)])
            @test rmsd(x, y) ≈ sqrt(msd(x, y))
            @test nrmsd(x, y) ≈ sqrt(msd(x, y)) / (maximum(x) - minimum(x))

            w = ones(4)
            @test sqeuclidean(x, y) ≈ wsqeuclidean(x, y, w)

            w = rand(size(x))
            @test wsqeuclidean(x, y, w) ≈ dot((x - vec(y)).^2, w)
            @test weuclidean(x, y, w) == sqrt(wsqeuclidean(x, y, w))
            @test wcityblock(x, y, w) ≈ dot(abs.(x - vec(y)), w)
            @test wminkowski(x, y, w, 2) ≈ weuclidean(x, y, w)
        end


        # Test weighted Hamming distances with even weights
        a = T.([1.0, 2.0, 1.0, 3.0, 2.0, 1.0])
        b = T.([1.0, 3.0, 0.0, 2.0, 2.0, 0.0])
        w = rand(T, size(a))

        @test whamming(a, a, w) === T(0.0)
        @test whamming(a, b, w) === sum((a .!= b) .* w)

        # Minimal test of Jaccard - test return type stability.
        @inferred evaluate(Jaccard(), rand(T, 3), rand(T, 3))
        @inferred evaluate(Jaccard(), [1, 2, 3], [1, 2, 3])
        @inferred evaluate(Jaccard(), [true, false, true], [false, true, true])

        # Test KL, Renyi and JS divergences
        r = rand(T, 12)
        p = copy(r)
        p[p .< median(p)] = 0.0
        scale = sum(p) / sum(r)
        r /= sum(r)
        p /= sum(p)
        q = rand(T, 12)
        q /= sum(q)

        klv = 0.0
        for i = 1:length(p)
            if p[i] > 0
                klv += p[i] * log(p[i] / q[i])
            end
        end
        @test kl_divergence(p, q) ≈ klv
        @test typeof(kl_divergence(p, q)) == T


        @test renyi_divergence(p, r, 0) ≈ -log(scale)
        @test renyi_divergence(p, r, 1) ≈ -log(scale)
        @test renyi_divergence(p, r, 10) ≈ -log(scale)
        @test renyi_divergence(p, r, rand()) ≈ -log(scale)
        @test renyi_divergence(p, r, Inf) ≈ -log(scale)
        @test isinf(renyi_divergence([0.0, 0.5, 0.5], [0.0, 1.0, 0.0], Inf))
        @test renyi_divergence([0.0, 1.0, 0.0], [0.0, 0.5, 0.5], Inf) ≈ log(2.0)
        @test renyi_divergence(p, q, 1) ≈ kl_divergence(p, q)

        pm = (p + q) / 2
        jsv = kl_divergence(p, pm) / 2 + kl_divergence(q, pm) / 2
        @test js_divergence(p, q) ≈ jsv
    end
end # testset

@testset "NaN behavior" begin
    a = [NaN, 0]; b = [0, NaN]
    @test isnan(chebyshev(a, b)) == isnan(maximum(a - b))
    a = [NaN, 0]; b = [0, 1]
    @test isnan(chebyshev(a, b)) == isnan(maximum(a - b))
    @test isnan(renyi_divergence([0.5, 0.0, 0.5], [0.5, 0.5, NaN], 2))
end #testset

@testset "empty vector" begin
    for T in (Float64, F64)
        a = T[]
        b = T[]
        @test sqeuclidean(a, b) == 0.0
        @test isa(sqeuclidean(a, b), T)
        @test euclidean(a, b) == 0.0
        @test isa(euclidean(a, b), T)
        @test cityblock(a, b) == 0.0
        @test isa(cityblock(a, b), T)
        @test chebyshev(a, b) == 0.0
        @test isa(chebyshev(a, b), T)
        @test minkowski(a, b, 2) == 0.0
        @test isa(minkowski(a, b, 2), T)
        @test hamming(a, b) == 0.0
        @test isa(hamming(a, b), Int)
        @test renyi_divergence(a, b, 1.0) == 0.0
        @test isa(renyi_divergence(a, b, 2.0), T)

        w = T[]
        @test isa(whamming(a, b, w), T)
    end
end # testset

@testset "DimensionMismatch throwing" begin
    a = [1, 0]; b = [2]
    @test_throws DimensionMismatch sqeuclidean(a, b)
    a = [1, 0]; b = [2.0] ; w = [3.0]
    @test_throws DimensionMismatch wsqeuclidean(a, b, w)
    a = [1, 0]; b = [2.0, 4.0] ; w = [3.0]
    @test_throws DimensionMismatch wsqeuclidean(a, b, w)
    p = [0.5, 0.5]; q = [0.3, 0.3, 0.4]
    @test_throws DimensionMismatch bhattacharyya(p, q)
    @test_throws DimensionMismatch hellinger(q, p)
    Q = rand(length(p), length(p))
    Q = Q * Q'  # make sure Q is positive-definite
    @test_throws DimensionMismatch mahalanobis(p, q, Q)
    @test_throws DimensionMismatch mahalanobis(q, q, Q)
    mat23 = [0.3 0.2 0.0; 0.1 0.0 0.4]
    mat22 = [0.3 0.2; 0.1 0.4]
    @test_throws DimensionMismatch colwise!(mat23, Euclidean(), mat23, mat23)
    @test_throws DimensionMismatch colwise!(mat23, Euclidean(), mat23, q)
    @test_throws DimensionMismatch colwise!(mat23, Euclidean(), mat23, mat22)
end # testset

@testset "mahalanobis" begin
    for T in (Float64, F64)
        x, y = T.([4.0, 5.0, 6.0, 7.0]), T.([3.0, 9.0, 8.0, 1.0])
        a = T.([1.0, 2.0, 1.0, 3.0, 2.0, 1.0])
        b = T.([1.0, 3.0, 0.0, 2.0, 2.0, 0.0])

        Q = rand(T, length(x), length(x))
        Q = Q * Q'  # make sure Q is positive-definite
        @test sqmahalanobis(x, y, Q) ≈ dot(x - y, Q * (x - y))
        @test eltype(sqmahalanobis(x, y, Q)) == T
        @test mahalanobis(x, y, Q) == sqrt(sqmahalanobis(x, y, Q))
        @test eltype(mahalanobis(x, y, Q)) == T
    end
end #testset

@testset "ellipsoidal" begin
    dist = Ellipsoidal([1.,.5,.5], [π/4,0.,0.])
    @test evaluate(dist, [1.,1.,0.], [0.,0.,0.]) ≈ √2
    @test evaluate(dist, [-1.,1.,0.], [0.,0.,0.]) ≈ √8
end

@testset "haversine" begin
    for T in (Float64, F64)
        @test haversine([-180.,0.], [180.,0.], 1.) ≈ 0 atol=1e-10
        @test haversine([0.,-90.],  [0.,90.],  1.) ≈ π atol=1e-10
        @test haversine((-180.,0.), (180.,0.), 1.) ≈ 0 atol=1e-10
        @test haversine((0.,-90.),  (0.,90.),  1.) ≈ π atol=1e-10
        @test_throws ArgumentError haversine([0.,-90., 0.25], [0.,90.], 1.)
    end
end

@testset "bhattacharyya / hellinger" begin
    for T in (Float64, F64)
        x, y = T.([4.0, 5.0, 6.0, 7.0]), T.([3.0, 9.0, 8.0, 1.0])
        a = T.([1.0, 2.0, 1.0, 3.0, 2.0, 1.0])
        b = T.([1.0, 3.0, 0.0, 2.0, 2.0, 0.0])
        p = rand(T, 12)
        p[p .< median(p)] = 0.0
        q = rand(T, 12)

        # Bhattacharyya and Hellinger distances are defined for discrete
        # probability distributions so to calculate the expected values
        # we need to normalize vectors.
        px = x ./ sum(x)
        py = y ./ sum(y)
        expected_bc_x_y = sum(sqrt.(px .* py))
        @test Distances.bhattacharyya_coeff(x, y) ≈ expected_bc_x_y
        @test bhattacharyya(x, y) ≈ (-log(expected_bc_x_y))
        @test hellinger(x, y) ≈ sqrt(1 - expected_bc_x_y)

        pa = a ./ sum(a)
        pb = b ./ sum(b)
        expected_bc_a_b = sum(sqrt.(pa .* pb))
        @test Distances.bhattacharyya_coeff(a, b) ≈ expected_bc_a_b
        @test bhattacharyya(a, b) ≈ (-log(expected_bc_a_b))
        @test hellinger(a, b) ≈ sqrt(1 - expected_bc_a_b)

        pp = p ./ sum(p)
        pq = q ./ sum(q)
        expected_bc_p_q = sum(sqrt.(pp .* pq))
        @test Distances.bhattacharyya_coeff(p, q) ≈ expected_bc_p_q
        @test bhattacharyya(p, q) ≈ (-log(expected_bc_p_q))
        @test hellinger(p, q) ≈ sqrt(1 - expected_bc_p_q)

        # Ensure it is semimetric
        @test bhattacharyya(x, y) ≈ bhattacharyya(y, x)
    end
end #testset


function test_colwise(dist, x, y, T)
    n = size(x, 2)
    r1 = zeros(T, n)
    r2 = zeros(T, n)
    r3 = zeros(T, n)
    for j = 1:n
        r1[j] = evaluate(dist, x[:, j], y[:, j])
        r2[j] = evaluate(dist, x[:, 1], y[:, j])
        r3[j] = evaluate(dist, x[:, j], y[:, 1])
    end
    @test colwise(dist, x, y) ≈ r1
    @test colwise(dist, x[:, 1], y) ≈ r2
    @test colwise(dist, x, y[:, 1]) ≈ r3
end

@testset "column-wise metrics" begin
    for T in (Float64, F64)
        m = 5
        n = 8
        X = rand(T, m, n)
        Y = rand(T, m, n)
        A = rand(1:3, m, n)
        B = rand(1:3, m, n)

        P = rand(T, m, n)
        Q = rand(T, m, n)
        # Make sure not to remove all of the non-zeros from any column
        for i in 1:n
            P[P[:, i] .< median(P[:, i]) / 2, i] = 0.0
        end

        test_colwise(SqEuclidean(), X, Y, T)
        test_colwise(Euclidean(), X, Y, T)
        test_colwise(Cityblock(), X, Y, T)
        test_colwise(Chebyshev(), X, Y, T)
        test_colwise(Minkowski(2.5), X, Y, T)
        test_colwise(Hamming(), A, B, T)

        test_colwise(CosineDist(), X, Y, T)
        test_colwise(CorrDist(), X, Y, T)

        test_colwise(ChiSqDist(), X, Y, T)
        test_colwise(KLDivergence(), P, Q, T)
        test_colwise(RenyiDivergence(0.0), P, Q, T)
        test_colwise(RenyiDivergence(1.0), P, Q, T)
        test_colwise(RenyiDivergence(Inf), P, Q, T)
        test_colwise(RenyiDivergence(0.5), P, Q, T)
        test_colwise(RenyiDivergence(2), P, Q, T)
        test_colwise(RenyiDivergence(10), P, Q, T)
        test_colwise(JSDivergence(), P, Q, T)
        test_colwise(SpanNormDist(), X, Y, T)

        test_colwise(BhattacharyyaDist(), X, Y, T)
        test_colwise(HellingerDist(), X, Y, T)

        w = rand(T, m)

        test_colwise(WeightedSqEuclidean(w), X, Y, T)
        test_colwise(WeightedEuclidean(w), X, Y, T)
        test_colwise(WeightedCityblock(w), X, Y, T)
        test_colwise(WeightedMinkowski(w, 2.5), X, Y, T)
        test_colwise(WeightedHamming(w), A, B, T)

        Q = rand(T, m, m)
        Q = Q * Q'  # make sure Q is positive-definite

        test_colwise(SqMahalanobis(Q), X, Y, T)
        test_colwise(Mahalanobis(Q), X, Y, T)
    end
end # testset


function test_pairwise(dist, x, y, T)
    quote
        nx = size(x, 2)
        ny = size(y, 2)
        rxy = zeros(T, nx, ny)
        rxx = zeros(T, nx, nx)
        for j = 1:ny, i = 1:nx
            rxy[i, j] = evaluate(dist, x[:, i], y[:, j])
        end
        for j = 1:nx, i = 1:nx
            rxx[i, j] = evaluate(dist, x[:, i], x[:, j])
        end
        @test pairwise(dist, x, y) ≈ rxy
        @test pairwise(dist, x) ≈ rxx
    end
end

@testset "pairwise metrics" begin
    for T in (Float64, F64)
        m = 5
        n = 8
        nx = 6
        ny = 8

        X = rand(T, m, nx)
        Y = rand(T, m, ny)
        A = rand(1:3, m, nx)
        B = rand(1:3, m, ny)

        P = rand(T, m, nx)
        Q = rand(T, m, ny)


        test_pairwise(SqEuclidean(), X, Y, T)
        test_pairwise(Euclidean(), X, Y, T)
        test_pairwise(Cityblock(), X, Y, T)
        test_pairwise(Chebyshev(), X, Y, T)
        test_pairwise(Minkowski(2.5), X, Y, T)
        test_pairwise(Hamming(), A, B, T)

        test_pairwise(CosineDist(), X, Y, T)
        test_pairwise(CorrDist(), X, Y, T)

        test_pairwise(ChiSqDist(), X, Y, T)
        test_pairwise(KLDivergence(), P, Q, T)
        test_pairwise(RenyiDivergence(0.0), P, Q, T)
        test_pairwise(RenyiDivergence(1.0), P, Q, T)
        test_pairwise(RenyiDivergence(Inf), P, Q, T)
        test_pairwise(RenyiDivergence(0.5), P, Q, T)
        test_pairwise(RenyiDivergence(2), P, Q, T)
        test_pairwise(JSDivergence(), P, Q, T)

        test_pairwise(BhattacharyyaDist(), X, Y, T)
        test_pairwise(HellingerDist(), X, Y, T)

        w = rand(m)

        test_pairwise(WeightedSqEuclidean(w), X, Y, T)
        test_pairwise(WeightedEuclidean(w), X, Y, T)
        test_pairwise(WeightedCityblock(w), X, Y, T)
        test_pairwise(WeightedMinkowski(w, 2.5), X, Y, T)
        test_pairwise(WeightedHamming(w), A, B, T)

        Q = rand(m, m)
        Q = Q * Q'  # make sure Q is positive-definite

        test_pairwise(SqMahalanobis(Q), X, Y, T)
        test_pairwise(Mahalanobis(Q), X, Y, T)
    end

end #testset

@testset "Euclidean precision" begin
    X = [0.1 0.2; 0.3 0.4; -0.1 -0.1]
    pd = pairwise(Euclidean(1e-12), X, X)
    @test pd[1, 1] == 0
    @test pd[2, 2] == 0
    pd = pairwise(Euclidean(1e-12), X)
    @test pd[1, 1] == 0
    @test pd[2, 2] == 0
    pd = pairwise(SqEuclidean(1e-12), X, X)
    @test pd[1, 1] == 0
    @test pd[2, 2] == 0
    pd = pairwise(SqEuclidean(1e-12), X)
    @test pd[1, 1] == 0
    @test pd[2, 2] == 0
end
