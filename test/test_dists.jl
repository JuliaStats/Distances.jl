# Unit tests for Distances

struct FooDist <: PreMetric end # Julia 1.0 Compat: struct definition must be put in global scope

@testset "result_type" begin
    foodist(a, b) = a + b
    (::FooDist)(a, b) = foodist(a, b)
    for (Ta, Tb) in [
        (Int, Int),
        (Int, Float64),
        (Float32, Float32),
        (Float32, Float64),
    ]
        A, B = rand(Ta, 2, 3), rand(Tb, 2, 3)
        @test result_type(FooDist(), A, B) == result_type(FooDist(), Ta, Tb)
        @test result_type(foodist, A, B) == result_type(foodist, Ta, Tb) == typeof(foodist(oneunit(Ta), oneunit(Tb)))

        a, b = rand(Ta), rand(Tb)
        @test result_type(FooDist(), a, b) == result_type(FooDist(), Ta, Tb)
        @test result_type(foodist, a, b) == result_type(foodist, Ta, Tb) == typeof(foodist(oneunit(Ta), oneunit(Tb)))
    end
end

function test_metricity(dist, x, y, z)
    @testset "Test metricity of $(typeof(dist))" begin
        @test dist(x, y) == evaluate(dist, x, y)

        dxy = dist(x, y)
        dxz = dist(x, z)
        dyz = dist(y, z)
        if isa(dist, PreMetric)
            # Unfortunately small non-zero numbers (~10^-16) are appearing
            # in our tests due to accumulating floating point rounding errors.
            # We either need to allow small errors in our tests or change the
            # way we do accumulations...
            @test dist(x, x) + one(eltype(x)) ≈ one(eltype(x))
            @test dist(y, y) + one(eltype(y)) ≈ one(eltype(y))
            @test dist(z, z) + one(eltype(z)) ≈ one(eltype(z))
            @test dxy ≥ zero(eltype(x))
            @test dxz ≥ zero(eltype(x))
            @test dyz ≥ zero(eltype(x))
        end
        if isa(dist, SemiMetric)
            @test dxy ≈ dist(y, x)
            @test dxz ≈ dist(z, x)
            @test dyz ≈ dist(y, z)
        else # Not symmetric, so more PreMetric tests
            @test dist(y, x) ≥ zero(eltype(x))
            @test dist(z, x) ≥ zero(eltype(x))
            @test dist(z, y) ≥ zero(eltype(x))
        end
        if isa(dist, Metric)
            # Again we have small rounding errors in accumulations
            @test dxz ≤ dxy + dyz || dxz ≈ dxy + dyz
            dyx = dist(y, x)
            @test dyz ≤ dyx + dxz || dyz ≈ dyx + dxz
            dzy = dist(z, y)
            @test dxy ≤ dxz + dzy || dxy ≈ dxz + dzy
        end
    end
end

@testset "PreMetric, SemiMetric, Metric on $T" for T in (Float64, F64)
    Random.seed!(123)
    n = 100
    x = rand(T, n)
    y = rand(T, n)
    z = rand(T, n)

    test_metricity(SqEuclidean(), x, y, z)
    test_metricity(Euclidean(), x, y, z)
    test_metricity(Cityblock(), x, y, z)
    test_metricity(TotalVariation(), x, y, z)
    test_metricity(Chebyshev(), x, y, z)
    test_metricity(Minkowski(2.5), x, y, z)

    test_metricity(CosineDist(), x, y, z)
    test_metricity(CorrDist(), x, y, z)

    test_metricity(ChiSqDist(), x, y, z)

    test_metricity(Jaccard(), x, y, z)
    test_metricity(SpanNormDist(), x, y, z)

    test_metricity(BhattacharyyaDist(), x, y, z)
    test_metricity(HellingerDist(), x, y, z)
    test_metricity(Bregman(x -> sqeuclidean(x, zero(x)), x -> 2*x), x, y, z);


    x₁ = rand(T, 2)
    x₂ = rand(T, 2)
    x₃ = rand(T, 2)

    test_metricity(Haversine(6371.0), x₁, x₂, x₃)
    test_metricity(SphericalAngle(), x₁, x₂, x₃)

    k = rand(1:3, n)
    l = rand(1:3, n)
    m = rand(1:3, n)

    test_metricity(Hamming(), k, l, m)

    a = rand(Bool, n)
    b = rand(Bool, n)
    c = rand(Bool, n)

    test_metricity(RogersTanimoto(), a, b, c)
    test_metricity(BrayCurtis(), a, b, c)
    test_metricity(Jaccard(), a, b, c)

    for w in (rand(T, n), (rand(T, n)...,))
        test_metricity(PeriodicEuclidean(w), x, y, z)
        test_metricity(WeightedSqEuclidean(w), x, y, z)
        test_metricity(WeightedEuclidean(w), x, y, z)
        test_metricity(WeightedCityblock(w), x, y, z)
        test_metricity(WeightedMinkowski(w, 2.5), x, y, z)
        test_metricity(WeightedHamming(w), a, b, c)
    end

    Q = rand(T, n, n)
    Q = Q * Q'  # make sure Q is positive-definite

    test_metricity(SqMahalanobis(Q), x, y, z)
    test_metricity(Mahalanobis(Q), x, y, z)

    p = rand(T, n)
    q = rand(T, n)
    r = rand(T, n)
    p[p .< median(p)] .= 0
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

@testset "individual metrics" begin
    a = 1
    b = 2
    @test sqeuclidean(a, b) === 1
    @test euclidean(a, b) === 1.0
    @test jaccard(a, b) === 0.5
    @test cityblock(a, b) === 1
    @test totalvariation(a, b) === 0.5
    @test chebyshev(a, b) == 1.0
    @test braycurtis(a, b) === 1/3
    @test minkowski(a, b, 2) == 1.0
    @test hamming(a, b) === 1
    @test hamming("martha", "marhta") === 2
    @test hamming("es an ", " vs an") === 6
    @test hamming("", "") === 0
    @test peuclidean(a, b, 0.5) === 0.0
    @test peuclidean(a, b, 2) === 1.0
    @test cosine_dist(a, b) === 0.0
    @test bhattacharyya(a, b) === bhattacharyya([a], [b]) === -0.0
    @test bhattacharyya(a, b) === bhattacharyya((a,), (b,))
    @test isnan(corr_dist(a, b))
    @test spannorm_dist(a, b) === 0

    bt = [true, false, true]
    bf = [false, true, true]
    @test rogerstanimoto(bt, bf) == 4.0 / 5.0
    @test braycurtis(bt, bf) == 0.5

    for w in (2, (2,))
        @test wsqeuclidean(a, b, w) === 2
        @test weuclidean(a, b, w) === sqrt(2)
        @test wcityblock(a, b, w) === 2
        @test wminkowski(a, b, w, 2) === sqrt(2)
        @test whamming(a, b, w) === 2
    end

    for T in (Float64, F64)
        for (_x, _y) in (([4.0, 5.0, 6.0, 7.0], [3.0, 9.0, 8.0, 1.0]),
                         ([4.0, 5.0, 6.0, 7.0], [3. 8.; 9. 1.0]))
            x, y = T.(_x), T.(_y)
            for (x, y) in ((x, y),
                           (convert(Array{Union{Missing, T}}, x), convert(Array{Union{Missing, T}}, y)),
                           ((Iterators.take(x, 4), Iterators.take(y, 4))), # iterator
                           (((x[i] for i in 1:length(x)), (y[i] for i in 1:length(y)))), # generator
                          )
                xc, yc = collect(x), collect(y)
                @test sqeuclidean(x, y) == 57.0
                @test euclidean(x, y) == sqrt(57.0)
                @test jaccard(x, y) == 13.0 / 28
                @test cityblock(x, y) == 13.0
                @test totalvariation(x, y) == 6.5
                @test chebyshev(x, y) == 6.0
                @test braycurtis(x, y) == 1.0 - (30.0 / 43.0)
                @test minkowski(x, y, 2) == sqrt(57.0)
                @test peuclidean(x, y, fill(10.0, 4)) == sqrt(37)
                @test peuclidean(xc - vec(yc), zero(yc), fill(10.0, 4)) == peuclidean(x, y, fill(10.0, 4))
                @test peuclidean(x, y, [10.0, 10.0, 10.0, Inf]) == sqrt(57)
                @test_throws DimensionMismatch cosine_dist(1.0:2, 1.0:3)
                @test cosine_dist(x, y) ≈ (1.0 - 112. / sqrt(19530.0))
                x_int, y_int = Int64.(x), Int64.(y)
                @test cosine_dist(x_int, y_int) == (1.0 - 112.0 / sqrt(19530.0))
                @test corr_dist(x, y) ≈ cosine_dist(x .- mean(x), vec(yc) .- mean(y))
                @test corr_dist(OffsetVector(xc, -1:length(xc)-2), yc) == corr_dist(x, y)
                @test chisq_dist(x, y) == sum((xc - vec(yc)).^2 ./ (xc + vec(yc)))
                @test spannorm_dist(x, y) == maximum(xc - vec(yc)) - minimum(xc - vec(yc))

                @test gkl_divergence(x, y) ≈ sum(i -> xc[i] * log(xc[i] / yc[i]) - xc[i] + yc[i], 1:length(x))

                @test meanad(x, y) ≈ mean(Float64[abs(xc[i] - yc[i]) for i in 1:length(x)])
                @test msd(x, y) ≈ mean(Float64[abs2(xc[i] - yc[i]) for i in 1:length(x)])
                @test rmsd(x, y) ≈ sqrt(msd(x, y))
                @test nrmsd(x, y) ≈ sqrt(msd(x, y)) / (maximum(x) - minimum(x))

                w = ones(4)
                @test sqeuclidean(x, y) ≈ wsqeuclidean(x, y, w)

                w = rand(Float64, length(x))
                @test wsqeuclidean(x, y, w) ≈ dot((xc - vec(yc)).^2, w)
                @test weuclidean(x, y, w) == sqrt(wsqeuclidean(x, y, w))
                @test wcityblock(x, y, w) ≈ dot(abs.(xc - vec(yc)), w)
                @test wminkowski(x, y, w, 2) ≈ weuclidean(x, y, w)
            end
        end

        # Test ChiSq doesn't give NaN at zero
        @test chisq_dist([0.0], [0.0]) == 0.0

        # Test weighted Hamming distances with even weights
        a = T.([1.0, 2.0, 1.0, 3.0, 2.0, 1.0])
        b = T.([1.0, 3.0, 0.0, 2.0, 2.0, 0.0])
        w = rand(T, size(a))

        @test whamming(a, a, w) === T(0.0)
        @test whamming(a, b, w) === sum((a .!= b) .* w)

        # Minimal test of Jaccard - test return type stability.
        @inferred Jaccard()(rand(T, 3), rand(T, 3))
        @inferred Jaccard()([1, 2, 3], [1, 2, 3])
        @inferred Jaccard()([true, false, true], [false, true, true])

        # Test Bray-Curtis. Should be 1 if no elements are shared, 0 if all are the same
        @test braycurtis([1,0,3],[0,1,0]) == 1.0
        @test braycurtis(rand(10), zeros(10)) == 1.0
        @test braycurtis([1,0],[1,0]) == 0.0

        # Test KL, Renyi and JS divergences
        r = rand(T, 12)
        p = copy(r)
        p[p .< median(p)] .= 0.0
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

        pm = (p + q) / 2
        for (r, p, pm) in ((r, p, pm),
                           (Iterators.take(r, length(r)), Iterators.take(p, length(p)), Iterators.take(pm, length(pm))),
                           ((r[i] for i in 1:length(r)), (p[i] for i in 1:length(p)), (pm[i] for i in 1:length(pm))),
                          )
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

            jsv = kl_divergence(p, pm) / 2 + kl_divergence(q, pm) / 2
            @test js_divergence(p, q) ≈ jsv
        end
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
    for T in (Float64, F64), (a, b) in ((T[], T[]), (Iterators.take(T[], 0), Iterators.take(T[], 0)))
        @test sqeuclidean(a, b) == 0.0
        @test isa(sqeuclidean(a, b), T)
        @test euclidean(a, b) == 0.0
        @test isa(euclidean(a, b), T)
        @test cityblock(a, b) == 0.0
        @test isa(cityblock(a, b), T)
        @test totalvariation(a, b) == 0.0
        @test isa(totalvariation(a, b), T)
        @test chebyshev(a, b) == 0.0
        @test isa(chebyshev(a, b), T)
        @test braycurtis(a, b) == 0.0
        @test isa(braycurtis(a, b), T)
        @test minkowski(a, b, 2) == 0.0
        @test isa(minkowski(a, b, 2), T)
        @test hamming(a, b) == 0.0
        @test isa(hamming(a, b), Int)
        @test renyi_divergence(a, b, 1.0) == 0.0
        @test isa(renyi_divergence(a, b, 2.0), T)
        @test braycurtis(a, b) == 0.0
        @test isa(braycurtis(a, b), T)

        w = T[]
        @test isa(whamming(a, b, w), T)
        @test peuclidean(a, b, w) == 0.0
        @test isa(peuclidean(a, b, w), T)
    end
end # testset

@testset "DimensionMismatch throwing" begin
    a = [1, 0]; b = [2]
    @test_throws DimensionMismatch sqeuclidean(a, b)
    a = (1, 0); b = (2,)
    @test_throws DimensionMismatch sqeuclidean(a, b)
    a = (1, 0); b = (2.0,); w = (3.0,)
    @test_throws DimensionMismatch wsqeuclidean(a, b, w)
    @test_throws DimensionMismatch peuclidean(a, b, w)
    a = [1, 0]; b = [2.0, 4.0] ; w = [3.0]
    @test_throws DimensionMismatch wsqeuclidean(a, b, w)
    @test_throws DimensionMismatch peuclidean(a, b, w)
    a = (1, 0); b = (2.0, 4.0) ; w = (3.0,)
    @test_throws DimensionMismatch wsqeuclidean(a, b, w)
    @test_throws DimensionMismatch peuclidean(a, b, w)
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
    @test_throws DimensionMismatch colwise!(mat23, Bregman(x -> sqeuclidean(x, zero(x)), x -> 2*x), mat23, mat22)
    @test_throws DimensionMismatch Bregman(x -> sqeuclidean(x, zero(x)), x -> 2*x)([1, 2, 3], [1, 2])
    @test_throws DimensionMismatch Bregman(x -> sqeuclidean(x, zero(x)), x -> [1, 2])([1, 2, 3], [1, 2, 3])
end # testset

@testset "Different input types" begin
    for (x, y) in (([4, 5, 6, 7], [3.0, 9.0, 8.0, 1.0]),
                   ([4, 5, 6, 7], [3//1 8; 9 1]))
        @test (@inferred sqeuclidean(x, y)) == 57
        @test (@inferred euclidean(x, y)) == sqrt(57)
        @test (@inferred jaccard(x, y)) == convert(Base.promote_eltype(x, y), 13 // 28)
        @test (@inferred cityblock(x, y)) == 13
        @test (@inferred totalvariation(x, y)) == 6.5
        @test (@inferred chebyshev(x, y)) == 6
        @test (@inferred braycurtis(x, y)) == convert(Base.promote_eltype(x, y), 13 // 43)
        @test (@inferred minkowski(x, y, 2)) == sqrt(57)
        @test (@inferred peuclidean(x, y, fill(10, 4))) == sqrt(37)
        @test (@inferred peuclidean(x - vec(y), zero(y), fill(10, 4))) == peuclidean(x, y, fill(10, 4))
        @test (@inferred peuclidean(x, y, [10.0, 10.0, 10.0, Inf])) == sqrt(57)
        @test_throws DimensionMismatch cosine_dist(1.0:2, 1.0:3)
        @test (@inferred cosine_dist(x, y)) ≈ (1 - 112 / sqrt(19530))
        @test (@inferred corr_dist(x, y)) ≈ cosine_dist(x .- mean(x), vec(y) .- mean(y))
        @test (@inferred chisq_dist(x, y)) == sum((x - vec(y)).^2 ./ (x + vec(y)))
        @test (@inferred spannorm_dist(x, y)) == maximum(x - vec(y)) - minimum(x - vec(y))

        @test (@inferred gkl_divergence(x, y)) ≈ sum(i -> x[i] * log(x[i] / y[i]) - x[i] + y[i], 1:length(x))

        @test (@inferred meanad(x, y)) ≈ mean(Float64[abs(x[i] - y[i]) for i in 1:length(x)])
        @test (@inferred msd(x, y)) ≈ mean(Float64[abs2(x[i] - y[i]) for i in 1:length(x)])
        @test (@inferred rmsd(x, y)) ≈ sqrt(msd(x, y))
        @test (@inferred nrmsd(x, y)) ≈ sqrt(msd(x, y)) / (maximum(x) - minimum(x))

        w = ones(Int, 4)
        @test sqeuclidean(x, y) ≈ wsqeuclidean(x, y, w)

        w = rand(1:length(x), size(x))
        @test (@inferred wsqeuclidean(x, y, w)) ≈ dot((x - vec(y)).^2, w)
        @test (@inferred weuclidean(x, y, w)) == sqrt(wsqeuclidean(x, y, w))
        @test (@inferred wcityblock(x, y, w)) ≈ dot(abs.(x - vec(y)), w)
        @test (@inferred wminkowski(x, y, w, 2)) ≈ weuclidean(x, y, w)
    end
end

@testset "mahalanobis" begin
    for T in (Float64, F64, ComplexF64)
        x, y = T.([4.0, 5.0, 6.0, 7.0]), T.([3.0, 9.0, 8.0, 1.0])

        Q = rand(T, length(x), length(x))
        Q = Q * Q'  # make sure Q is positive-definite
        for Q in (Q, T <: Complex ? Hermitian(Q) : Symmetric(Q))
            @test sqmahalanobis(x, y, Q) ≈ dot(x - y, Q * (x - y))
            @test eltype(sqmahalanobis(x, y, Q)) == T
            @test mahalanobis(x, y, Q) == sqrt(sqmahalanobis(x, y, Q))
            @test eltype(mahalanobis(x, y, Q)) == T
        end
    end
end #testset

@testset "haversine" begin
    for T in (Float64, F64)
        @test haversine([-180.,0.], [180.,0.], 1.) ≈ 0 atol=1e-10
        @test haversine([0.,-90.],  [0.,90.],  1.) ≈ π atol=1e-10
        @test haversine((-180.,0.), (180.,0.), 1.) ≈ 0 atol=1e-10
        @test haversine((0.,-90.),  (0.,90.),  1.) ≈ π atol=1e-10
        x, y = (1.,-15.625), (-179.,15.625)
        @test haversine(x, y, 6371.) ≈ 20015 atol=1e-1
        @test haversine(x, y) ≈ 20015086 rtol=1e-7
        @test Haversine()(x, y) ≈ 20015086 rtol=1e-7
        @test haversine(Float32.(x), Float32.(y)) isa Float32
        @test haversine(Iterators.take(x, 2), Iterators.take(y, 2), 6371.) ≈ 20015 atol=1e-1
        @test_throws ArgumentError haversine([0.,-90., 0.25], [0.,90.], 1.)
        x, y = (1.0°,-15.625°), (-179.0°,15.625°)
        @test haversine(x, y, 6371.0km) ≈ 20015km atol=1e-1km
    end
end

@testset "spherical angle" begin
    @test spherical_angle([-π, 0],    [π,0]) ≈ 0 atol=1e-10
    @test spherical_angle([0,-π/2], [0,π/2]) ≈ π atol=1e-10
    @test spherical_angle((-π,0),     (π,0)) ≈ 0 atol=1e-10
    @test spherical_angle((0,-π/2), (0,π/2)) ≈ π atol=1e-10
    x, y = rand(2), rand(2)
    x_deg, y_deg = rad2deg.(x), rad2deg.(y)
    @test spherical_angle(x, y) ≈ haversine(x_deg, y_deg, 1) atol=1e-10
    @test spherical_angle(Iterators.take(x, 2), Iterators.take(y, 2)) ≈ haversine(x_deg, y_deg, 1) atol=1e-10
    @test_throws ArgumentError spherical_angle([0.,-π/2, 0.25], [0.,π/2])
end

@testset "bhattacharyya / hellinger" begin
    for T in (Int, Float64, F64)
        x, y = T.([4, 5, 6, 7]), T.([3, 9, 8, 1])
        a = T.([1, 2, 1, 3, 2, 1])
        b = T.([1, 3, 0, 2, 2, 0])
        p = T == Int ? rand(0:10, 12) : rand(T, 12)
        p[p .< median(p)] .= 0
        q = T == Int ? rand(0:10, 12) : rand(T, 12)

        # Bhattacharyya and Hellinger distances are defined for discrete
        # probability distributions so to calculate the expected values
        # we need to normalize vectors.
        px = x ./ sum(x)
        py = y ./ sum(y)
        expected_bc_x_y = sum(sqrt.(px .* py))
        for (x, y) in ((x, y), (Iterators.take(x, 12), Iterators.take(y, 12)))
            @test Distances.bhattacharyya_coeff(x, y) ≈ expected_bc_x_y
            @test bhattacharyya(x, y) ≈ (-log(expected_bc_x_y))
            @test hellinger(x, y) ≈ sqrt(1 - expected_bc_x_y)
        end

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
    @testset "Colwise test for $(typeof(dist))" begin
        n = size(x, 2)
        r1 = zeros(T, n)
        r2 = zeros(T, n)
        r3 = zeros(T, n)
        r4 = zeros(T, 1, n)
        for j = 1:n
            r1[j] = dist(x[:, j], y[:, j])
            r2[j] = dist(x[:, 1], y[:, j])
            r3[j] = dist(x[:, j], y[:, 1])
        end
        # ≈ and all( .≈ ) seem to behave slightly differently for F64
        @test all(colwise(dist, x, y) .≈ r1)
        @test all(colwise(dist, (x[:,i] for i in axes(x, 2)), (y[:,i] for i in axes(y, 2))) .≈ r1)
        colwise!(r4, dist, x, y)
        @test all(r4[i] ≈ r1[i] for i in 1:n)
        @test all(colwise(dist, x[:, 1], y) .≈ r2)
        @test all(colwise(dist, x, y[:, 1]) .≈ r3)
    end
end

@testset "column-wise metrics on $T" for T in (Float64, F64)
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
        P[P[:, i] .< median(P[:, i]) / 2, i] .= 0.0
    end

    test_colwise(SqEuclidean(), X, Y, T)
    test_colwise(Euclidean(), X, Y, T)
    test_colwise(Cityblock(), X, Y, T)
    test_colwise(TotalVariation(), X, Y, T)
    test_colwise(Chebyshev(), X, Y, T)
    test_colwise(Minkowski(2.5), X, Y, T)
    test_colwise(Hamming(), A, B, T)
    test_colwise(Bregman(x -> sqeuclidean(x, zero(x)), x -> 2*x), X, Y, T);

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
    test_colwise(BrayCurtis(), X, Y, T)

    w = rand(T, m)

    test_colwise(WeightedSqEuclidean(w), X, Y, T)
    test_colwise(WeightedEuclidean(w), X, Y, T)
    test_colwise(WeightedCityblock(w), X, Y, T)
    test_colwise(WeightedMinkowski(w, 2.5), X, Y, T)
    test_colwise(WeightedHamming(w), A, B, T)
    test_colwise(PeriodicEuclidean(w), X, Y, T)

    Q = rand(T, m, m)
    Q = Q * Q'  # make sure Q is positive-definite

    test_colwise(SqMahalanobis(Q), X, Y, T)
    test_colwise(Mahalanobis(Q), X, Y, T)
end

function test_pairwise(dist, x, y, T)
    @testset "Pairwise test for $(typeof(dist))" begin
        nx = size(x, 2)
        ny = size(y, 2)
        rxy = zeros(T, nx, ny)
        rxx = zeros(T, nx, nx)
        for j = 1:ny, i = 1:nx
            rxy[i, j] = dist(x[:, i], y[:, j])
        end
        for j = 1:nx, i = 1:nx
            rxx[i, j] = dist(x[:, i], x[:, j])
        end
        # As earlier, we have small rounding errors in accumulations
        @test pairwise(dist, x, y, dims=2) ≈ rxy
        @test pairwise(dist, x, dims=2) ≈ rxx
        @test pairwise(dist, permutedims(x), permutedims(y), dims=1) ≈ rxy
        @test pairwise(dist, permutedims(x), dims=1) ≈ rxx
        vecx = (x[:, i] for i in 1:nx)
        vecy = (y[:, i] for i in 1:ny)
        for (vecx, vecy) in ((vecx, vecy), (collect(vecx), collect(vecy)))
            @test pairwise(dist, vecx, vecy) ≈ rxy
            @test pairwise(dist, vecx) ≈ rxx
            @test pairwise!(similar(rxy), dist, vecx, vecy) ≈ rxy
            @test pairwise!(similar(rxx), dist, vecx) ≈ rxx
        end
    end
end

@testset "pairwise metrics on $T" for T in (Float64, F64)
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
    test_pairwise(TotalVariation(), X, Y, T)
    test_pairwise(Chebyshev(), X, Y, T)
    test_pairwise(Minkowski(2.5), X, Y, T)
    test_pairwise(Hamming(), A, B, T)

    test_pairwise(CosineDist(), X, Y, T)
    test_pairwise(CosineDist(), A, B, T)
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
    test_pairwise(BrayCurtis(), X, Y, T)
    test_pairwise(Bregman(x -> sqeuclidean(x, zero(x)), x -> 2*x), X, Y, T)

    w = rand(m)

    test_pairwise(WeightedSqEuclidean(w), X, Y, T)
    test_pairwise(WeightedEuclidean(w), X, Y, T)
    test_pairwise(WeightedCityblock(w), X, Y, T)
    test_pairwise(WeightedMinkowski(w, 2.5), X, Y, T)
    test_pairwise(WeightedHamming(w), A, B, T)
    test_pairwise(PeriodicEuclidean(w), X, Y, T)

    Q = rand(m, m)
    Q = Q * Q'  # make sure Q is positive-definite

    test_pairwise(SqMahalanobis(Q), X, Y, T)
    test_pairwise(Mahalanobis(Q), X, Y, T)

    m, nx, ny = 2, 8, 6

    X = rand(T, m, nx)
    Y = rand(T, m, ny)
    test_pairwise(Haversine(), X, Y, T)
    test_pairwise(SphericalAngle(), X, Y, T)
end

@testset "pairwise metrics on complex arrays" begin
    m = 5
    n = 8
    nx = 6
    ny = 8

    X = rand(ComplexF64, m, nx)
    Y = rand(ComplexF64, m, ny)
    
    test_pairwise(SqEuclidean(), X, Y, Float64)
    test_pairwise(Euclidean(), X, Y, Float64)

    w = rand(m)

    test_pairwise(WeightedSqEuclidean(w), X, Y, Float64)
    test_pairwise(WeightedEuclidean(w), X, Y, Float64)
end

function test_scalar_pairwise(dist, x, y, T)
    @testset "Scalar pairwise test for $(typeof(dist))" begin
        rxy = dist.(x, permutedims(y))
        rxx = dist.(x, permutedims(x))
        # As earlier, we have small rounding errors in accumulations
        @test pairwise(dist, x, y) ≈ rxy
        @test pairwise(dist, x) ≈ rxx
        @test pairwise(dist, permutedims(x), permutedims(y), dims=2) ≈ rxy
        @test pairwise(dist, permutedims(x), dims=2) ≈ rxx
        @test_throws DimensionMismatch pairwise(dist, permutedims(x), permutedims(y), dims=1)
    end
end

@testset "scalar pairwise metrics on $T" for T in (Float64, F64)
    m = 5
    n = 8
    nx = 6
    ny = 8
    x = rand(T, nx)
    y = rand(T, ny)
    a = rand(1:3, nx)
    b = rand(1:3, ny)
    test_scalar_pairwise(SqEuclidean(), x, y, T)
    test_scalar_pairwise(Euclidean(), x, y, T)
    test_scalar_pairwise(Cityblock(), x, y, T)
    test_scalar_pairwise(TotalVariation(), x, y, T)
    test_scalar_pairwise(Chebyshev(), x, y, T)
    test_scalar_pairwise(Minkowski(2.5), x, y, T)
    test_scalar_pairwise(Hamming(), a, b, T)
    test_scalar_pairwise(CosineDist(), x, y, T)
    test_scalar_pairwise(CosineDist(), a, b, T)
    test_scalar_pairwise(ChiSqDist(), x, y, T)
    test_scalar_pairwise(KLDivergence(), x, y, T)
    test_scalar_pairwise(JSDivergence(), x, y, T)
    test_scalar_pairwise(BrayCurtis(), x, y, T)
    w = rand(1, 1)
    test_scalar_pairwise(WeightedSqEuclidean(w), x, y, T)
    test_scalar_pairwise(WeightedEuclidean(w), x, y, T)
    test_scalar_pairwise(WeightedCityblock(w), x, y, T)
    test_scalar_pairwise(WeightedMinkowski(w, 2.5), x, y, T)
    test_scalar_pairwise(WeightedHamming(w), a, b, T)
    test_scalar_pairwise(PeriodicEuclidean(w), x, y, T)
end

@testset "Euclidean precision" begin
    x = [0.1 0.2; 0.3 0.4; -0.1 -0.1]
    for X in (x, complex(x))
        pd = pairwise(Euclidean(1e-12), X, X, dims=2)
        @test pd[1, 1] == 0
        @test pd[2, 2] == 0
        pd = pairwise(Euclidean(1e-12), X, dims=2)
        @test pd[1, 1] == 0
        @test pd[2, 2] == 0
        pd = pairwise(SqEuclidean(1e-12), X, X, dims=2)
        @test pd[1, 1] == 0
        @test pd[2, 2] == 0
        pd = pairwise(SqEuclidean(1e-12), X, dims=2)
        @test pd[1, 1] == 0
        @test pd[2, 2] == 0
    end
end

@testset "non-negativity" begin
    X = [0.3 0.3 + eps()]

    @test all(x -> x >= 0, pairwise(SqEuclidean(), X; dims = 2))
    @test all(x -> x >= 0, pairwise(SqEuclidean(), X, X; dims = 2))
    @test all(x -> x >= 0, pairwise(Euclidean(), X; dims = 2))
    @test all(x -> x >= 0, pairwise(Euclidean(), X, X; dims = 2))
    @test all(x -> x >= 0, pairwise(WeightedSqEuclidean([1.0]), X; dims = 2))
    @test all(x -> x >= 0, pairwise(WeightedSqEuclidean([1.0]), X, X; dims = 2))
    @test all(x -> x >= 0, pairwise(SqMahalanobis(ones(1, 1)), X; dims = 2))
    @test all(x -> x >= 0, pairwise(SqMahalanobis(ones(1, 1)), X, X; dims = 2))
end

@testset "Bregman Divergence" begin
    # Some basic tests.
    @test_throws ArgumentError bregman(x -> x, x -> 2*x, [1, 2, 3], [1, 2, 3])
    # Test if Bregman() correctly implements the gkl divergence between two random vectors.
    F(p) = LinearAlgebra.dot(p, log.(p));
    ∇F(p) = map(x -> log(x) + 1, p)
    testDist = Bregman(F, ∇F)
    p = rand(4)
    q = rand(4)
    p = p/sum(p)
    q = q/sum(q)
    for (p, q) in ((p, q),
                   (Iterators.take(p, 4), Iterators.take(q, 4)),
                   ((p[i] for i in 1:4), (q[i] for i in 1:4)),
                   )
        @test testDist(p, q) ≈ gkl_divergence(p, q)
        # Test if Bregman() correctly implements the squared euclidean dist. between them.
        @test bregman(x -> norm(x)^2, x -> 2 .* x, p, q) ≈ sqeuclidean(p, q)
    end
    # Test if Bregman() correctly implements the IS distance.
    G(p) = -1 * sum(log.(p))
    ∇G(p) = map(x -> -1 * x^(-1), p)
    function ISdist(p::AbstractVector, q::AbstractVector)
        return sum([p[i]/q[i] - log(p[i]/q[i]) - 1 for i in 1:length(p)])
    end
    @test bregman(G, ∇G, p, q) ≈ ISdist(p, q)
end

@testset "Unitful vectors" begin
    x = [1m, 2m, 3m]; y = [2m, 3m, 4m]; w = [1, 1, 1]; p = [2m, 2m, 2m]
    @test @inferred sqeuclidean(x, y) == 3m^2
    @test @inferred euclidean(x, y) == sqrt(3)m
    @test @inferred cityblock(x, y) == 3m
    @test @inferred totalvariation(x, y) == 1.5m
    @test @inferred chebyshev(x, y) == 1m
    @test @inferred minkowski(x, y, 2) == sqrt(3)m
    @test @inferred jaccard(x, y) == 1 - sum(min.(x, y)) / sum(max.(x, y))
    @test @inferred braycurtis(x, y) == sum(abs.(x .- y)) / sum(abs.(x .+ y))
    @test @inferred cosine_dist(x, y) == 1 - dot(x, y) / (norm(x) * norm(y))
    @test @inferred corr_dist(x, y) == cosine_dist(x .- mean(x), y .- mean(y))
    @test @inferred chisq_dist(x, y) == sum((x .- y).^2 ./ (x .+ y))
    @test @inferred spannorm_dist(x, y) == 0m
    @test @inferred hellinger(x, y) == sqrt(1 - sum(sqrt.(x .* y) / sqrt(sum(x) * sum(y))))
    @test @inferred meanad(x, y) == 1m
    @test @inferred msd(x, y) == 1m^2
    @test @inferred rmsd(x, y) == 1m
    @test @inferred nrmsd(x, y) == rmsd(x, y) / (maximum(x) - minimum(x))
    @test @inferred weuclidean(x, y, w) == euclidean(x, y)
    @test @inferred wsqeuclidean(x, y, w) == sqeuclidean(x, y)
    @test @inferred wcityblock(x, y, w) == cityblock(x, y)
    @test @inferred wminkowski(x, y, w, 2) == euclidean(x, y)
    @test @inferred whamming(x, y, w) == hamming(x, y)
    @test @inferred peuclidean(x, y, p) == sqrt(3)m
end

#=
@testset "zero allocation colwise!" begin
    d = Euclidean()
    a = rand(2, 41)
    b = rand(2, 41)
    z = zeros(41)
    colwise!(z, d, a, b)
    # This fails when bounds checking is enforced
    bounds = Base.JLOptions().check_bounds
    if bounds == 0
        @test (@allocated colwise!(z, d, a, b)) == 0
    else
        @test_broken (@allocated colwise!(z, d, a, b)) == 0
    end
end
=#
