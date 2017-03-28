# Unit tests for Distances

@testset "individual metrics" begin

a = 1
b = 2
@test sqeuclidean(a, a) == 0.
@test sqeuclidean(a, b) == 1.

@test euclidean(a, a) == 0.
@test euclidean(a, b) == 1.

@test cityblock(a, a) == 0.
@test cityblock(a, b) == 1.

@test chebyshev(a, a) == 0.
@test chebyshev(a, b) == 1.

@test chebyshev(a, a) == 0.

@test minkowski(a, a, 2) == 0.
@test minkowski(a, b, 2) == 1.

@test hamming(a, a) == 0
@test hamming(a, b) == 1

bt = [true, false, true]
bf = [false, true, true]
@test rogerstanimoto(bt, bt) == 0
@test rogerstanimoto(bt, bf) == 4./5

for (x, y) in (([4., 5., 6., 7.], [3., 9., 8., 1.]),
                ([4., 5., 6., 7.], [3. 8.; 9. 1.]))
    @test sqeuclidean(x, x) == 0.
    @test sqeuclidean(x, y) == 57.

    @test euclidean(x, x) == 0.
    @test euclidean(x, y) == sqrt(57.)

    @test jaccard(x, x) == 0
    @test jaccard(x, y) == 13./28

    @test cityblock(x, x) == 0.
    @test cityblock(x, y) == 13.

    @test chebyshev(x, x) == 0.
    @test chebyshev(x, y) == 6.

    @test minkowski(x, x, 2) == 0.
    @test minkowski(x, y, 2) == sqrt(57.)

    @test cosine_dist(x, x) ≈ 0.0

    @test_throws DimensionMismatch cosine_dist(1.:2, 1.:3)
    @test cosine_dist(x, y) ≈ (1.0 - 112. / sqrt(19530.))
    x_int, y_int = map(Int64, x), map(Int64, y)
    @test cosine_dist(x_int, y_int) == (1.0 - 112. / sqrt(19530.))

    @test corr_dist(x, x) < 1.0e-14
    @test corr_dist(x, y) ≈ cosine_dist(x .- mean(x), vec(y) .- mean(y))

    @test chisq_dist(x, x) == 0.
    @test chisq_dist(x, y) == sum((x - vec(y)).^2 ./ (x + vec(y)))

    @test spannorm_dist(x, x) == 0.
    @test spannorm_dist(x, y) == maximum(x - vec(y)) - minimum(x - vec(y))

    @test gkl_divergence(x, y) ≈ sum(i -> x[i] * log(x[i] / y[i]) - x[i] + y[i], 1:length(x))

    @test meanad(x, y) ≈ mean(Float64[abs(x[i] - y[i]) for i in 1:length(x)])
    @test msd(x, y) ≈ mean(Float64[abs2(x[i] - y[i]) for i in 1:length(x)])
    @test rmsd(x, y) ≈ sqrt(msd(x, y))
    @test nrmsd(x, y) ≈ sqrt(msd(x, y)) / (maximum(x) - minimum(x))
    @test cvrmsd(x, y) ≈ sqrt(msd(x, y)) / mean(x)

    w = ones(4)
    @test sqeuclidean(x, y) ≈ wsqeuclidean(x, y, w)


    w = rand(size(x))

    @test wsqeuclidean(x, x, w) == 0.
    @test wsqeuclidean(x, y, w) ≈ dot((x - vec(y)).^2, w)

    @test weuclidean(x, x, w) == 0.
    @test weuclidean(x, y, w) == sqrt(wsqeuclidean(x, y, w))

    @test wcityblock(x, x, w) == 0.
    @test wcityblock(x, y, w) ≈ dot(abs.(x - vec(y)), w)

    @test wminkowski(x, x, w, 2) == 0.
    @test wminkowski(x, y, w, 2) ≈ weuclidean(x, y, w)
end

# Test weighted Hamming distances with even weights
a = [1., 2., 1., 3., 2., 1.]
b = [1., 3., 0., 2., 2., 0.]
w = rand(size(a))

@test whamming(a, a, w) == 0.
@test whamming(a, b, w) == sum((a .!= b) .* w)

# Minimal test of Jaccard - test return type stability.
@inferred evaluate(Jaccard(), rand(3), rand(3))
@inferred evaluate(Jaccard(), [1,2,3], [1,2,3])
@inferred evaluate(Jaccard(), [true, false, true], [false, true, true])

# Test KL, Renyi and JS divergences
p = r = rand(12)
p[p .< 0.3] = 0.0
scale = sum(p) / sum(r)
r /= sum(r)
p /= sum(p)
q = rand(12)
q /= sum(q)

klv = 0.
for i = 1 : length(p)
    if p[i] > 0
        klv += p[i] * log(p[i] / q[i])
    end
end
@test kl_divergence(p, q) ≈ klv

@test renyi_divergence(p, p, 0) ≈ 0
@test renyi_divergence(p, p, 1) ≈ 0
@test renyi_divergence(p, p, rand()) ≈ 0
@test renyi_divergence(p, p, 1.0 + rand()) ≈ 0
@test renyi_divergence(p, p, Inf) ≈ 0
@test renyi_divergence(p, r, 0) ≈ -log(scale)
@test renyi_divergence(p, r, 1) ≈ -log(scale)
@test renyi_divergence(p, r, rand()) ≈ -log(scale)
@test renyi_divergence(p, r, Inf) ≈ -log(scale)
@test isinf(renyi_divergence([0.0, 0.5, 0.5], [0.0, 1.0, 0.0], Inf))
@test renyi_divergence([0.0, 1.0, 0.0], [0.0, 0.5, 0.5], Inf) ≈ log(2.0)
@test renyi_divergence(p, q, 1) ≈ kl_divergence(p, q)

pm = (p + q) / 2
jsv = kl_divergence(p, pm) / 2 + kl_divergence(q, pm) / 2
@test js_divergence(p, p) ≈ 0.0
@test js_divergence(p, q) ≈ jsv


end # testset


@testset "NaN behavior" begin

a = [NaN, 0]; b = [0, NaN]
@test isnan(chebyshev(a, b)) == isnan(maximum(a-b))
a = [NaN, 0]; b = [0, 1]
@test isnan(chebyshev(a, b)) == isnan(maximum(a-b))
@test !isnan(renyi_divergence([0.5, 0.0, 0.5], [0.5, NaN, 0.5], 2))
@test isnan(renyi_divergence([0.5, 0.0, 0.5], [0.5, 0.5, NaN], 2))
end #testset


@testset "empty vector" begin

a = Float64[]
b = Float64[]
@test sqeuclidean(a, b) == 0.
@test isa(sqeuclidean(a, b), Float64)
@test euclidean(a, b) == 0.
@test isa(euclidean(a, b), Float64)
@test cityblock(a, b) == 0.
@test isa(cityblock(a, b), Float64)
@test chebyshev(a, b) == 0
@test isa(chebyshev(a, b), Float64)
@test minkowski(a, b, 2) == 0.
@test isa(minkowski(a, b, 2), Float64)
@test hamming(a, b) == 0.0
@test isa(hamming(a, b), Int)
@test renyi_divergence(a, b, 1.0) == 0.0
@test isa(renyi_divergence(a, b, 2.0), Float64)

w = Float64[]
@test isa(whamming(a, b, w), Float64)

end # testset


@testset "DimensionMismatch throwing" begin

a = [1, 0]; b = [2]
@test_throws DimensionMismatch sqeuclidean(a, b)
a = [1, 0]; b = [2.0] ; w = [3.0]
@test_throws DimensionMismatch wsqeuclidean(a, b, w)
a = [1, 0]; b = [2.0, 4.0] ; w = [3.0]
@test_throws DimensionMismatch wsqeuclidean(a, b, w)

end # testset


@testset "mahalanobis" begin

x, y = [4., 5., 6., 7.], [3., 9., 8., 1.]
a = [1., 2., 1., 3., 2., 1.]
b = [1., 3., 0., 2., 2., 0.]

Q = rand(length(x), length(x))
Q = Q * Q'  # make sure Q is positive-definite
@test sqmahalanobis(x, x, Q) == 0.
@test sqmahalanobis(x, y, Q) ≈ dot(x - y, Q * (x - y))

@test mahalanobis(x, x, Q) == 0.
@test mahalanobis(x, y, Q) == sqrt(sqmahalanobis(x, y, Q))

end #testset


@testset "bhattacharyya / hellinger" begin

x, y = [4., 5., 6., 7.], [3., 9., 8., 1.]
a = [1., 2., 1., 3., 2., 1.]
b = [1., 3., 0., 2., 2., 0.]
p = rand(12)
p[p .< 0.3] = 0.
q = rand(12)

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

end #testset


macro test_colwise(_dist, _x, _y)
    quote
        dist = $(esc(_dist))
        x = $(esc(_x))
        y = $(esc(_y))
        local n = size(x, 2)
        r1 = zeros(n)
        r2 = zeros(n)
        r3 = zeros(n)
        for j = 1 : n
            r1[j] = evaluate(dist, x[:,j], y[:,j])
            r2[j] = evaluate(dist, x[:,1], y[:,j])
            r3[j] = evaluate(dist, x[:,j], y[:,1])
        end
        @test colwise(dist, x, y) ≈ r1
        @test colwise(dist, x[:,1], y) ≈ r2
        @test colwise(dist, x, y[:,1]) ≈ r3
    end
end

@testset "column-wise metrics" begin

m = 5
n = 8
X = rand(m, n)
Y = rand(m, n)
A = rand(1:3, m, n)
B = rand(1:3, m, n)

P = rand(m, n)
Q = rand(m, n)
P[P .< 0.3] = 0.


@test_colwise SqEuclidean() X Y
@test_colwise Euclidean() X Y
@test_colwise Cityblock() X Y
@test_colwise Chebyshev() X Y
@test_colwise Minkowski(2.5) X Y
@test_colwise Hamming() A B

@test_colwise CosineDist() X Y
@test_colwise CorrDist() X Y

@test_colwise ChiSqDist() X Y
@test_colwise KLDivergence() P Q
@test_colwise RenyiDivergence(0.0) P Q
@test_colwise RenyiDivergence(1.0) P Q
@test_colwise RenyiDivergence(Inf) P Q
@test_colwise RenyiDivergence(0.5) P Q
@test_colwise RenyiDivergence(2) P Q
@test_colwise JSDivergence() P Q
@test_colwise SpanNormDist() X Y

@test_colwise BhattacharyyaDist() X Y
@test_colwise HellingerDist() X Y

w = rand(m)

@test_colwise WeightedSqEuclidean(w) X Y
@test_colwise WeightedEuclidean(w) X Y
@test_colwise WeightedCityblock(w) X Y
@test_colwise WeightedMinkowski(w, 2.5) X Y
@test_colwise WeightedHamming(w) A B

Q = rand(m, m)
Q = Q * Q'  # make sure Q is positive-definite

@test_colwise SqMahalanobis(Q) X Y
@test_colwise Mahalanobis(Q) X Y

end # testset


macro test_pairwise(_dist, _x, _y)
    quote
        dist = $(esc(_dist))
        x = $(esc(_x))
        y = $(esc(_y))
        local nx = size(x, 2)
        local ny = size(y, 2)
        rxy = zeros(nx, ny)
        rxx = zeros(nx, nx)
        for j = 1 : ny, i = 1 : nx
            rxy[i, j] = evaluate(dist, x[:,i], y[:,j])
        end
        for j = 1 : nx, i = 1 : nx
            rxx[i, j] = evaluate(dist, x[:,i], x[:,j])
        end
        @test pairwise(dist, x, y) ≈ rxy
        @test pairwise(dist, x) ≈ rxx
    end
end

@testset "pairwise metrics" begin

m = 5
n = 8
nx = 6
ny = 8

X = rand(m, nx)
Y = rand(m, ny)
A = rand(1:3, m, nx)
B = rand(1:3, m, ny)

P = rand(m, nx)
Q = rand(m, ny)


@test_pairwise SqEuclidean() X Y
@test_pairwise Euclidean() X Y
@test_pairwise Cityblock() X Y
@test_pairwise Chebyshev() X Y
@test_pairwise Minkowski(2.5) X Y
@test_pairwise Hamming() A B

@test_pairwise CosineDist() X Y
@test_pairwise CorrDist() X Y

@test_pairwise ChiSqDist() X Y
@test_pairwise KLDivergence() P Q
@test_pairwise RenyiDivergence(0.0) P Q
@test_pairwise RenyiDivergence(1.0) P Q
@test_pairwise RenyiDivergence(Inf) P Q
@test_pairwise RenyiDivergence(0.5) P Q
@test_pairwise RenyiDivergence(2) P Q
@test_pairwise JSDivergence() P Q

@test_pairwise BhattacharyyaDist() X Y
@test_pairwise HellingerDist() X Y

w = rand(m)

@test_pairwise WeightedSqEuclidean(w) X Y
@test_pairwise WeightedEuclidean(w) X Y
@test_pairwise WeightedCityblock(w) X Y
@test_pairwise WeightedMinkowski(w, 2.5) X Y
@test_pairwise WeightedHamming(w) A B

Q = rand(m, m)
Q = Q * Q'  # make sure Q is positive-definite

@test_pairwise SqMahalanobis(Q) X Y
@test_pairwise Mahalanobis(Q) X Y

end #testset

@testset "Euclidean precision" begin
    X = [0.1 0.2; 0.3 0.4; -0.1 -0.1]
    pd = pairwise(Euclidean(1e-12), X, X)
    @test pd[1,1] == 0
    @test pd[2,2] == 0
    pd = pairwise(Euclidean(1e-12), X)
    @test pd[1,1] == 0
    @test pd[2,2] == 0
    pd = pairwise(SqEuclidean(1e-12), X, X)
    @test pd[1,1] == 0
    @test pd[2,2] == 0
    pd = pairwise(SqEuclidean(1e-12), X)
    @test pd[1,1] == 0
    @test pd[2,2] == 0
end
