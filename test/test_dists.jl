# Unit tests for Distances

using Distances
using Base.Test

# helpers

# is_approx(a::Number, b::Number, tol::Number) = abs(a - b) < tol
# all_approx(a::ContiguousArray, b::ContiguousArray, tol::Number) = size(a) == size(b) && all(abs(a - b) .< tol)


# test individual metrics
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



p = rand(12)
p[p .< 0.3] = 0.
q = rand(12)
a = [1., 2., 1., 3., 2., 1.]
b = [1., 3., 0., 2., 2., 0.]
for (x, y) in (([4., 5., 6., 7.], [3., 9., 8., 1.]),
                ([4., 5., 6., 7.], [3. 8.; 9. 1.]))
    @test sqeuclidean(x, x) == 0.
    @test sqeuclidean(x, y) == 57.

    @test euclidean(x, x) == 0.
    @test euclidean(x, y) == sqrt(57.)

    @test cityblock(x, x) == 0.
    @test cityblock(x, y) == 13.

    @test chebyshev(x, x) == 0.
    @test chebyshev(x, y) == 6.

    @test minkowski(x, x, 2) == 0.
    @test minkowski(x, y, 2) == sqrt(57.)


    @test_approx_eq_eps cosine_dist(x, x) 0.0 1.0e-12
    @test_throws DimensionMismatch cosine_dist(1.:2, 1.:3)
    @test_approx_eq_eps cosine_dist(x, y) (1.0 - 112. / sqrt(19530.)) 1.0e-12

    @test_approx_eq_eps corr_dist(x, x) 0. 1.0e-12
    @test_approx_eq corr_dist(x, y) cosine_dist(x .- mean(x), vec(y) .- mean(y))

    @test chisq_dist(x, x) == 0.
    @test chisq_dist(x, y) == sum((x - vec(y)).^2 ./ (x + vec(y)))

    klv = 0.
    for i = 1 : length(p)
        if p[i] > 0
            klv += p[i] * log(p[i] / q[i])
        end
    end
    @test_approx_eq_eps kl_divergence(p, q) klv 1.0e-12

    pm = (p + q) / 2
    jsv = kl_divergence(p, pm) / 2 + kl_divergence(q, pm) / 2
    @test_approx_eq_eps js_divergence(p, p) 0.0 1.0e-12
    @test_approx_eq_eps js_divergence(p, q) jsv 1.0e-12

    @test spannorm_dist(x, x) == 0.
    @test spannorm_dist(x, y) == maximum(x - vec(y)) - minimum(x - vec(y))



    w = ones(4)
    @test_approx_eq sqeuclidean(x, y) wsqeuclidean(x, y, w)


    w = rand(size(x))

    @test wsqeuclidean(x, x, w) == 0.
    @test_approx_eq_eps wsqeuclidean(x, y, w) dot((x - vec(y)).^2, w) 1.0e-12

    @test weuclidean(x, x, w) == 0.
    @test weuclidean(x, y, w) == sqrt(wsqeuclidean(x, y, w))

    @test wcityblock(x, x, w) == 0.
    @test_approx_eq_eps wcityblock(x, y, w) dot(abs(x - vec(y)), w) 1.0e-12

    @test wminkowski(x, x, w, 2) == 0.
    @test_approx_eq_eps wminkowski(x, y, w, 2) weuclidean(x, y, w) 1.0e-12

    w = rand(size(a))

    @test whamming(a, a, w) == 0.
    @test whamming(a, b, w) == sum((a .!= b) .* w)


end



# test NaN behavior
a = [NaN, 0]; b = [0, NaN]
@test isnan(chebyshev(a, b)) == isnan(maximum(a-b))
a = [NaN, 0]; b = [0, 1]
@test isnan(chebyshev(a, b)) == isnan(maximum(a-b))


# test empty vector
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

w = Float64[]
@test isa(whamming(a, b, w), Float64)



a = [1, 0]; b = [2]
@test_throws DimensionMismatch sqeuclidean(a, b)
a = [1, 0]; b = [2.0] ; w = [3.0]
@test_throws DimensionMismatch wsqeuclidean(a, b, w)
a = [1, 0]; b = [2.0, 4.0] ; w = [3.0]
@test_throws DimensionMismatch wsqeuclidean(a, b, w)







x, y = [4., 5., 6., 7.], [3., 9., 8., 1.]
a = [1., 2., 1., 3., 2., 1.]
b = [1., 3., 0., 2., 2., 0.]
Q = rand(length(x), length(x))
Q = Q * Q'  # make sure Q is positive-definite
@test sqmahalanobis(x, x, Q) == 0.
@test_approx_eq_eps sqmahalanobis(x, y, Q) dot(x - y, Q * (x - y)) 1.0e-12

@test mahalanobis(x, x, Q) == 0.
@test mahalanobis(x, y, Q) == sqrt(sqmahalanobis(x, y, Q))
# Bhattacharyya and Hellinger distances are defined for discrete
# probability distributions so to calculate the expected values
# we need to normalize vectors.
px = x ./ sum(x)
py = y ./ sum(y)
expected_bc_x_y = sum(sqrt(px .* py))
@test_approx_eq_eps Distances.bhattacharyya_coeff(x, y) expected_bc_x_y 1.0e-12
@test_approx_eq_eps bhattacharyya(x, y) (-log(expected_bc_x_y)) 1.0e-12
@test_approx_eq_eps hellinger(x, y) sqrt(1 - expected_bc_x_y) 1.0e-12



pa = a ./ sum(a)
pb = b ./ sum(b)
expected_bc_a_b = sum(sqrt(pa .* pb))
@test_approx_eq_eps Distances.bhattacharyya_coeff(a, b) expected_bc_a_b 1.0e-12
@test_approx_eq_eps bhattacharyya(a, b) (-log(expected_bc_a_b)) 1.0e-12
@test_approx_eq_eps hellinger(a, b) sqrt(1 - expected_bc_a_b) 1.0e-12

pp = p ./ sum(p)
pq = q ./ sum(q)
expected_bc_p_q = sum(sqrt(pp .* pq))
@test_approx_eq_eps Distances.bhattacharyya_coeff(p, q) expected_bc_p_q 1.0e-12
@test_approx_eq_eps bhattacharyya(p, q) (-log(expected_bc_p_q)) 1.0e-12
@test_approx_eq_eps hellinger(p, q) sqrt(1 - expected_bc_p_q) 1.0e-12

# Ensure it is semimetric
@test_approx_eq_eps bhattacharyya(x, y) bhattacharyya(y, x) 1.0e-12



# test column-wise metrics

m = 5
n = 8
X = rand(m, n)
Y = rand(m, n)
A = rand(1:3, m, n)
B = rand(1:3, m, n)

P = rand(m, n)
Q = rand(m, n)
P[P .< 0.3] = 0.

macro test_colwise(dist, x, y, tol)
    quote
        local n = size($x, 2)
        r1 = zeros(n)
        r2 = zeros(n)
        r3 = zeros(n)
        for j = 1 : n
            r1[j] = evaluate($dist, ($x)[:,j], ($y)[:,j])
            r2[j] = evaluate($dist, ($x)[:,1], ($y)[:,j])
            r3[j] = evaluate($dist, ($x)[:,j], ($y)[:,1])
        end
        @test_approx_eq_eps colwise($dist, $x, $y) r1 $tol
        @test_approx_eq_eps colwise($dist, ($x)[:,1], $y) r2 $tol
        @test_approx_eq_eps colwise($dist, $x, ($y)[:,1]) r3 $tol
    end
end

@test_colwise SqEuclidean() X Y 1.0e-12
@test_colwise Euclidean() X Y 1.0e-12
@test_colwise Cityblock() X Y 1.0e-12
@test_colwise Chebyshev() X Y 1.0e-16
@test_colwise Minkowski(2.5) X Y 1.0e-13
@test_colwise Hamming() A B 1.0e-16

@test_colwise CosineDist() X Y 1.0e-12
@test_colwise CorrDist() X Y 1.0e-12

@test_colwise ChiSqDist() X Y 1.0e-12
@test_colwise KLDivergence() P Q 1.0e-13
@test_colwise JSDivergence() P Q 1.0e-13
@test_colwise SpanNormDist() X Y 1.0e-12

@test_colwise BhattacharyyaDist() X Y 1.0e-12
@test_colwise HellingerDist() X Y 1.0e-12

w = rand(m)

@test_colwise WeightedSqEuclidean(w) X Y 1.0e-12
@test_colwise WeightedEuclidean(w) X Y 1.0e-12
@test_colwise WeightedCityblock(w) X Y 1.0e-12
@test_colwise WeightedMinkowski(w, 2.5) X Y 1.0e-12
@test_colwise WeightedHamming(w) A B 1.0e-12

Q = rand(m, m)
Q = Q * Q'  # make sure Q is positive-definite

@test_colwise SqMahalanobis(Q) X Y 1.0e-13
@test_colwise Mahalanobis(Q) X Y 1.0e-13







# test pairwise metrics

nx = 6
ny = 8

X = rand(m, nx)
Y = rand(m, ny)
A = rand(1:3, m, nx)
B = rand(1:3, m, ny)

P = rand(m, nx)
Q = rand(m, ny)

macro test_pairwise(dist, x, y, tol)
    quote
        local nx = size($x, 2)
        local ny = size($y, 2)
        rxy = zeros(nx, ny)
        rxx = zeros(nx, nx)
        for j = 1 : ny, i = 1 : nx
            rxy[i, j] = evaluate($dist, ($x)[:,i], ($y)[:,j])
        end
        for j = 1 : nx, i = 1 : nx
            rxx[i, j] = evaluate($dist, ($x)[:,i], ($x)[:,j])
        end
        @test_approx_eq_eps pairwise($dist, $x, $y) rxy $tol
        @test_approx_eq_eps pairwise($dist, $x) rxx $tol
    end
end

@test_pairwise SqEuclidean() X Y 1.0e-12
@test_pairwise Euclidean() X Y 1.0e-12
@test_pairwise Cityblock() X Y 1.0e-12
@test_pairwise Chebyshev() X Y 1.0e-16
@test_pairwise Minkowski(2.5) X Y 1.0e-12
@test_pairwise Hamming() A B 1.0e-16

@test_pairwise CosineDist() X Y 1.0e-12
@test_pairwise CorrDist() X Y 1.0e-12

@test_pairwise ChiSqDist() X Y 1.0e-12
@test_pairwise KLDivergence() P Q 1.0e-13
@test_pairwise JSDivergence() P Q 1.0e-13

@test_pairwise BhattacharyyaDist() X Y 1.0e-12
@test_pairwise HellingerDist() X Y 1.0e-12

w = rand(m)

@test_pairwise WeightedSqEuclidean(w) X Y 1.0e-12
@test_pairwise WeightedEuclidean(w) X Y 1.0e-12
@test_pairwise WeightedCityblock(w) X Y 1.0e-12
@test_pairwise WeightedMinkowski(w, 2.5) X Y 1.0e-12
@test_pairwise WeightedHamming(w) A B 1.0e-12

Q = rand(m, m)
Q = Q * Q'  # make sure Q is positive-definite

@test_pairwise SqMahalanobis(Q) X Y 1.0e-13
@test_pairwise Mahalanobis(Q) X Y 1.0e-13

