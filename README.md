# Distances.jl

[![Build Status](https://github.com/JuliaStats/Distances.jl/workflows/CI/badge.svg?branch=master)](https://github.com/JuliaStats/Distances.jl/actions?query=workflow%3ACI+branch%3Amaster)
[![Coverage Status](http://codecov.io/github/JuliaStats/Distances.jl/coverage.svg?branch=master)](http://codecov.io/github/JuliaStats/Distances.jl?branch=master)

A Julia package for evaluating distances (metrics) between vectors.

This package also provides optimized functions to compute column-wise and
pairwise distances, which are often substantially faster than a straightforward
loop implementation. (See the benchmark section below for details).

## Supported distances

* Euclidean distance
* Squared Euclidean distance
* Periodic Euclidean distance
* Cityblock distance
* Total variation distance
* Jaccard distance
* Rogers-Tanimoto distance
* Chebyshev distance
* Minkowski distance
* Hamming distance
* Cosine distance
* Correlation distance
* Chi-square distance
* Kullback-Leibler divergence
* Generalized Kullback-Leibler divergence
* Rényi divergence
* Jensen-Shannon divergence
* Mahalanobis distance
* Squared Mahalanobis distance
* Bhattacharyya distance
* Hellinger distance
* Haversine distance
* Spherical angle distance
* Mean absolute deviation
* Mean squared deviation
* Root mean squared deviation
* Normalized root mean squared deviation
* Bray-Curtis dissimilarity
* Bregman divergence

For `Euclidean distance`, `Squared Euclidean distance`, `Cityblock distance`,
`Minkowski distance`, and `Hamming distance`, a weighted version is also provided.

## Basic use

The library supports three ways of computation: *computing the distance between*
*two iterators/vectors*, *"zip"-wise computation*, and *pairwise computation*.
Each of these computation modes works with arbitrary iterable objects of known
size.

### Computing the distance between two iterators or vectors

Each distance corresponds to a *distance type*. You can always compute a certain
distance between two iterators or vectors of equal length using the following
syntax

```julia
r = evaluate(dist, x, y)
r = dist(x, y)
```

Here, `dist` is an instance of a distance type: for example, the type for Euclidean
distance is `Euclidean` (more distance types will be introduced in the next section).
You can compute the Euclidean distance between `x` and `y` as

```julia
r = evaluate(Euclidean(), x, y)
r = Euclidean()(x, y)
```

Common distances also come with convenient functions for distance evaluation. For
example, you may also compute Euclidean distance between two vectors as below

```julia
r = euclidean(x, y)
```

### Computing distances between corresponding objects ("column-wise")

Suppose you have two `m-by-n` matrix `X` and `Y`, then you can compute all
distances between corresponding columns of `X` and `Y` in one batch, using
the `colwise` function, as

```julia
r = colwise(dist, X, Y)
```

The output `r` is a vector of length `n`. In particular, `r[i]` is the distance
between `X[:,i]` and `Y[:,i]`. The batch computation typically runs considerably
faster than calling `evaluate` column-by-column.

Note that either of `X` and `Y` can be just a single vector -- then the `colwise`
function computes the distance between this vector and each column of the other
argument.

### Computing pairwise distances

Let `X` and `Y` have `m` and `n` columns, respectively, and the same number of rows.
Then the `pairwise` function with the `dims=2` argument computes distances between
each pair of columns in `X` and `Y`:

```julia
R = pairwise(dist, X, Y, dims=2)
```

In the output, `R` is a matrix of size `(m, n)`, such that `R[i,j]` is the
distance between `X[:,i]` and `Y[:,j]`. Computing distances for all pairs using
`pairwise` function is often remarkably faster than evaluting for each pair
individually.

If you just want to just compute distances between all columns of a matrix `X`,
you can write

```julia
R = pairwise(dist, X, dims=2)
```

This statement will result in an `m-by-m` matrix, where `R[i,j]` is the distance
between `X[:,i]` and `X[:,j]`. `pairwise(dist, X)` is typically more efficient
than `pairwise(dist, X, X)`, as the former will take advantage of the symmetry
when `dist` is a semi-metric (including metric).

To compute pairwise distances for matrices with observations stored in rows use
the argument `dims=1`.

### Computing column-wise and pairwise distances inplace

If the vector/matrix to store the results are pre-allocated, you may use the
storage (without creating a new array) using the following syntax
(`i` being either `1` or `2`):

```julia
colwise!(dist, r, X, Y)
pairwise!(dist, R, X, Y, dims=i)
pairwise!(dist, R, X, dims=i)
```

Please pay attention to the difference, the functions for inplace computation are
`colwise!` and `pairwise!` (instead of `colwise` and `pairwise`).

#### Deprecated alternative syntax

The syntax

```julia
colwise!(r, dist, X, Y)
pairwise!(R, dist, X, Y, dims=i)
pairwise!(R, dist, X, dims=i)
```

with the first two arguments (metric and results) interchanged is supported as well.
However, its use is discouraged since
[it is deprecated](https://github.com/JuliaStats/Distances.jl/pull/239) and will be
removed in a future release.

## Distance type hierarchy

The distances are organized into a type hierarchy.

At the top of this hierarchy is an abstract class **PreMetric**, which is defined to be a function `d` that satisfies

    d(x, x) == 0  for all x
    d(x, y) >= 0  for all x, y

**SemiMetric** is a abstract type that refines **PreMetric**. Formally, a *semi-metric* is a *pre-metric* that is also symmetric, as

    d(x, y) == d(y, x)  for all x, y

**Metric** is a abstract type that further refines **SemiMetric**. Formally, a *metric* is a *semi-metric* that also satisfies triangle inequality, as

    d(x, z) <= d(x, y) + d(y, z)  for all x, y, z

This type system has practical significance. For example, when computing pairwise distances
between a set of vectors, you may only perform computation for half of the pairs, derive the
values immediately for the remaining half by leveraging the symmetry of *semi-metrics*. Note
that the types of `SemiMetric` and `Metric` do not completely follow the definition in
mathematics as they do not require the "distance" to be able to distinguish between points:
for these types `x != y` does not imply that `d(x, y) != 0` in general compared to the
mathematical definition of semi-metric and metric, as this property does not change
computations in practice.

Each distance corresponds to a distance type. The type name and the corresponding mathematical
definitions of the distances are listed in the following table.

| type name            |  convenient syntax                | math definition     |
| -------------------- | --------------------------------- | --------------------|
|  Euclidean           |  `euclidean(x, y)`                | `sqrt(sum((x - y) .^ 2))` |
|  SqEuclidean         |  `sqeuclidean(x, y)`              | `sum((x - y).^2)` |
|  PeriodicEuclidean   |  `peuclidean(x, y, w)`            | `sqrt(sum(min(mod(abs(x - y), w), w - mod(abs(x - y), w)).^2))`  |
|  Cityblock           |  `cityblock(x, y)`                | `sum(abs(x - y))` |
|  TotalVariation      |  `totalvariation(x, y)`           | `sum(abs(x - y)) / 2` |
|  Chebyshev           |  `chebyshev(x, y)`                | `max(abs(x - y))` |
|  Minkowski           |  `minkowski(x, y, p)`             | `sum(abs(x - y).^p) ^ (1/p)` |
|  Hamming             |  `hamming(k, l)`                  | `sum(k .!= l)` |
|  RogersTanimoto      |  `rogerstanimoto(a, b)`           | `2(sum(a&!b) + sum(!a&b)) / (2(sum(a&!b) + sum(!a&b)) + sum(a&b) + sum(!a&!b))` |
|  Jaccard             |  `jaccard(x, y)`                  | `1 - sum(min(x, y)) / sum(max(x, y))` |
|  BrayCurtis          |  `braycurtis(x, y)`               | `sum(abs(x - y)) / sum(abs(x + y))`  |
|  CosineDist          |  `cosine_dist(x, y)`              | `1 - dot(x, y) / (norm(x) * norm(y))` |
|  CorrDist            |  `corr_dist(x, y)`                | `cosine_dist(x - mean(x), y - mean(y))` |
|  ChiSqDist           |  `chisq_dist(x, y)`               | `sum((x - y).^2 / (x + y))` |
|  KLDivergence        |  `kl_divergence(p, q)`            | `sum(p .* log(p ./ q))` |
|  GenKLDivergence     |  `gkl_divergence(x, y)`           | `sum(p .* log(p ./ q) - p + q)` |
|  RenyiDivergence     |  `renyi_divergence(p, q, k)`      | `log(sum( p .* (p ./ q) .^ (k - 1))) / (k - 1)` |
|  JSDivergence        |  `js_divergence(p, q)`            | `KL(p, m) / 2 + KL(q, m) / 2 with m = (p + q) / 2` |
|  SpanNormDist        |  `spannorm_dist(x, y)`            | `max(x - y) - min(x - y)` |
|  BhattacharyyaDist   |  `bhattacharyya(x, y)`            | `-log(sum(sqrt(x .* y) / sqrt(sum(x) * sum(y)))` |
|  HellingerDist       |  `hellinger(x, y)`                | `sqrt(1 - sum(sqrt(x .* y) / sqrt(sum(x) * sum(y))))` |
|  Haversine           |  `haversine(x, y, r = 6_371_000)` | [Haversine formula](https://en.wikipedia.org/wiki/Haversine_formula) |
|  SphericalAngle      |  `spherical_angle(x, y)`          | [Haversine formula](https://en.wikipedia.org/wiki/Haversine_formula) |
|  Mahalanobis         |  `mahalanobis(x, y, Q)`           | `sqrt((x - y)' * Q * (x - y))` |
|  SqMahalanobis       |  `sqmahalanobis(x, y, Q)`         | `(x - y)' * Q * (x - y)` |
|  MeanAbsDeviation    |  `meanad(x, y)`                   | `mean(abs.(x - y))` |
|  MeanSqDeviation     |  `msd(x, y)`                      | `mean(abs2.(x - y))` |
|  RMSDeviation        |  `rmsd(x, y)`                     | `sqrt(msd(x, y))` |
|  NormRMSDeviation    |  `nrmsd(x, y)`                    | `rmsd(x, y) / (maximum(x) - minimum(x))` |
|  WeightedEuclidean   |  `weuclidean(x, y, w)`            | `sqrt(sum((x - y).^2 .* w))`  |
|  WeightedSqEuclidean |  `wsqeuclidean(x, y, w)`          | `sum((x - y).^2 .* w)`  |
|  WeightedCityblock   |  `wcityblock(x, y, w)`            | `sum(abs(x - y) .* w)`  |
|  WeightedMinkowski   |  `wminkowski(x, y, w, p)`         | `sum(abs(x - y).^p .* w) ^ (1/p)` |
|  WeightedHamming     |  `whamming(x, y, w)`              | `sum((x .!= y) .* w)`  |
|  Bregman             |  `bregman(F, ∇, x, y; inner=dot)` | `F(x) - F(y) - inner(∇(y), x - y)` |

**Note:** The formulas above are using *Julia*'s functions. These formulas are
mainly for conveying the math concepts in a concise way. The actual implementation
may use a faster way. The arguments `x` and `y` are iterable objects, typically
arrays of real numbers; `w` is an iterator/array of parameters (like weights or
periods); `k` and `l` are iterators/arrays of distinct elements of
any kind; `a` and `b` are iterators/arrays of Bools; and finally, `p` and `q` are
iterators/arrays forming a discrete probability distribution and are therefore
both expected to sum to one.

### Precision for Euclidean and SqEuclidean

For efficiency (see the benchmarks below), `Euclidean` and
`SqEuclidean` make use of BLAS3 matrix-matrix multiplication to
calculate distances.  This corresponds to the following expansion:

```julia
(x-y)^2 == x^2 - 2xy + y^2
```

However, equality is not precise in the presence of roundoff error,
and particularly when `x` and `y` are nearby points this may not be
accurate.  Consequently, `Euclidean` and `SqEuclidean` allow you to
supply a relative tolerance to force recalculation:

```julia
julia> x = reshape([0.1, 0.3, -0.1], 3, 1);

julia> pairwise(Euclidean(), x, x)
1×1 Array{Float64,2}:
 7.45058e-9

julia> pairwise(Euclidean(1e-12), x, x)
1×1 Array{Float64,2}:
 0.0
```

## Benchmarks

The implementation has been carefully optimized based on benchmarks. The script in
`benchmark/benchmarks.jl` defines a benchmark suite for a variety of distances,
under column-wise and pairwise settings.

Here are benchmarks obtained running Julia 1.5 on a computer with a quad-core Intel
Core i5-2300K processor @ 3.2 GHz. Extended versions of the tables below can be
replicated using the script in `benchmark/print_table.jl`.

### Column-wise benchmark

Generically, column-wise distances are computed using a straightforward loop
implementation. For `[Sq]Mahalanobis`, however, specialized methods are
provided in *Distances.jl*, and the table below compares the performance
(measured in terms of average elapsed time of each iteration) of the generic
to the specialized implementation. The task in each iteration is to compute a
specific distance between corresponding columns in two `200-by-10000` matrices.

|  distance     | loop      |  colwise   |  gain       |
|---------------|-----------|------------|-------------|
| SqMahalanobis | 0.089470s |  0.014424s |  **6.2027** |
| Mahalanobis   | 0.090882s |  0.014096s |  **6.4475** |

### Pairwise benchmark

Generically, pairwise distances are computed using a straightforward loop
implementation. For distances of which a major part of the computation is a
quadratic form, however, the performance can be drastically improved by restructuring
the computation and delegating the core part to `GEMM` in *BLAS*. The table below
compares the performance (measured in terms of average elapsed time of each
iteration) of generic to the specialized implementations provided in *Distances.jl*.
The task in each iteration is to compute a specific distance in a pairwise manner
between columns in a `100-by-200` and `100-by-250` matrices, which will result in
a `200-by-250` distance matrix.

|  distance             |  loop     |  pairwise  |  gain  |
|---------------------- | --------- | -----------| -------|
| SqEuclidean           | 0.001273s |  0.000124s | **10.2290** |
| Euclidean             | 0.001445s |  0.000194s |  **7.4529** |
| CosineDist            | 0.001928s |  0.000149s | **12.9543** |
| CorrDist              | 0.016837s |  0.000187s | **90.1854** |
| WeightedSqEuclidean   | 0.001603s |  0.000143s | **11.2119** |
| WeightedEuclidean     | 0.001811s |  0.000238s |  **7.6032** |
| SqMahalanobis         | 0.308990s |  0.000248s | **1248.1892** |
| Mahalanobis           | 0.313415s |  0.000346s | **906.1836** |
