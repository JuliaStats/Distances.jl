# Distances.jl

[![Build Status](https://travis-ci.org/JuliaStats/Distances.jl.svg?branch=master)](https://travis-ci.org/JuliaStats/Distances.jl)
[![Coverage Status](https://coveralls.io/repos/JuliaStats/Distances.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/JuliaStats/Distances.jl?branch=master)

[![Distances](http://pkg.julialang.org/badges/Distances_0.5.svg)](http://pkg.julialang.org/?pkg=Distances)

A Julia package for evaluating distances(metrics) between vectors.

This package also provides optimized functions to compute column-wise and pairwise distances, which are often substantially faster than a straightforward loop implementation. (See the benchmark section below for details).

## Supported distances

* Euclidean distance
* Squared Euclidean distance
* Cityblock distance
* Jaccard distance
* Rogers-Tanimoto distance
* Chebyshev distance
* Minkowski distance
* Hamming distance
* Cosine distance
* Correlation distance
* Chi-square distance
* Kullback-Leibler divergence
* Rényi divergence
* Jensen-Shannon divergence
* Mahalanobis distance
* Squared Mahalanobis distance
* Bhattacharyya distance
* Hellinger distance

For ``Euclidean distance``, ``Squared Euclidean distance``, ``Cityblock distance``, ``Minkowski distance``, and ``Hamming distance``, a weighted version is also provided.

## Basic Use

The library supports three ways of computation: *computing the distance between two vectors*, *column-wise computation*, and *pairwise computation*.


#### Computing the distance between two vectors

Each distance corresponds to a *distance type*. You can always compute a certain distance between two vectors using the following syntax

```julia
r = evaluate(dist, x, y)
```

Here, dist is an instance of a distance type. For example, the type for Euclidean distance is ``Euclidean`` (more distance types will be introduced in the next section), then you can compute the Euclidean distance between ``x`` and ``y`` as

```julia
r = evaluate(Euclidean(), x, y)
```

Common distances also come with convenient functions for distance evaluation. For example, you may also compute Euclidean distance between two vectors as below

```julia
r = euclidean(x, y)
```

#### Computing distances between corresponding columns

Suppose you have two ``m-by-n`` matrix ``X`` and ``Y``, then you can compute all distances between corresponding columns of X and Y in one batch, using the ``colwise`` function, as

```julia
r = colwise(dist, X, Y)
```

The output ``r`` is a vector of length ``n``. In particular, ``r[i]`` is the distance between ``X[:,i]`` and ``Y[:,i]``. The batch computation typically runs considerably faster than calling ``evaluate`` column-by-column.

Note that either of ``X`` and ``Y`` can be just a single vector -- then the ``colwise`` function will compute the distance between this vector and each column of the other parameter.

#### Computing pairwise distances

Let ``X`` and ``Y`` respectively have ``m`` and ``n`` columns. Then the ``pairwise`` function computes distances between each pair of columns in ``X`` and ``Y``:

```julia
R = pairwise(dist, X, Y)
```

In the output, ``R`` is a matrix of size ``(m, n)``, such that ``R[i,j]`` is the distance between ``X[:,i]`` and ``Y[:,j]``. Computing distances for all pairs using ``pairwise`` function is often remarkably faster than evaluting for each pair individually.

If you just want to just compute distances between columns of a matrix ``X``, you can write

```julia
R = pairwise(dist, X)
```

This statement will result in an ``m-by-m`` matrix, where ``R[i,j]`` is the distance between ``X[:,i]`` and ``X[:,j]``.
``pairwise(dist, X)`` is typically more efficient than ``pairwise(dist, X, X)``, as the former will take advantage of the symmetry when ``dist`` is a semi-metric (including metric).


#### Computing column-wise and pairwise distances inplace

If the vector/matrix to store the results are pre-allocated, you may use the storage (without creating a new array) using the following syntax:

```julia
colwise!(r, dist, X, Y)
pairwise!(R, dist, X, Y)
pairwise!(R, dist, X)
```

Please pay attention to the difference, the functions for inplace computation are ``colwise!`` and ``pairwise!`` (instead of ``colwise`` and ``pairwise``).



## Distance type hierarchy

The distances are organized into a type hierarchy.

At the top of this hierarchy is an abstract class **PreMetric**, which is defined to be a function ``d`` that satisfies

	d(x, x) == 0  for all x
	d(x, y) >= 0  for all x, y

**SemiMetric** is a abstract type that refines **PreMetric**. Formally, a *semi-metric* is a *pre-metric* that is also symmetric, as

	d(x, y) == d(y, x)  for all x, y

**Metric** is a abstract type that further refines **SemiMetric**. Formally, a *metric* is a *semi-metric* that also satisfies triangle inequality, as

	d(x, z) <= d(x, y) + d(y, z)  for all x, y, z

This type system has practical significance. For example, when computing pairwise distances between a set of vectors, you may only perform computation for half of the pairs, and derive the values immediately for the remaining halve by leveraging the symmetry of *semi-metrics*.

Each distance corresponds to a distance type. The type name and the corresponding mathematical definitions of the distances are listed in the following table.

| type name            |  convenient syntax         | math definition     |
| -------------------- | -------------------------- | --------------------|
|  Euclidean           |  `euclidean(x, y)`         | `sqrt(sum((x - y) .^ 2))` |
|  SqEuclidean         |  `sqeuclidean(x, y)`       | `sum((x - y).^2)` |
|  Cityblock           |  `cityblock(x, y)`         | `sum(abs(x - y))` |
|  Chebyshev           |  `chebyshev(x, y)`         | `max(abs(x - y))` |
|  Minkowski           |  `minkowski(x, y, p)`      | `sum(abs(x - y).^p) ^ (1/p)` |
|  Hamming             |  `hamming(k, l)`           | `sum(k .!= l)` |
|  Rogers-Tanimoto     |  `rogerstanimoto(a, b)`    | `2(sum(a&!b) + sum(!a&b)) / (2(sum(a&!b) + sum(!a&b)) + sum(a&b) + sum(!a&!b))` |
|  Jaccard             |  `jaccard(x, y)`           | `1 - sum(min(x, y)) / sum(max(x, y))` |
|  CosineDist          |  `cosine_dist(x, y)`       | `1 - dot(x, y) / (norm(x) * norm(y))` |
|  CorrDist            |  `corr_dist(x, y)`         | `cosine_dist(x - mean(x), y - mean(y))` |
|  ChiSqDist           |  `chisq_dist(x, y)`        | `sum((x - y).^2 / (x + y))` |
|  KLDivergence        |  `kl_divergence(p, q)`     | `sum(p .* log(p ./ q))` |
|  RenyiDivergence     | `renyi_divergence(p, q, k)`| `log(sum( p .* (p ./ q) .^ (k - 1))) / (k - 1)` |
|  JSDivergence        |  `js_divergence(p, q)`     | `KL(p, m) / 2 + KL(p, m) / 2 with m = (p + q) / 2` |
|  SpanNormDist        |  `spannorm_dist(x, y)`     | `max(x - y) - min(x - y )` |
|  BhattacharyyaDist   |  `bhattacharyya(x, y)`     | `-log(sum(sqrt(x .* y) / sqrt(sum(x) * sum(y)))` |
|  HellingerDist       |  `hellinger(x, y) `        | `sqrt(1 - sum(sqrt(x .* y) / sqrt(sum(x) * sum(y))))` |
|  Mahalanobis         |  `mahalanobis(x, y, Q)`    | `sqrt((x - y)' * Q * (x - y))` |
|  SqMahalanobis       |  `sqmahalanobis(x, y, Q)`  | ` (x - y)' * Q * (x - y)`  |
|  WeightedEuclidean   |  `weuclidean(x, y, w)`     | `sqrt(sum((x - y).^2 .* w))`  |
|  WeightedSqEuclidean |  `wsqeuclidean(x, y, w)`   | `sum((x - y).^2 .* w)`  |
|  WeightedCityblock   |  `wcityblock(x, y, w)`     | `sum(abs(x - y) .* w)`  |
|  WeightedMinkowski   |  `wminkowski(x, y, w, p)`  | `sum(abs(x - y).^p .* w) ^ (1/p)` |
|  WeightedHamming     |  `whamming(x, y, w)`       | `sum((x .!= y) .* w)`  |

**Note:** The formulas above are using *Julia*'s functions. These formulas are mainly for conveying the math concepts in a concise way. The actual implementation may use a faster way. x and y arguments are arrays of real numbers; k and l arguments are arrays of distinct elements of any kind; a and b arguments are arrays of `Bool`s; and p and q arguments are arrays forming a discrete probability distribution and are therefore expected to sum to one.

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


The implementation has been carefully optimized based on benchmarks. The Julia scripts ``test/bench_colwise.jl`` and ``test/bench_pairwise.jl`` run the benchmarks on a variety of distances, respectively under column-wise and pairwise settings.

Here are benchmarks obtained on Linux with Intel Core i7-4770K 3.5 GHz.

#### Column-wise benchmark

The table below compares the performance (measured in terms of average elapsed time of each iteration) of a straightforward loop implementation and an optimized implementation provided in *Distances.jl*. The task in each iteration is to compute a specific distance between corresponding columns in two ``200-by-10000`` matrices.

|  distance  |  loop  |  colwise  |  gain  |
|----------- | -------| ----------| -------|
| SqEuclidean | 0.007464s |  0.001993s |  3.7459 |
| Euclidean | 0.007294s |  0.002035s |  3.5840 |
| Cityblock | 0.007411s |  0.001991s |  3.7231 |
| Chebyshev | 0.011455s |  0.005540s |  2.0678 |
| Minkowski | 0.023713s |  0.016705s |  1.4195 |
| Hamming | 0.006996s |  0.001985s |  3.5250 |
| CosineDist | 0.008863s |  0.003301s |  2.6850 |
| CorrDist | 0.014315s |  0.016693s |  0.8575 |
| ChiSqDist | 0.008136s |  0.002109s |  3.8578 |
| KLDivergence | 0.035595s |  0.032610s |  1.0916 |
| RenyiDivergence(0) | 0.014064s |  0.009562s |  1.4707 |
| RenyiDivergence(1) | 0.042286s |  0.039520s |  1.0700 |
| RenyiDivergence(2) | 0.017232s |  0.013540s |  1.2727 |
| RenyiDivergence(∞) | 0.015273s |  0.010613s |  1.4391 |
| JSDivergence | 0.085558s |  0.075936s |  1.1267 |
| BhattacharyyaDist | 0.009933s |  0.004510s |  2.2023 |
| HellingerDist | 0.010375s |  0.003817s |  2.7184 |
| WeightedSqEuclidean | 0.007462s |  0.002019s |  3.6957 |
| WeightedEuclidean | 0.007483s |  0.002075s |  3.6067 |
| WeightedCityblock | 0.007437s |  0.002034s |  3.6563 |
| WeightedMinkowski | 0.023364s |  0.017775s |  1.3144 |
| WeightedHamming | 0.009027s |  0.002754s |  3.2776 |
| SqMahalanobis | 0.103410s |  0.032976s |  3.1359 |
| Mahalanobis | 0.104362s |  0.033305s |  3.1336 |

We can see that using ``colwise`` instead of a simple loop yields considerable gain (2x - 4x), especially when the internal computation of each distance is simple. Nonetheless, when the computaton of a single distance is heavy enough (e.g. *KLDivergence*,  *RenyiDivergence*), the gain is not as significant.

#### Pairwise benchmark

The table below compares the performance (measured in terms of average elapsed time of each iteration) of a straightforward loop implementation and an optimized implementation provided in *Distances.jl*. The task in each iteration is to compute a specific distance in a pairwise manner between columns in a ``100-by-200`` and ``100-by-250`` matrices, which will result in a ``200-by-250`` distance matrix.

|  distance  |  loop  |  pairwise |  gain  |
|----------- | -------| ----------| -------|
| SqEuclidean | 0.022127s |  0.000145s | **152.1941** |
| Euclidean | 0.021477s |  0.000844s | **25.4365** |
| Cityblock | 0.021622s |  0.004129s |  5.2366 |
| Chebyshev | 0.033059s |  0.015156s |  2.1813 |
| Minkowski | 0.063735s |  0.046181s |  1.3801 |
| Hamming | 0.020737s |  0.003304s |  6.2757 |
| CosineDist | 0.025623s |  0.000831s | **30.8470** |
| CorrDist | 0.035662s |  0.000888s | **40.1586** |
| ChiSqDist | 0.021997s |  0.004236s |  5.1928 |
| KLDivergence | 0.094585s |  0.083738s |  1.1295 |
| RenyiDivergence(0) | 0.041105s |  0.022306s |  1.8428 |
| RenyiDivergence(1) | 0.112891s |  0.100909s |  1.1187 |
| RenyiDivergence(2) | 0.048346s |  0.031279s |  1.5456 |
| RenyiDivergence(∞) | 0.042273s |  0.026941s |  1.5691 |
| JSDivergence | 0.203695s |  0.195379s |  1.0426 |
| BhattacharyyaDist | 0.029058s |  0.010801s |  2.6904 |
| HellingerDist | 0.027940s |  0.009818s |  2.8458 |
| WeightedSqEuclidean | 0.022282s |  0.000173s | **128.5042** |
| WeightedEuclidean | 0.022500s |  0.000268s | **84.1009** |
| WeightedCityblock | 0.022641s |  0.004501s |  5.0296 |
| WeightedMinkowski | 0.066117s |  0.047862s |  1.3814 |
| WeightedHamming | 0.027153s |  0.007034s |  3.8601 |
| SqMahalanobis | 0.338031s |  0.000863s | **391.9013** |
| Mahalanobis | 0.341765s |  0.000953s | **358.5008** |

For distances of which a major part of the computation is a quadratic form (e.g. *Euclidean*, *CosineDist*, *Mahalanobis*), the performance can be drastically improved by restructuring the computation and delegating the core part to ``GEMM`` in *BLAS*. The use of this strategy can easily lead to 100x performance gain over simple loops (see the highlighted part of the table above).
