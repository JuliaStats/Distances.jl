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

**Note:** The formulas above are using *Julia*'s functions. These formulas are mainly for conveying the math concepts in a concise way. The actual implementation may use a faster way. The arguments `x` and `y` are arrays of real numbers; `k` and `l` are arrays of distinct elements of any kind; a and b are arrays of Bools; and finally, `p` and `q` are arrays forming a discrete probability distribution and are therefore both expected to sum to one.

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

Here are benchmarks obtained running Julia 0.5.1 on a late-2016 MacBook Pro running MacOS 10.12.3 with an quad-core Intel Core i7 processor @ 2.9 GHz.

#### Column-wise benchmark

The table below compares the performance (measured in terms of average elapsed time of each iteration) of a straightforward loop implementation and an optimized implementation provided in *Distances.jl*. The task in each iteration is to compute a specific distance between corresponding columns in two ``200-by-10000`` matrices.

|  distance  |  loop  |  colwise  |  gain  |
|----------- | -------| ----------| -------|
| SqEuclidean | 0.007267s |  0.002000s |  3.6334 |
| Euclidean | 0.007471s |  0.002042s |  3.6584 |
| Cityblock | 0.007239s |  0.001980s |  3.6556 |
| Chebyshev | 0.011396s |  0.005274s |  2.1606 |
| Minkowski | 0.022127s |  0.017161s |  1.2894 |
| Hamming | 0.006777s |  0.001841s |  3.6804 |
| CosineDist | 0.008709s |  0.003046s |  2.8592 |
| CorrDist | 0.012766s |  0.014199s |  0.8991 |
| ChiSqDist | 0.007321s |  0.002042s |  3.5856 |
| KLDivergence | 0.037239s |  0.033535s |  1.1105 |
| RenyiDivergence(0) | 0.014607s |  0.009587s |  1.5237 |
| RenyiDivergence(1) | 0.044142s |  0.040953s |  1.0779 |
| RenyiDivergence(2) | 0.019056s |  0.012029s |  1.5842 |
| RenyiDivergence(∞) | 0.014469s |  0.010906s |  1.3267 |
| JSDivergence | 0.077435s |  0.081599s |  0.9490 |
| BhattacharyyaDist | 0.009805s |  0.004355s |  2.2514 |
| HellingerDist | 0.010007s |  0.004030s |  2.4832 |
| WeightedSqEuclidean | 0.007435s |  0.002051s |  3.6254 |
| WeightedEuclidean | 0.008217s |  0.002075s |  3.9591 |
| WeightedCityblock | 0.007486s |  0.002058s |  3.6378 |
| WeightedMinkowski | 0.024653s |  0.019632s |  1.2557 |
| WeightedHamming | 0.008467s |  0.002962s |  2.8587 |
| SqMahalanobis | 0.101976s |  0.031780s |  3.2088 |
| Mahalanobis | 0.105060s |  0.031806s |  3.3032 |

We can see that using ``colwise`` instead of a simple loop yields considerable gain (2x - 4x), especially when the internal computation of each distance is simple. Nonetheless, when the computation of a single distance is heavy enough (e.g. *KLDivergence*,  *RenyiDivergence*), the gain is not as significant.

#### Pairwise benchmark

The table below compares the performance (measured in terms of average elapsed time of each iteration) of a straightforward loop implementation and an optimized implementation provided in *Distances.jl*. The task in each iteration is to compute a specific distance in a pairwise manner between columns in a ``100-by-200`` and ``100-by-250`` matrices, which will result in a ``200-by-250`` distance matrix.

|  distance  |  loop  |  pairwise |  gain  |
|----------- | -------| ----------| -------|
| SqEuclidean | 0.022982s |  0.000145s | **158.9554** |
| Euclidean | 0.022155s |  0.000843s | **26.2716** |
| Cityblock | 0.022382s |  0.003899s |  5.7407 |
| Chebyshev | 0.034491s |  0.014600s |  2.3624 |
| Minkowski | 0.065968s |  0.046761s |  1.4107 |
| Hamming | 0.021016s |  0.003139s |  6.6946 |
| CosineDist | 0.024394s |  0.000828s | **29.4478** |
| CorrDist | 0.039089s |  0.000852s | **45.8839** |
| ChiSqDist | 0.022152s |  0.004361s |  5.0793 |
| KLDivergence | 0.096694s |  0.086728s |  1.1149 |
| RenyiDivergence(0) | 0.042658s |  0.023323s |  1.8290 |
| RenyiDivergence(1) | 0.122015s |  0.104527s |  1.1673 |
| RenyiDivergence(2) | 0.052896s |  0.033865s |  1.5620 |
| RenyiDivergence(∞) | 0.039993s |  0.027331s |  1.4632 |
| JSDivergence | 0.211276s |  0.204046s |  1.0354 |
| BhattacharyyaDist | 0.030378s |  0.011189s |  2.7151 |
| HellingerDist | 0.029592s |  0.010109s |  2.9273 |
| WeightedSqEuclidean | 0.025619s |  0.000217s | **117.8128** |
| WeightedEuclidean | 0.023366s |  0.000264s | **88.3711** |
| WeightedCityblock | 0.026213s |  0.004610s |  5.6855 |
| WeightedMinkowski | 0.068588s |  0.050033s |  1.3708 |
| WeightedHamming | 0.025936s |  0.007225s |  3.5895 |
| SqMahalanobis | 0.520046s |  0.000939s | **553.6694** |
| Mahalanobis | 0.480556s |  0.000954s | **503.6009** |

For distances of which a major part of the computation is a quadratic form (e.g. *Euclidean*, *CosineDist*, *Mahalanobis*), the performance can be drastically improved by restructuring the computation and delegating the core part to ``GEMM`` in *BLAS*. The use of this strategy can easily lead to 100x performance gain over simple loops (see the highlighted part of the table above).
