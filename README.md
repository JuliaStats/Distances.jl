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
* Generalized Kullback-Leibler divergence
* Rényi divergence
* Jensen-Shannon divergence
* Mahalanobis distance
* Squared Mahalanobis distance
* Bhattacharyya distance
* Hellinger distance
* Mean absolute deviation
* Mean squared deviation
* Root mean squared deviation
* Normalized root mean squared deviation (both range- and mean-based)

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

| type name            |  convenient syntax           | math definition     |
| -------------------- | ---------------------------- | --------------------|
|  Euclidean           |  `euclidean(x, y)`           | `sqrt(sum((x - y) .^ 2))` |
|  SqEuclidean         |  `sqeuclidean(x, y)`         | `sum((x - y).^2)` |
|  Cityblock           |  `cityblock(x, y)`           | `sum(abs(x - y))` |
|  Chebyshev           |  `chebyshev(x, y)`           | `max(abs(x - y))` |
|  Minkowski           |  `minkowski(x, y, p)`        | `sum(abs(x - y).^p) ^ (1/p)` |
|  Hamming             |  `hamming(x, y)`             | `sum(x .!= y)` |
|  Rogers-Tanimoto     |  `rogerstanimoto(x, y)`      | `2(sum(x&!y) + sum(!x&y)) / (2(sum(x&!y) + sum(!x&y)) + sum(x&y) + sum(!x&!y))` |
|  Jaccard             |  `jaccard(x, y)`             | `1 - sum(min(x, y)) / sum(max(x, y))` |
|  CosineDist          |  `cosine_dist(x, y)`         | `1 - dot(x, y) / (norm(x) * norm(y))` |
|  CorrDist            |  `corr_dist(x, y)`           | `cosine_dist(x - mean(x), y - mean(y))` |
|  ChiSqDist           |  `chisq_dist(x, y)`          | `sum((x - y).^2 / (x + y))` |
|  KLDivergence        |  `kl_divergence(x, y)`       | `sum(p .* log(p ./ q))` |
|  GenKLDivergence     |  `gkl_divergence(x, y)`      | `sum(p .* log(p ./ q) - p + q)` |
|  RenyiDivergence     |  `renyi_divergence(x, y, k)` | `log(sum(x .* (x ./ y) .^ (k - 1))) / (k - 1)` |
|  JSDivergence        |  `js_divergence(x, y)`       | `KL(x, m) / 2 + KL(y, m) / 2 with m = (x + y) / 2` |
|  SpanNormDist        |  `spannorm_dist(x, y)`       | `max(x - y) - min(x - y)` |
|  BhattacharyyaDist   |  `bhattacharyya(x, y)`       | `-log(sum(sqrt(x .* y) / sqrt(sum(x) * sum(y)))` |
|  HellingerDist       |  `hellinger(x, y)`           | `sqrt(1 - sum(sqrt(x .* y) / sqrt(sum(x) * sum(y))))` |
|  Mahalanobis         |  `mahalanobis(x, y, Q)`      | `sqrt((x - y)' * Q * (x - y))` |
|  SqMahalanobis       |  `sqmahalanobis(x, y, Q)`    | `(x - y)' * Q * (x - y)`  |
|  MeanAbsDeviation    |  `meanad(x, y)`              | `mean(abs.(x - y))` |
|  MeanSqDeviation     |  `msd(x, y)`                 | `mean(abs2.(x - y))` |
|  RMSDeviation        |  `rmsd(x, y)`                | `sqrt(msd(x, y))` |
|  NormRMSDeviation    |  `nrmsd(x, y)`               | `rmsd(x, y) / (maximum(x) - minimum(x))` |
|  CVRMSDeviation      |  `cvrmsd(x, y)`              | `rmsd(x, y) / mean(x)` |
|  WeightedEuclidean   |  `weuclidean(x, y, w)`       | `sqrt(sum((x - y).^2 .* w))` |
|  WeightedSqEuclidean |  `wsqeuclidean(x, y, w)`     | `sum((x - y).^2 .* w)` |
|  WeightedCityblock   |  `wcityblock(x, y, w)`       | `sum(abs(x - y) .* w)` |
|  WeightedMinkowski   |  `wminkowski(x, y, w, p)`    | `sum(abs(x - y).^p .* w) ^ (1/p)` |
|  WeightedHamming     |  `whamming(x, y, w)`         | `sum((x .!= y) .* w)` |

**Note:** The formulas above are using *Julia*'s functions. These formulas are mainly for conveying the math concepts in a concise way. The actual implementation may use a faster way.

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
| SqEuclidean | 0.012308s |  0.003860s |  3.1884 |
| Euclidean | 0.012484s |  0.003995s |  3.1246 |
| Cityblock | 0.012463s |  0.003927s |  3.1735 |
| Chebyshev | 0.014897s |  0.005898s |  2.5258 |
| Minkowski | 0.028154s |  0.017812s |  1.5806 |
| Hamming | 0.012200s |  0.003896s |  3.1317 |
| CosineDist | 0.013816s |  0.004670s |  2.9583 |
| CorrDist | 0.023349s |  0.016626s |  1.4044 |
| ChiSqDist | 0.015375s |  0.004788s |  3.2109 |
| KLDivergence | 0.044360s |  0.036123s |  1.2280 |
| JSDivergence | 0.098587s |  0.085595s |  1.1518 |
| BhattacharyyaDist | 0.023103s |  0.013002s |  1.7769 |
| HellingerDist | 0.023329s |  0.012555s |  1.8581 |
| WeightedSqEuclidean | 0.012136s |  0.003758s |  3.2296 |
| WeightedEuclidean | 0.012307s |  0.003789s |  3.2482 |
| WeightedCityblock | 0.012287s |  0.003923s |  3.1321 |
| WeightedMinkowski | 0.029895s |  0.018471s |  1.6185 |
| WeightedHamming | 0.013427s |  0.004082s |  3.2896 |
| SqMahalanobis | 0.121636s |  0.019370s |  6.2796 |
| Mahalanobis | 0.117871s |  0.019939s |  5.9117 |

We can see that using ``colwise`` instead of a simple loop yields considerable gain (2x - 6x), especially when the internal computation of each distance is simple. Nonetheless, when the computaton of a single distance is heavy enough (e.g. *Minkowski* and *JSDivergence*), the gain is not as significant.

#### Pairwise benchmark

The table below compares the performance (measured in terms of average elapsed time of each iteration) of a straightforward loop implementation and an optimized implementation provided in *Distances.jl*. The task in each iteration is to compute a specific distance in a pairwise manner between columns in a ``100-by-200`` and ``100-by-250`` matrices, which will result in a ``200-by-250`` distance matrix.

|  distance  |  loop  |  pairwise |  gain  |
|----------- | -------| ----------| -------|
| SqEuclidean | 0.032179s |  0.000170s | **189.7468** |
| Euclidean | 0.031646s |  0.000326s | **97.1773** |
| Cityblock | 0.031594s |  0.002771s | 11.4032 |
| Chebyshev | 0.036732s |  0.011575s |  3.1735 |
| Minkowski | 0.073685s |  0.047725s |  1.5440 |
| Hamming | 0.030016s |  0.002539s | 11.8236 |
| CosineDist | 0.035426s |  0.000235s | **150.8504** |
| CorrDist | 0.061430s |  0.000341s | **180.1693** |
| ChiSqDist | 0.037702s |  0.011709s |  3.2199 |
| KLDivergence | 0.119043s |  0.086861s |  1.3705 |
| JSDivergence | 0.255449s |  0.227079s |  1.1249 |
| BhattacharyyaDist | 0.059165s |  0.033330s |  1.7751 |
| HellingerDist | 0.056953s |  0.031163s |  1.8276 |
| WeightedSqEuclidean | 0.031781s |  0.000218s | **145.9820** |
| WeightedEuclidean | 0.031365s |  0.000410s | **76.4517** |
| WeightedCityblock | 0.031239s |  0.003242s |  9.6360 |
| WeightedMinkowski | 0.077039s |  0.049319s |  1.5621 |
| WeightedHamming | 0.032584s |  0.005673s |  5.7442 |
| SqMahalanobis | 0.280485s |  0.000297s | **943.6018** |
| Mahalanobis | 0.295715s |  0.000498s | **593.6096** |

For distances of which a major part of the computation is a quadratic form (e.g. *Euclidean*, *CosineDist*, *Mahalanobis*), the performance can be drastically improved by restructuring the computation and delegating the core part to ``GEMM`` in *BLAS*. The use of this strategy can easily lead to 100x performance gain over simple loops (see the highlighted part of the table above).
