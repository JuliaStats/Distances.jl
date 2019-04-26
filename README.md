# Distances.jl

[![Build Status](https://travis-ci.org/JuliaStats/Distances.jl.svg?branch=master)](https://travis-ci.org/JuliaStats/Distances.jl)
[![Coverage Status](https://coveralls.io/repos/JuliaStats/Distances.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/JuliaStats/Distances.jl?branch=master)

[![Distances](http://pkg.julialang.org/badges/Distances_0.6.svg)](http://pkg.julialang.org/?pkg=Distances)

A Julia package for evaluating distances(metrics) between vectors.

This package also provides optimized functions to compute column-wise and pairwise distances, which are often substantially faster than a straightforward loop implementation. (See the benchmark section below for details).


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
* Mean absolute deviation
* Mean squared deviation
* Root mean squared deviation
* Normalized root mean squared deviation
* Bray-Curtis dissimilarity
* Bregman divergence

For ``Euclidean distance``, ``Squared Euclidean distance``, ``Cityblock distance``, ``Minkowski distance``, and ``Hamming distance``, a weighted version is also provided.


## Basic use

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

Suppose you have two ``m-by-n`` matrix ``X`` and ``Y``, then you can compute all distances between corresponding columns of ``X`` and ``Y`` in one batch, using the ``colwise`` function, as

```julia
r = colwise(dist, X, Y)
```

The output ``r`` is a vector of length ``n``. In particular, ``r[i]`` is the distance between ``X[:,i]`` and ``Y[:,i]``. The batch computation typically runs considerably faster than calling ``evaluate`` column-by-column.

Note that either of ``X`` and ``Y`` can be just a single vector -- then the ``colwise`` function will compute the distance between this vector and each column of the other parameter.

#### Computing pairwise distances

Let ``X`` and ``Y`` respectively have ``m`` and ``n`` columns. Then the ``pairwise`` function with the ``dims=2`` argument computes distances between each pair of columns in ``X`` and ``Y``:

```julia
R = pairwise(dist, X, Y, dims=2)
```

In the output, ``R`` is a matrix of size ``(m, n)``, such that ``R[i,j]`` is the distance between ``X[:,i]`` and ``Y[:,j]``. Computing distances for all pairs using ``pairwise`` function is often remarkably faster than evaluting for each pair individually.

If you just want to just compute distances between columns of a matrix ``X``, you can write

```julia
R = pairwise(dist, X, dims=2)
```

This statement will result in an ``m-by-m`` matrix, where ``R[i,j]`` is the distance between ``X[:,i]`` and ``X[:,j]``.
``pairwise(dist, X)`` is typically more efficient than ``pairwise(dist, X, X)``, as the former will take advantage of the symmetry when ``dist`` is a semi-metric (including metric).

For performance reasons, it is recommended to use matrices with observations in columns (as shown above). Indeed,
the ``Array`` type in Julia is column-major, making it more efficient to access memory column by column. However,
matrices with observations stored in rows are also supported via the argument ``dims=1``.

#### Computing column-wise and pairwise distances inplace

If the vector/matrix to store the results are pre-allocated, you may use the storage (without creating a new array) using the following syntax (``i`` being either ``1`` or ``2``):

```julia
colwise!(r, dist, X, Y)
pairwise!(R, dist, X, Y, dims=i)
pairwise!(R, dist, X, dims=i)
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
|  PeriodicEuclidean   |  `peuclidean(x, y, p)`     | `sqrt(sum(min(mod(abs(x - y), p), p - mod(abs(x - y), p)).^2))`  |
|  Cityblock           |  `cityblock(x, y)`         | `sum(abs(x - y))` |
|  TotalVariation      |  `totalvariation(x, y)`    | `sum(abs(x - y)) / 2` |
|  Chebyshev           |  `chebyshev(x, y)`         | `max(abs(x - y))` |
|  Minkowski           |  `minkowski(x, y, p)`      | `sum(abs(x - y).^p) ^ (1/p)` |
|  Hamming             |  `hamming(k, l)`           | `sum(k .!= l)` |
|  RogersTanimoto      |  `rogerstanimoto(a, b)`    | `2(sum(a&!b) + sum(!a&b)) / (2(sum(a&!b) + sum(!a&b)) + sum(a&b) + sum(!a&!b))` |
|  Jaccard             |  `jaccard(x, y)`           | `1 - sum(min(x, y)) / sum(max(x, y))` |
|  BrayCurtis          |  `braycurtis(x, y)`        | `sum(abs(x - y)) / sum(abs(x + y))`  |
|  CosineDist          |  `cosine_dist(x, y)`       | `1 - dot(x, y) / (norm(x) * norm(y))` |
|  CorrDist            |  `corr_dist(x, y)`         | `cosine_dist(x - mean(x), y - mean(y))` |
|  ChiSqDist           |  `chisq_dist(x, y)`        | `sum((x - y).^2 / (x + y))` |
|  KLDivergence        |  `kl_divergence(p, q)`     | `sum(p .* log(p ./ q))` |
|  GenKLDivergence     |  `gkl_divergence(x, y)`    | `sum(p .* log(p ./ q) - p + q)` |
|  RenyiDivergence     | `renyi_divergence(p, q, k)`| `log(sum( p .* (p ./ q) .^ (k - 1))) / (k - 1)` |
|  JSDivergence        |  `js_divergence(p, q)`     | `KL(p, m) / 2 + KL(p, m) / 2 with m = (p + q) / 2` |
|  SpanNormDist        |  `spannorm_dist(x, y)`     | `max(x - y) - min(x - y)` |
|  BhattacharyyaDist   |  `bhattacharyya(x, y)`     | `-log(sum(sqrt(x .* y) / sqrt(sum(x) * sum(y)))` |
|  HellingerDist       |  `hellinger(x, y) `        | `sqrt(1 - sum(sqrt(x .* y) / sqrt(sum(x) * sum(y))))` |
|  Haversine           |  `haversine(x, y, r)`      | [Haversine formula](https://en.wikipedia.org/wiki/Haversine_formula) |
|  Mahalanobis         |  `mahalanobis(x, y, Q)`    | `sqrt((x - y)' * Q * (x - y))` |
|  SqMahalanobis       |  `sqmahalanobis(x, y, Q)`  | `(x - y)' * Q * (x - y)` |
|  MeanAbsDeviation    |  `meanad(x, y)`            | `mean(abs.(x - y))` |
|  MeanSqDeviation     |  `msd(x, y)`               | `mean(abs2.(x - y))` |
|  RMSDeviation        |  `rmsd(x, y)`              | `sqrt(msd(x, y))` |
|  NormRMSDeviation    |  `nrmsd(x, y)`             | `rmsd(x, y) / (maximum(x) - minimum(x))` |
|  WeightedEuclidean   |  `weuclidean(x, y, w)`     | `sqrt(sum((x - y).^2 .* w))`  |
|  WeightedSqEuclidean |  `wsqeuclidean(x, y, w)`   | `sum((x - y).^2 .* w)`  |
|  WeightedCityblock   |  `wcityblock(x, y, w)`     | `sum(abs(x - y) .* w)`  |
|  WeightedMinkowski   |  `wminkowski(x, y, w, p)`  | `sum(abs(x - y).^p .* w) ^ (1/p)` |
|  WeightedHamming     |  `whamming(x, y, w)`       | `sum((x .!= y) .* w)`  |
|  Bregman             |  `bregman(F, ∇, x, y; inner = LinearAlgebra.dot)` | `F(x) - F(y) - inner(∇(y), x - y)` |

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

The implementation has been carefully optimized based on benchmarks. The script in `benchmark/benchmarks.jl` defines a benchmark suite
for a variety of distances, under column-wise and pairwise settings.

Here are benchmarks obtained running Julia 1.0 on a computer with a dual-core Intel Core i5-2300K processor @ 2.3 GHz.
The tables below can be replicated using the script in `benchmark/print_table.jl`.

#### Column-wise benchmark

The table below compares the performance (measured in terms of average elapsed time of each iteration) of a straightforward loop implementation and an optimized implementation provided in *Distances.jl*. The task in each iteration is to compute a specific distance between corresponding columns in two ``200-by-10000`` matrices.

|  distance  |  loop  |  colwise  |  gain  |
|----------- | -------| ----------| -------|
| SqEuclidean | 0.004451s |  0.001042s |  4.2715 |
| Euclidean | 0.004750s |  0.001055s |  4.5026 |
| PeriodicEuclidean | 0.011670s |  0.006915s |  1.6876 |
| Cityblock | 0.004386s |  0.001054s |  4.1600 |
| TotalVariation | 0.004409s |  0.001059s |  4.1630 |
| Chebyshev | 0.008783s |  0.004980s |  1.7638 |
| Minkowski | 0.047764s |  0.042185s |  1.1323 |
| Hamming | 0.004681s |  0.001087s |  4.3042 |
| CosineDist | 0.006289s |  0.002298s |  2.7365 |
| CorrDist | 0.010240s |  0.010571s |  0.9687 |
| ChiSqDist | 0.005468s |  0.001278s |  4.2785 |
| KLDivergence | 0.030543s |  0.024907s |  1.2263 |
| RenyiDivergence | 0.016365s |  0.009929s |  1.6482 |
| RenyiDivergence | 0.049968s |  0.044778s |  1.1159 |
| JSDivergence | 0.045897s |  0.042678s |  1.0754 |
| BhattacharyyaDist | 0.007597s |  0.003793s |  2.0029 |
| HellingerDist | 0.008023s |  0.003677s |  2.1818 |
| WeightedSqEuclidean | 0.004435s |  0.001180s |  3.7582 |
| WeightedEuclidean | 0.004489s |  0.001199s |  3.7448 |
| WeightedCityblock | 0.004447s |  0.001178s |  3.7752 |
| WeightedMinkowski | 0.047526s |  0.040818s |  1.1644 |
| WeightedHamming | 0.004438s |  0.001181s |  3.7568 |
| SqMahalanobis | 0.077402s |  0.020820s |  3.7177 |
| Mahalanobis | 0.078468s |  0.020794s |  3.7736 |
| BrayCurtis | 0.004579s |  0.001125s |  4.0703 |

We can see that using ``colwise`` instead of a simple loop yields considerable gain (2x - 4x), especially when the internal computation of each distance is simple. Nonetheless, when the computation of a single distance is heavy enough (e.g. *KLDivergence*,  *RenyiDivergence*), the gain is not as significant.

#### Pairwise benchmark

The table below compares the performance (measured in terms of average elapsed time of each iteration) of a straightforward loop implementation and an optimized implementation provided in *Distances.jl*. The task in each iteration is to compute a specific distance in a pairwise manner between columns in a ``100-by-200`` and ``100-by-250`` matrices, which will result in a ``200-by-250`` distance matrix.

|  distance  |  loop  |  pairwise  |  gain  |
|----------- | -------| ----------| -------|
| SqEuclidean | 0.010946s |  0.001245s |  8.7912 |
| Euclidean | 0.011314s |  0.001328s |  8.5189 |
| PeriodicEuclidean | 0.028662s |  0.018289s |  1.5672 |
| Cityblock | 0.011112s |  0.001841s |  6.0349 |
| TotalVariation | 0.011052s |  0.001906s |  5.7993 |
| Chebyshev | 0.022198s |  0.012849s |  1.7276 |
| Minkowski | 0.116707s |  0.104321s |  1.1187 |
| Hamming | 0.010888s |  0.001733s |  6.2837 |
| CosineDist | 0.015977s |  0.001278s | 12.5042 |
| CorrDist | 0.026070s |  0.001338s | 19.4795 |
| ChiSqDist | 0.013399s |  0.004151s |  3.2276 |
| KLDivergence | 0.075753s |  0.062430s |  1.2134 |
| RenyiDivergence | 0.041190s |  0.027761s |  1.4838 |
| RenyiDivergence | 0.127884s |  0.114440s |  1.1175 |
| JSDivergence | 0.114959s |  0.106575s |  1.0787 |
| BhattacharyyaDist | 0.019642s |  0.010409s |  1.8870 |
| HellingerDist | 0.019028s |  0.009799s |  1.9418 |
| WeightedSqEuclidean | 0.011327s |  0.001252s |  9.0478 |
| WeightedEuclidean | 0.011520s |  0.001338s |  8.6124 |
| WeightedCityblock | 0.011308s |  0.002538s |  4.4547 |
| WeightedMinkowski | 0.121861s |  0.105896s |  1.1508 |
| WeightedHamming | 0.011249s |  0.002448s |  4.5945 |
| SqMahalanobis | 0.189282s |  0.001452s | 130.3337 |
| Mahalanobis | 0.188376s |  0.001537s | 122.5507 |
| BrayCurtis | 0.011306s |  0.002471s |  4.5760 |

For distances of which a major part of the computation is a quadratic form (e.g. *Euclidean*, *CosineDist*, *Mahalanobis*), the performance can be drastically improved by restructuring the computation and delegating the core part to ``GEMM`` in *BLAS*. The use of this strategy can easily lead to 100x performance gain over simple loops (see the highlighted part of the table above).
