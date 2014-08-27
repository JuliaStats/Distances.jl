# Distances.jl


[![Build Status](https://travis-ci.org/JuliaStats/Distances.jl.svg?branch=master)](https://travis-ci.org/JuliaStats/Distances.jl)

A Julia package for evaluating distances(metrics) between vectors.

This package also provides optimized functions to compute column-wise and pairwise distances, which are often substantially faster than a straightforward loop implementation. (See the benchmark section below for details).

## Supported distances

* Euclidean distance
* Squared Euclidean distance
* Cityblock distance 
* Chebyshev distance
* Minkowski distance
* Hamming distance
* Cosine distance
* Correlation distance
* Chi-square distance
* Kullback-Leibler divergence
* Jensen-Shannon divergence
* Mahalanobis distance
* Squared Mahalanobis distance
* Bhattacharyya distance
* Hellinger distance
* Jensen-Shannon metric

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

| type name            |  convenient syntax   | math definition     | 
| -------------------- | -------------------- | --------------------|
|  Euclidean           |  euclidean(x, y)     | sqrt(sum((x - y) .^ 2)) |
|  SqEuclidean         |  sqeuclidean(x, y)   | sum((x - y).^2) |
|  Cityblock           |  cityblock(x, y)     | sum(abs(x - y)) |
|  Chebyshev           |  chebyshev(x, y)     | max(abs(x - y)) |
|  Minkowski           |  minkowski(x, y, p)  | sum(abs(x - y).^p) ^ (1/p) |
|  Hamming             |  hamming(x, y)       | sum(x .!= y) |
|  CosineDist          |  cosine_dist(x, y)   | 1 - dot(x, y) / (norm(x) * norm(y)) |
|  CorrDist            |  corr_dist(x, y)     | cosine_dist(x - mean(x), y - mean(y)) |
|  ChiSqDist           |  chisq_dist(x, y)    | sum((x - y).^2 / (x + y)) | 
|  KLDivergence        |  kl_divergence(x, y) | sum(p .* log(p ./ q)) |
|  JSDivergence        |  js_divergence(x, y) | KL(x, m) / 2 + KL(y, m) / 2 with m = (x + y) / 2 |
|  SpanNormDist        |  spannorm_dist(x, y) | max(x - y) - min(x - y ) |
|  BhattacharyyaDist   |  bhattacharyya(x, y) | -log(sum(sqrt(x .* y) / sqrt(sum(x) * sum(y))) |
|  HellingerDist       |  hellinger(x, y)     | sqrt(1 - sum(sqrt(x .* y) / sqrt(sum(x) * sum(y)))) |
|  Mahalanobis         |  mahalanobis(x, y, Q)    | sqrt((x - y)' * Q * (x - y)) |
|  SqMahalanobis       |  sqmahalanobis(x, y, Q)  |  (x - y)' * Q * (x - y)  |
|  WeightedEuclidean   |  euclidean(x, y, w)      | sqrt(sum((x - y).^2 .* w))  |
|  WeightedSqEuclidean |  sqeuclidean(x, y, w)    | sum((x - y).^2 .* w)  |
|  WeightedCityblock   |  cityblock(x, y, w)      | sum(abs(x - y) .* w)  |
|  WeightedMinkowski   |  minkowski(x, y, w, p)   | sum(abs(x - y).^p .* w) ^ (1/p)  |
|  WeightedHamming     |  hamming(x, y, w)        | sum((x .!= y) .* w)  |
|  JSMetric            |  js_metric(X)        | entropy(X ./ m) - sum(colwise(entropy,X)) / m|
  
**Note:** The formulas above are using *Julia*'s functions. These formulas are mainly for conveying the math concepts in a concise way. The actual implementation may use a faster way.


## Benchmarks


The implementation has been carefully optimized based on benchmarks. The Julia scripts ``test/bench_colwise.jl`` and ``test/bench_pairwise.jl`` run the benchmarks on a variety of distances, respectively under column-wise and pairwise settings.

Here are the benchmarks that I obtained on Mac OS X 10.8 with Intel Core i7 2.6 GHz.

#### Column-wise benchmark

The table below compares the performance (measured in terms of average elapsed time of each iteration) of a straightforward loop implementation and an optimized implementation provided in *Distances.jl*. The task in each iteration is to compute a specific distance between corresponding columns in two ``200-by-10000`` matrices.

|  distance   |   loop  |   colwise   |   gain     |
|------------ | --------| ------------| -----------|
| SqEuclidean | 0.046962 | 0.002782 | 16.8782 |
| Euclidean | 0.046667 | 0.0029 | 16.0937 |
| Cityblock | 0.046619 | 0.0031 | 15.039 |
| Chebyshev | 0.053578 | 0.010856 | 4.9356 |
| Minkowski | 0.061804 | 0.02357 | 2.6221 |
| Hamming | 0.044047 | 0.00219 | 20.1131 |
| CosineDist | 0.04496 | 0.002855 | 15.7457 |
| CorrDist | 0.080828 | 0.029708 | 2.7207 |
| ChiSqDist | 0.051009 | 0.008088 | 6.307 |
| KLDivergence | 0.079598 | 0.035353 | 2.2515 |
| JSDivergence | 0.545789 | 0.493362 | 1.1063 |
| WeightedSqEuclidean | 0.046182 | 0.003219 | 14.3477 |
| WeightedEuclidean | 0.046831 | 0.004122 | 11.3603 |
| WeightedCityblock | 0.046457 | 0.003636 | 12.7781 |
| WeightedMinkowski | 0.062532 | 0.020486 | 3.0524 |
| WeightedHamming | 0.046217 | 0.002269 | 20.3667 |
| SqMahalanobis | 0.150364 | 0.042335 | 3.5518 |
| Mahalanobis | 0.159638 | 0.041071 | 3.8869 |

We can see that using ``colwise`` instead of a simple loop yields considerable gain (2x - 9x), especially when the internal computation of each distance is simple. Nonetheless, when the computaton of a single distance is heavy enough (e.g. *Minkowski* and *JSDivergence*), the gain is not as significant.

#### Pairwise benchmark

The table below compares the performance (measured in terms of average elapsed time of each iteration) of a straightforward loop implementation and an optimized implementation provided in *Distances.jl*. The task in each iteration is to compute a specific distance in a pairwise manner between columns in a ``100-by-200`` and ``100-by-250`` matrices, which will result in a ``200-by-250`` distance matrix.

|  distance   |   loop  |   pairwise  |   gain     |
|------------ | --------| ------------| -----------|
| SqEuclidean | 0.119961 | 0.00037 | **324.6457** |
| Euclidean | 0.122645 | 0.000678 | **180.9180** |
| Cityblock | 0.116956 | 0.007997 | 14.6251 |
| Chebyshev | 0.137985 | 0.028489 | 4.8434 |
| Minkowski | 0.170101 | 0.059991 | 2.8354 |
| Hamming | 0.110742 | 0.004781 | 23.1627 |
| CosineDist | 0.110913 | 0.000514 | **215.8028** |
| CorrDist | 0.1992 | 0.000808 | 246.4574 |
| ChiSqDist | 0.124782 | 0.020781 | 6.0046 |
| KLDivergence | 0.1994 | 0.088366 | 2.2565 |
| JSDivergence | 1.35502 | 1.215785 | 1.1145 |
| WeightedSqEuclidean | 0.119797 | 0.000444 | **269.531** |
| WeightedEuclidean | 0.126304 | 0.000712 | **177.5122** |
| WeightedCityblock | 0.117185 | 0.011475 | 10.2122 |
| WeightedMinkowski | 0.172614 | 0.061693 | 2.7979 |
| WeightedHamming | 0.112525 | 0.005072 | 22.1871 |
| SqMahalanobis | 0.377342 | 0.000577 | **653.9759** |
| Mahalanobis | 0.373796 | 0.002359 | **158.4337** |

For distances of which a major part of the computation is a quadratic form (e.g. *Euclidean*, *CosineDist*, *Mahalanobis*), the performance can be drastically improved by restructuring the computation and delegating the core part to ``GEMM`` in *BLAS*. The use of this strategy can easily lead to 100x performance gain over simple loops (see the highlighted part of the table above).


