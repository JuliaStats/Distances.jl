# Distances.jl

[![Build Status](https://travis-ci.org/JuliaStats/Distances.jl.svg?branch=master)](https://travis-ci.org/JuliaStats/Distances.jl)
[![Coverage Status](https://coveralls.io/repos/JuliaStats/Distances.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/JuliaStats/Distances.jl?branch=master)

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

For `Euclidean distance`, `Squared Euclidean distance`, `Cityblock distance`, `Minkowski distance`, and `Hamming distance`, a weighted version is also provided.


## Basic use

The library supports three ways of computation: *computing the distance between two vectors*, *column-wise computation*, and *pairwise computation*.

#### Computing the distance between two vectors

Each distance corresponds to a *distance type*. You can always compute a certain distance between two vectors using the following syntax

```julia
r = evaluate(dist, x, y)
r = dist(x, y)
```

Here, `dist` is an instance of a distance type. For example, the type for Euclidean distance is `Euclidean` (more distance types will be introduced in the next section), then you can compute the Euclidean distance between `x` and `y` as

```julia
r = evaluate(Euclidean(), x, y)
r = Euclidean()(x, y)
```

Common distances also come with convenient functions for distance evaluation. For example, you may also compute Euclidean distance between two vectors as below

```julia
r = euclidean(x, y)
```

#### Computing distances between corresponding columns

Suppose you have two `m-by-n` matrix `X` and `Y`, then you can compute all distances between corresponding columns of `X` and `Y` in one batch, using the `colwise` function, as

```julia
r = colwise(dist, X, Y)
```

The output `r` is a vector of length `n`. In particular, `r[i]` is the distance between `X[:,i]` and `Y[:,i]`. The batch computation typically runs considerably faster than calling `evaluate` column-by-column.

Note that either of `X` and `Y` can be just a single vector -- then the `colwise` function will compute the distance between this vector and each column of the other parameter.

#### Computing pairwise distances

Let `X` and `Y` respectively have `m` and `n` columns. Then the `pairwise` function with the `dims=2` argument computes distances between each pair of columns in `X` and `Y`:

```julia
R = pairwise(dist, X, Y, dims=2)
```

In the output, `R` is a matrix of size `(m, n)`, such that `R[i,j]` is the distance between `X[:,i]` and `Y[:,j]`. Computing distances for all pairs using `pairwise` function is often remarkably faster than evaluting for each pair individually.

If you just want to just compute distances between columns of a matrix `X`, you can write

```julia
R = pairwise(dist, X, dims=2)
```

This statement will result in an `m-by-m` matrix, where `R[i,j]` is the distance between `X[:,i]` and `X[:,j]`.
`pairwise(dist, X)` is typically more efficient than `pairwise(dist, X, X)`, as the former will take advantage of the symmetry when `dist` is a semi-metric (including metric).

For performance reasons, it is recommended to use matrices with observations in columns (as shown above). Indeed,
the `Array` type in Julia is column-major, making it more efficient to access memory column by column. However,
matrices with observations stored in rows are also supported via the argument `dims=1`.

#### Computing column-wise and pairwise distances inplace

If the vector/matrix to store the results are pre-allocated, you may use the storage (without creating a new array) using the following syntax (`i` being either `1` or `2`):

```julia
colwise!(r, dist, X, Y)
pairwise!(R, dist, X, Y, dims=i)
pairwise!(R, dist, X, dims=i)
```

Please pay attention to the difference, the functions for inplace computation are `colwise!` and `pairwise!` (instead of `colwise` and `pairwise`).


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

The table below compares the performance (measured in terms of average elapsed time of each iteration) of a straightforward loop implementation and an optimized implementation provided in *Distances.jl*. The task in each iteration is to compute a specific distance between corresponding columns in two `200-by-10000` matrices.

|  distance  |  loop  |  colwise  |  gain  |
|----------- | -------| ----------| -------|
| SqEuclidean | 0.004432s |  0.001049s |  4.2270 |
| Euclidean | 0.004537s |  0.001054s |  4.3031 |
| PeriodicEuclidean | 0.012092s |  0.006714s |  1.8011 |
| Cityblock | 0.004515s |  0.001060s |  4.2585 |
| TotalVariation | 0.004496s |  0.001062s |  4.2337 |
| Chebyshev | 0.009123s |  0.005034s |  1.8123 |
| Minkowski | 0.047573s |  0.042508s |  1.1191 |
| Hamming | 0.004355s |  0.001099s |  3.9638 |
| CosineDist | 0.006432s |  0.002282s |  2.8185 |
| CorrDist | 0.010273s |  0.012500s |  0.8219 |
| ChiSqDist | 0.005291s |  0.001271s |  4.1635 |
| KLDivergence | 0.031491s |  0.025643s |  1.2281 |
| RenyiDivergence | 0.052420s |  0.048075s |  1.0904 |
| RenyiDivergence | 0.017317s |  0.009023s |  1.9193 |
| JSDivergence | 0.047905s |  0.044006s |  1.0886 |
| BhattacharyyaDist | 0.007761s |  0.003796s |  2.0445 |
| HellingerDist | 0.007636s |  0.003665s |  2.0836 |
| WeightedSqEuclidean | 0.004550s |  0.001151s |  3.9541 |
| WeightedEuclidean | 0.004687s |  0.001168s |  4.0125 |
| WeightedCityblock | 0.004493s |  0.001157s |  3.8849 |
| WeightedMinkowski | 0.049442s |  0.042145s |  1.1732 |
| WeightedHamming | 0.004431s |  0.001153s |  3.8440 |
| SqMahalanobis | 0.082493s |  0.019843s |  4.1574 |
| Mahalanobis | 0.082180s |  0.019618s |  4.1891 |
| BrayCurtis | 0.004464s |  0.001121s |  3.9809 |

We can see that using `colwise` instead of a simple loop yields considerable gain (2x - 4x), especially when the internal computation of each distance is simple. Nonetheless, when the computation of a single distance is heavy enough (e.g. *KLDivergence*,  *RenyiDivergence*), the gain is not as significant.

#### Pairwise benchmark

The table below compares the performance (measured in terms of average elapsed time of each iteration) of a straightforward loop implementation and an optimized implementation provided in *Distances.jl*. The task in each iteration is to compute a specific distance in a pairwise manner between columns in a `100-by-200` and `100-by-250` matrices, which will result in a `200-by-250` distance matrix.

|  distance  |  loop  |  pairwise  |  gain  |
|----------- | -------| ----------| -------|
| SqEuclidean | 0.012498s |  0.000170s | **73.6596** |
| Euclidean | 0.012583s |  0.000257s | 48.9628 |
| PeriodicEuclidean | 0.030935s |  0.017572s |  1.7605 |
| Cityblock | 0.012416s |  0.000910s | 13.6464 |
| TotalVariation | 0.012763s |  0.000959s | 13.3080 |
| Chebyshev | 0.023800s |  0.012042s |  1.9763 |
| Minkowski | 0.121388s |  0.107333s |  1.1310 |
| Hamming | 0.012171s |  0.000689s | 17.6538 |
| CosineDist | 0.017474s |  0.000214s | **81.6546** |
| CorrDist | 0.028195s |  0.000259s | **108.7360** |
| ChiSqDist | 0.014372s |  0.003129s |  4.5932 |
| KLDivergence | 0.079669s |  0.063491s |  1.2548 |
| RenyiDivergence | 0.134093s |  0.117737s |  1.1389 |
| RenyiDivergence | 0.047658s |  0.024960s |  1.9094 |
| JSDivergence | 0.121999s |  0.110984s |  1.0993 |
| BhattacharyyaDist | 0.021788s |  0.009414s |  2.3145 |
| HellingerDist | 0.020735s |  0.008784s |  2.3606 |
| WeightedSqEuclidean | 0.012671s |  0.000186s | **68.0345** |
| WeightedEuclidean | 0.012867s |  0.000276s | **46.6634** |
| WeightedCityblock | 0.012803s |  0.001539s |  8.3200 |
| WeightedMinkowski | 0.127386s |  0.107257s |  1.1877 |
| WeightedHamming | 0.012240s |  0.001462s |  8.3747 |
| SqMahalanobis | 0.214285s |  0.000330s | **650.0722** |
| Mahalanobis | 0.197294s |  0.000420s | **470.2354** |
| BrayCurtis | 0.012872s |  0.001489s |  8.6456 |

For distances of which a major part of the computation is a quadratic form (e.g. *Euclidean*, *CosineDist*, *Mahalanobis*), the performance can be drastically improved by restructuring the computation and delegating the core part to `GEMM` in *BLAS*. The use of this strategy can easily lead to 100x performance gain over simple loops (see the highlighted part of the table above).
