using BenchmarkTools
using Distances
using Printf

include("benchmarks.jl")

# BenchmarkTools stores things in a Dict so it loses ordering but we want to print the table
# in a special order. Therefore define an order here:

order = [
    :SqEuclidean,
    :Euclidean,
    :PeriodicEuclidean,
    :Cityblock,
    :TotalVariation,
    :Chebyshev,
    :Minkowski,
    :Hamming,
    :CosineDist,
    :CorrDist,
    :ChiSqDist,
    :KLDivergence,
    :RenyiDivergence,
    :RenyiDivergence,
    :RenyiDivergence,
    :RenyiDivergence,
    :JSDivergence,
    :BhattacharyyaDist,
    :HellingerDist,
    :WeightedSqEuclidean,
    :WeightedEuclidean,
    :WeightedCityblock,
    :WeightedMinkowski,
    :WeightedHamming,
    :SqMahalanobis,
    :Mahalanobis,
    :Haversine,
    :BrayCurtis,
]

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 2.0 # Long enough

# Tuning
if !isfile(@__DIR__, "params.json")
    tuning = tune!(SUITE; verbose = true);
    BenchmarkTools.save("params.json", "SUITE", params(SUITE))
end
loadparams!(SUITE, BenchmarkTools.load("params.json")[1], :evals, :samples);

# Run and judge
results = run(SUITE; verbose = true)
judgement = minimum(results)

# Output the comparison table
getname(T::DataType) = T.name.name

function print_table(judgement)
    for typ in ("colwise", "pairwise")
        io = IOBuffer()
        println(io, "|  distance  |  loop  |  $typ  |  gain  |")
        println(io, "|----------- | -------| ----------| -------|")
        sorted_distances = sort(collect(judgement[typ]), by = y -> findfirst(x -> x == getname(y[1]), order))

        for (dist, result) in sorted_distances
            t_loop = BenchmarkTools.time(result["loop"])
            t_spec = BenchmarkTools.time(result["specialized"])
            print(io, "| ", getname(dist), " |")
            print(io, @sprintf("%9.6fs | %9.6fs | %7.4f |\n", t_loop / 1e9, t_spec / 1e9, (t_loop / t_spec)))
        end
        print(stdout, String(take!(io)))
        println()
    end
end

print_table(judgement)
