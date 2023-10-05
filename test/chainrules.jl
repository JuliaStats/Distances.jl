using ChainRulesCore
using ChainRulesTestUtils
using StableRNGs

@testset "ChainRulesCore extension" begin
    n = 4
    rng = StableRNG(100)
    x = randn(rng, n)
    y = randn(rng, n)
    X = randn(rng, n, 3)
    Y = randn(rng, n, 3)
    Xrep = repeat(x, 1, 3)
    Yrep = repeat(y, 1, 3)

    @testset for metric in (SqEuclidean(), Euclidean())
        # Single evaluation
        test_rrule(metric ⊢ NoTangent(), x, y)
        test_rrule(metric ⊢ NoTangent(), x, x)

        for A in (X, Xrep)
            # Column-wise distance
            test_rrule(colwise, metric ⊢ NoTangent(), A, A)

            # Pairwise distances
            # Finite differencing yields impressively inaccurate derivatives for `Euclidean`,
            # see https://github.com/FluxML/Zygote.jl/blob/45bf883491d2b52580d716d577e2fa8577a07230/test/gradcheck.jl#L1206
            kwargs = metric isa Euclidean ? (rtol=1e-3, atol=1e-3) : ()
            test_rrule(pairwise, metric ⊢ NoTangent(), A; kwargs...)
            test_rrule(pairwise, metric ⊢ NoTangent(), A; fkwargs=(dims=1,), kwargs...)
            test_rrule(pairwise, metric ⊢ NoTangent(), A; fkwargs=(dims=2,), kwargs...)
            test_rrule(pairwise, metric ⊢ NoTangent(), A, A; kwargs...)
            test_rrule(pairwise, metric ⊢ NoTangent(), A, A; fkwargs=(dims=1,), kwargs...)
            test_rrule(pairwise, metric ⊢ NoTangent(), A, A; fkwargs=(dims=2,), kwargs...)

            for B in (Y, Yrep)
                # Column-wise distance
                test_rrule(colwise, metric ⊢ NoTangent(), A, B)

                # Pairwise distances
                test_rrule(pairwise, metric ⊢ NoTangent(), A, B)
                test_rrule(pairwise, metric ⊢ NoTangent(), A, B; fkwargs=(dims=1,))
                test_rrule(pairwise, metric ⊢ NoTangent(), A, B; fkwargs=(dims=2,))
            end
        end
    end
end
