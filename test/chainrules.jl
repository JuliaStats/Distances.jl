using ChainRulesCore
using ChainRulesTestUtils

@testset "ChainRulesCore extension" begin
    n = 4
    x = randn(n)
    y = randn(n)
    X = randn(n, 3)
    Y = randn(n, 3)

    @testset for metric in (SqEuclidean(), Euclidean())
        @testset "different arguments" begin
            # Single evaluation
            test_rrule(metric ⊢ NoTangent(), x, y)

            # Column-wise distance
            test_rrule(colwise, metric ⊢ NoTangent(), X, Y)

            # Pairwise distances
            test_rrule(pairwise, metric ⊢ NoTangent(), X)
            test_rrule(pairwise, metric ⊢ NoTangent(), X; fkwargs=(dims=1,))
            test_rrule(pairwise, metric ⊢ NoTangent(), X; fkwargs=(dims=2,))
            test_rrule(pairwise, metric ⊢ NoTangent(), X, Y)
            test_rrule(pairwise, metric ⊢ NoTangent(), X, Y; fkwargs=(dims=1,))
            test_rrule(pairwise, metric ⊢ NoTangent(), X, Y; fkwargs=(dims=2,))    
        end

        # check numerical issues if distances are zero
        @testset "equal arguments" begin
            # Single evaluation
            test_rrule(metric ⊢ NoTangent(), x, x)

            # Column-wise distance
            test_rrule(colwise, metric ⊢ NoTangent(), X, X)

            # Pairwise distances
            # Finite differencing yields impressively inaccurate derivatives for `Euclidean`,
            # see https://github.com/FluxML/Zygote.jl/blob/45bf883491d2b52580d716d577e2fa8577a07230/test/gradcheck.jl#L1206
            kwargs = metric isa Euclidean ? (rtol = 1e-5,) : ()
            test_rrule(pairwise, metric ⊢ NoTangent(), X, X; kwargs...)
            test_rrule(pairwise, metric ⊢ NoTangent(), X, X; fkwargs=(dims=1,), kwargs...)
            test_rrule(pairwise, metric ⊢ NoTangent(), X, X; fkwargs=(dims=2,), kwargs...)
        end
    end
end
