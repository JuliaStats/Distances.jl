# Wasserstein distance

using JuMP: Model, AffExpr, with_optimizer, @variable, @constraint, 
            @objective, add_to_expression!, optimize!, termination_status,
            objective_value
import JuMP
import Cbc

struct Wasserstein <: Metric
    p::Float64

    function Wasserstein(p::Float64)
        @assert p >= 1
        new(p)
    end
end

Wasserstein() = Wasserstein(1.0)

function (dist::Wasserstein)(a::AbstractArray{T}, b::AbstractArray{T}) where {T}
    @assert length(a) == length(b)

    isempty(a) && return zero(T)

    @assert isapprox(sum(a), 1.0, atol=1e-6) "sum(a) needs to be ~1 but is $(sum(a))"
    @assert isapprox(sum(b), 1.0, atol=1e-6) "sum(b) needs to be ~1 but is $(sum(b))"

    model = make_wasserstein_model(a, b, dist.p)
    optimize!(model)
    @assert termination_status(model) == JuMP.MOI.OPTIMAL

    objective_value(model)^(1/dist.p)
end

"""
Create JuMP `Model` for linear program to calculate the p-Wasserstein distance 
of two discrete vectors from the same probability simplex. See also formula 
(2.5) in [Optimal Transport on Discrete Domains](https://arxiv.org/abs/1801.07745).
"""
function make_wasserstein_model(a::AbstractArray, b::AbstractArray, p::Float64) :: Model
    model = Model(with_optimizer(Cbc.Optimizer, logLevel=0))

    N = length(a)
    T = @variable(model, T[1:N, 1:N] >= 0)

    for i in 1:N
        row_expression = AffExpr()
        for j in 1:N
            add_to_expression!(row_expression, 1.0, T[i, j])
        end
        @constraint(model, row_expression == a[i])
    end

    for j in 1:N
        column_expression = AffExpr()
        for i in 1:N
            add_to_expression!(column_expression, 1.0, T[i, j])
        end
        @constraint(model, column_expression == b[j])
    end

    objective_expression = AffExpr()
    for i in 1:N
        for j in 1:N
            add_to_expression!(objective_expression, abs(i - j)^p, T[i, j])
        end
    end
    @objective(model, Min, objective_expression)

    model
end

wasserstein(a::AbstractArray, b::AbstractArray, p::Float64=1.0) = Wasserstein(p)(a, b)