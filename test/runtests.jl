using Test, NLPModels, LinearAlgebra, LinearOperators, Printf, SparseArrays

problems = ["BROWNDEN", "HS5", "HS6", "HS10", "HS11", "HS14", "LINCON", "LINSV"]
nls_problems = ["LLS", "MGH01", "NLSHS20", "NLSLC"]

# Including problems so that they won't be multiply loaded
for problem in problems ∪ ["GENROSE"] # GENROSE does not have a manual version, so it's separate
  include("problems/$(lowercase(problem)).jl")
end
for problem in nls_problems
  include("nls_problems/$(lowercase(problem)).jl")
end

@info("Testing printing of nlp.meta")
print(ADNLPModel(x->0, zeros(10), [-ones(5); -Inf*ones(5)],
                 [ones(3); Inf*ones(4); collect(2:4)],
                 name="Unconstrained example").meta)
print(ADNLPModel(x->0, zeros(10), x->[0.0;0.0;0.0], [0.0;0.0;-Inf],
                 [Inf;0.0;0.0], name="Constrained example").meta)

# A problem with zero variables doesn't make sense.
@test_throws(ErrorException, NLPModelMeta(0))

# Default methods should throw MethodError since they're not defined
mutable struct DummyModel <: AbstractNLPModel
  meta :: NLPModelMeta
end
model = DummyModel(NLPModelMeta(1))
@test_throws(MethodError, lagscale(model, 1.0))
for meth in [:obj, :varscale, :conscale]
  @eval @test_throws(MethodError, $meth(model, [0.0]))
end
for meth in [:jac_structure!, :hess_structure!]
  @eval @test_throws(MethodError, $meth(model, [0], [1]))
end
for meth in [:grad!, :cons!, :jac_coord!]
  @eval @test_throws(MethodError, $meth(model, [0.0], [1.0]))
end
for meth in [:jth_con, :jth_congrad, :jth_sparse_congrad]
  @eval @test_throws(MethodError, $meth(model, [0.0], 1))
end
@test_throws(MethodError, jth_congrad!(model, [0.0], 1, [2.0]))
for meth in [:jprod!, :jtprod!]
  @eval @test_throws(MethodError, $meth(model, [0.0], [1.0], [2.0]))
end
@test_throws(MethodError, jth_hprod(model, [0.0], [1.0], 2))
@test_throws(MethodError, jth_hprod!(model, [0.0], [1.0], 2, [3.0]))
for meth in [:ghjvprod!]
  @eval @test_throws(MethodError, $meth(model, [0.0], [1.0], [2.0], [3.0]))
end
@assert isa(hess_op(model, [0.]), LinearOperator)
@assert isa(jac_op(model, [0.]), LinearOperator)

# ADNLPModel with no functions
model = ADNLPModel(x->dot(x,x), zeros(2), name="square")
@assert model.meta.name == "square"

model = genrose_autodiff()
for counter in fieldnames(typeof(model.counters))
  @eval @assert $counter(model) == 0
end

obj(model, model.meta.x0)
@assert neval_obj(model) == 1

reset!(model)
@assert neval_obj(model) == 0

@test_throws(MethodError, jth_con(model, model.meta.x0, 1))

include("test_tools.jl")

include("test_slack_model.jl")
include("test_qn_model.jl")

@info("For tests to pass, all models must have been written identically.\n")
@info("Constraints, if any, must have been declared in the same order.\n")

include("multiple-precision.jl")
include("check-dimensions.jl")
include("consistency.jl")
for problem in problems
  @info "Checking consistency of problem $problem"
  nlp_ad = eval(Meta.parse(lowercase(problem) * "_autodiff"))()
  nlp_man = eval(Meta.parse(problem))()

  nlps = [nlp_ad, nlp_man]
  for nlp in nlps
    show(nlp)
  end

  consistent_nlps(nlps)
  @info "  Consistency checks ✓"

  for nlp in nlps ∪ SlackModel.(nlps)
    @info "  Checking that wrong input dimensions throw errors in $(nlp.meta.name)"
    check_nlp_dimensions(nlp)
  end

  for nlp in nlps ∪ SlackModel.(nlps)
    @info "  Checking multiple precision support by $(nlp.meta.name)"
    multiple_precision(nlp)
  end
end

include("test_autodiff_model.jl")
include("test_nlsmodels.jl")
include("nls_consistency.jl")
for problem in nls_problems
  @info "Checking consistency of NLS problem $problem"
  nls_ad = eval(Meta.parse(lowercase(problem) * "_autodiff"))()
  nls_man = eval(Meta.parse(problem))()

  nlss = [nls_ad, nls_man]
  spc = lowercase(problem) * "_special"
  if isdefined(Main, Symbol(spc))
    push!(nlss, eval(Meta.parse(spc))())
  end
  for nls in nlss
    show(nls)
  end

  consistent_nlss(nlss)
  @info "  Consistency checks ✓"

  for nls in nlss ∪ SlackNLSModel.(nlss) ∪ FeasibilityFormNLS.(nlss)
    @info "  Checking that wrong input dimensions throw errors in $(nls.meta.name)"
    check_nls_dimensions(nls)
    check_nlp_dimensions(nls, exclude_hess=true)
  end

  # AbstractLLSModels returns the internal A for jac, hence it doesn't respect type input
  idx = findall([typeof(nls) <: AbstractLLSModel for nls in nlss])
  if length(idx) > 0
    deleteat!(nlss, idx)
  end

  for nls in nlss ∪ SlackNLSModel.(nlss) ∪ FeasibilityFormNLS.(nlss)
    @info "  Checking multiple precision support by $(nls.meta.name)"
    multiple_precision(nls)
  end
end
include("test_feasibility_form_nls.jl")
include("test_view_subarray.jl")
test_view_subarrays()
include("test_memory_of_coord.jl")
test_memory_of_coord()