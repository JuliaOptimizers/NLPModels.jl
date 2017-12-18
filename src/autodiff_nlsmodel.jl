using ForwardDiff

export ADNLSModel,
       residual!, jac_residual, jprod_residual!, jtprod_residual!,
       jac_op_residual, hess_residual, hprod_residual!, cons, cons!, jac_coord,
       jac, jprod, jprod!, jtprod, jtprod!, hess, hess_coord, hprod, hprod!

"""ADNLSModel is an Nonlinear Least Squares model using ForwardDiff to
compute the derivatives.

````
ADNLSModel(F, x0, m; lvar = [-∞,…,-∞], uvar = [∞,…,∞], y0 = zeros,
  c = NotImplemented, lcon = [-∞,…,-∞], ucon = [∞,…,∞], name = "Generic")
````

  - `F :: Function` - The residual function \$F\$;
  - `x0 :: Vector` - The initial point of the problem;
  - `m :: Int` - The dimension of \$F(x)\$, i.e., the number of
  equations in the nonlinear system.

The other parameters are as in `ADNLPModel`.
"""
type ADNLSModel <: AbstractNLSModel
  meta :: NLPModelMeta
  nls_meta :: NLSMeta
  counters :: NLSCounters

  # Function
  F :: Function
  c :: Function
end

function ADNLSModel(F :: Function, x0 :: Vector, m :: Int;
                    name :: String = "Generic",
                    lvar :: Vector = fill(-Inf, length(x0)),
                    uvar :: Vector = fill(Inf, length(x0)),
                    c :: Function = (args...)->throw(NotImplementedError("cons")),
                    lcon :: Vector = Float64[],
                    ucon :: Vector = Float64[],
                    y0 :: Vector = zeros(max(length(lcon), length(ucon)))
                   )
  nvar = length(x0)
  ncon = maximum([length(lcon); length(ucon); length(y0)])
  if !(length(lcon) == length(ucon) == length(y0))
    error("lcon, ucon and y0 need to be the same length")
  end
  nnzj = nvar * ncon

  meta = NLPModelMeta(nvar, x0=x0, lvar=lvar, uvar=uvar, ncon=ncon, y0=y0,
                      lcon=lcon, ucon=ucon, nnzj=nnzj)
  nls_meta = NLSMeta(m, nvar)

  return ADNLSModel(meta, nls_meta, NLSCounters(), F, c)
end

ADNLSModel(F :: Function, n :: Int, m :: Int; kwargs...) = ADNLSModel(F, zeros(n), m; kwargs...)

function residual!(nls :: ADNLSModel, x :: AbstractVector, Fx :: AbstractVector)
  increment!(nls, :neval_residual)
  Fx[:] = nls.F(x)
  return Fx
end

function jac_residual(nls :: ADNLSModel, x :: Vector)
  increment!(nls, :neval_jac_residual)
  return ForwardDiff.jacobian(nls.F, x)
end

function jprod_residual!(nls :: ADNLSModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nls, :neval_jprod_residual)
  Jv[:] = ForwardDiff.jacobian(nls.F, x) * v
  return Jv
end

function jtprod_residual!(nls :: ADNLSModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nls, :neval_jtprod_residual)
  Jtv[:] = ForwardDiff.jacobian(nls.F, x)' * v
  return Jtv
end

function hess_residual(nls :: ADNLSModel, x :: AbstractVector, i :: Int)
  increment!(nls, :neval_hess_residual)
  return tril(ForwardDiff.hessian(x->nls.F(x)[i], x))
end

function hprod_residual!(nls :: ADNLSModel, x :: AbstractVector, i :: Int, v :: AbstractVector, Hiv :: AbstractVector)
  increment!(nls, :neval_hprod_residual)
  Hiv[:] = ForwardDiff.hessian(x->nls.F(x)[i], x) * v
  return Hiv
end

function cons(nls :: ADNLSModel, x :: Vector)
  increment!(nls, :neval_cons)
  return nls.c(x)
end

function cons!(nls :: ADNLSModel, x :: Vector, c :: Vector)
  increment!(nls, :neval_cons)
  c[1:nls.meta.ncon] = nls.c(x)
  return c
end

function jac_coord(nls :: ADNLSModel, x :: Vector)
  increment!(nls, :neval_jac)
  J = ForwardDiff.jacobian(nls.c, x)
  return findnz(J)
end

function jac(nls :: ADNLSModel, x :: Vector)
  increment!(nls, :neval_jac)
  return ForwardDiff.jacobian(nls.c, x)
end

function jprod(nls :: ADNLSModel, x :: Vector, v :: Vector)
  increment!(nls, :neval_jprod)
  return ForwardDiff.jacobian(nls.c, x) * v
end

function jprod!(nls :: ADNLSModel, x :: Vector, v :: Vector, Jv :: Vector)
  increment!(nls, :neval_jprod)
  Jv[1:nls.meta.ncon] = ForwardDiff.jacobian(nls.c, x) * v
  return Jv
end

function jtprod(nls :: ADNLSModel, x :: Vector, v :: Vector)
  increment!(nls, :neval_jtprod)
  return ForwardDiff.jacobian(nls.c, x)' * v
end

function jtprod!(nls :: ADNLSModel, x :: Vector, v :: Vector, Jtv :: Vector)
  increment!(nls, :neval_jtprod)
  Jtv[1:nls.meta.nvar] = ForwardDiff.jacobian(nls.c, x)' * v
  return Jtv
end

function hess(nls :: ADNLSModel, x :: Vector; obj_weight = 1.0, y :: Vector = [])
  increment!(nls, :neval_hess)
  Fx = residual(nls, x)
  Jx = jac_residual(nls, x)
  Hx = obj_weight == 0.0 ? spzeros(nls.meta.nvar, nls.meta.nvar) : Jx' * Jx * obj_weight
  if obj_weight != 0.0
    m = length(Fx)
    for i = 1:m
      Hx += obj_weight * Fx[i] * hess_residual(nls, x, i)
    end
  end
  for i = 1:min(length(y), nls.meta.ncon)
    if y[i] != 0.0
      Hx += ForwardDiff.hessian(x->nls.c(x)[i], x) * y[i]
    end
  end
  return tril(Hx)
end

function hess_coord(nls :: ADNLSModel, x :: Vector; obj_weight = 1.0, y :: Vector = [])
  H = hess(nls, x, obj_weight=obj_weight, y=y)
  return findnz(H)
end

function hprod(nls :: ADNLSModel, x :: Vector, v :: Vector;
    obj_weight = 1.0, y :: Vector = [])
  Hv = zeros(nls.meta.nvar)
  return hprod!(nls, x, v, Hv, obj_weight=obj_weight, y=y)
end

function hprod!(nls :: ADNLSModel, x :: Vector, v :: Vector, Hv :: Vector;
    obj_weight = 1.0, y :: Vector = [])
  increment!(nls, :neval_hprod)
  n = nls.meta.nvar
  if obj_weight != 0.0
    Fx = residual(nls, x)
    Jv = jprod_residual(nls, x, v)
    @views jtprod_residual!(nls, x, Jv, Hv[1:n])
    m = length(Fx)
    Hiv = zeros(n)
    for i = 1:m
      hprod_residual!(nls, x, i, v, Hiv)
      @views Hv[1:n] .= Hv[1:n] .+ Fx[i] * Hiv
    end
    Hv[1:n] .*= obj_weight
  end
  for i = 1:min(length(y), nls.meta.ncon)
    if y[i] != 0.0
      Hv[1:n] += ForwardDiff.hessian(x->nls.c(x)[i], x) * v * y[i]
    end
  end
  return Hv
end
