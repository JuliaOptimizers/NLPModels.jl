export FeasibilityFormNLS


"""Converts a nonlinear least-squares problem with residual ``F(x)`` to a nonlinear
optimization problem with constraints ``F(x) = r`` and objective ``\\tfrac{1}{2}\\|r\\|^2``.
In other words, converts
```math
\\begin{aligned}
       \\min_x \\quad & \\tfrac{1}{2}\\|F(x)\\|^2 \\\\
\\mathrm{s.t.} \\quad & c_L ≤ c(x) ≤ c_U \\\\
                      &   ℓ ≤   x  ≤ u
\\end{aligned}
```
to
```math
\\begin{aligned}
   \\min_{x,r} \\quad & \\tfrac{1}{2}\\|r\\|^2 \\\\
\\mathrm{s.t.} \\quad & F(x) - r = 0 \\\\
                      & c_L ≤ c(x) ≤ c_U \\\\
                      &   ℓ ≤   x  ≤ u
\\end{aligned}
```
If you rather have the first problem, the `nls` model already works as an NLPModel of
that format.
"""
mutable struct FeasibilityFormNLS{M <: AbstractNLSModel} <: AbstractNLSModel
  meta :: NLPModelMeta
  nls_meta :: NLSMeta
  internal :: M
  counters :: NLSCounters
end

show_header(io :: IO, nls :: FeasibilityFormNLS) = println(io, "FeasibilityFormNLS - Nonlinear least-squares moving the residual to constraints")

"""
    FeasibilityFormNLS(nls)

Converts a nonlinear least-squares problem with residual `F(x)` to a nonlinear
optimization problem with constraints `F(x) = r` and objective `¹/₂‖r‖²`.
"""
function FeasibilityFormNLS(nls :: AbstractNLSModel; name="$(nls.meta.name)-ffnls")
  nequ = nls.nls_meta.nequ
  meta = nls.meta
  nvar = meta.nvar + nequ
  ncon = meta.ncon + nequ
  nnzh = nls.nls_meta.nnzh + nequ + (meta.ncon == 0 ? 0 : meta.nnzh) # Some indexes can be repeated
  meta = NLPModelMeta(nvar, x0=[meta.x0; zeros(nequ)],
                      lvar=[meta.lvar; fill(-Inf, nequ)],
                      uvar=[meta.uvar; fill( Inf, nequ)],
                      ncon=ncon,
                      lcon=[zeros(nequ); meta.lcon],
                      ucon=[zeros(nequ); meta.ucon],
                      y0=[zeros(nequ); meta.y0],
                      lin=[nls.nls_meta.lin; meta.lin .+ nequ],
                      nln=[nls.nls_meta.nln; meta.nln .+ nequ],
                      nnzj=meta.nnzj + nls.nls_meta.nnzj + nequ,
                      nnzh=nnzh,
                      name=name
                     )
  nls_meta = NLSMeta(nequ, nvar, x0=[meta.x0; zeros(nequ)], nnzj=nequ, nnzh=0, lin=1:nequ, nln=Int[])

  nlp = FeasibilityFormNLS{typeof(nls)}(meta, nls_meta, nls, NLSCounters())
  finalizer(nlp -> finalize(nlp.internal), nlp)

  return nlp
end

function FeasibilityFormNLS(nls :: FeasibilityResidual; name="$(nls.meta.name)-ffnls")
  meta = nls.nlp.meta
  nequ = meta.ncon
  nvar = meta.nvar + nequ
  ncon = meta.ncon
  nnzj = meta.nnzj + nequ
  nnzh = meta.nnzh + nequ
  meta = NLPModelMeta(nvar, x0=[meta.x0; zeros(nequ)],
                      lvar=[meta.lvar; fill(-Inf, nequ)],
                      uvar=[meta.uvar; fill( Inf, nequ)],
                      ncon=ncon,
                      lcon=meta.lcon,
                      ucon=meta.ucon,
                      y0=meta.y0,
                      lin=meta.lin,
                      nln=meta.nln,
                      nnzj=nnzj, nnzh=nnzh,
                      name=name
                     )
  nls_meta = NLSMeta(nequ, nvar, x0=[meta.x0; zeros(nequ)], nnzj=nequ, nnzh=0)

  nlp = FeasibilityFormNLS{FeasibilityResidual}(meta, nls_meta, nls, NLSCounters())
  finalizer(nlp -> finalize(nlp.internal), nlp)

  return nlp
end

function FeasibilityFormNLS(nls :: LLS; name="$(nls.meta.name)-ffnls") where LLS <: AbstractLLSModel
  nequ = nls.nls_meta.nequ
  meta = nls.meta
  nvar = meta.nvar + nequ
  ncon = meta.ncon + nequ
  nnzh = nls.nls_meta.nnzh + nequ
  meta = NLPModelMeta(nvar, x0=[meta.x0; zeros(nequ)],
                      lvar=[meta.lvar; fill(-Inf, nequ)],
                      uvar=[meta.uvar; fill( Inf, nequ)],
                      ncon=ncon,
                      lcon=[zeros(nequ); meta.lcon],
                      ucon=[zeros(nequ); meta.ucon],
                      y0=[zeros(nequ); meta.y0],
                      lin=[nls.nls_meta.lin; meta.lin .+ nequ],
                      nln=[nls.nls_meta.nln; meta.nln .+ nequ],
                      nnzj=meta.nnzj + nls.nls_meta.nnzj + nequ,
                      nnzh=nnzh,
                      name=name
                     )
  nls_meta = NLSMeta(nequ, nvar, x0=[meta.x0; zeros(nequ)], nnzj=nequ, nnzh=0, lin=1:nequ, nln=Int[])

  nlp = FeasibilityFormNLS{typeof(nls)}(meta, nls_meta, nls, NLSCounters())
  finalizer(nlp -> finalize(nlp.internal), nlp)

  return nlp
end

function obj(nlp :: FeasibilityFormNLS, x :: AbstractVector)
  @lencheck nlp.meta.nvar x
  increment!(nlp, :neval_obj)
  n = nlp.internal.meta.nvar
  r = @view x[n+1:end]
  return dot(r, r) / 2
end

function grad!(nlp :: FeasibilityFormNLS, x :: AbstractVector, g :: AbstractVector)
  @lencheck nlp.meta.nvar x g
  increment!(nlp, :neval_grad)
  n = nlp.internal.meta.nvar
  g[1:n] .= 0.0
  g[n+1:end] .= @view x[n+1:end]
  return g
end

function objgrad!(nlp :: FeasibilityFormNLS, x :: Array{Float64}, g :: Array{Float64})
  @lencheck nlp.meta.nvar x g
  increment!(nlp, :neval_obj)
  increment!(nlp, :neval_grad)
  n = nlp.internal.meta.nvar
  r = @view x[n+1:end]
  f = dot(r, r) / 2
  g[1:n] .= 0.0
  g[n+1:end] .= @view x[n+1:end]
  return f, g
end

function cons!(nlp :: FeasibilityFormNLS, xr :: AbstractVector, c :: AbstractVector)
  @lencheck nlp.meta.nvar xr
  @lencheck nlp.meta.ncon c
  increment!(nlp, :neval_cons)
  n, m, ne = nlp.internal.meta.nvar, nlp.internal.meta.ncon, nlp.internal.nls_meta.nequ
  x = @view xr[1:n]
  r = @view xr[n+1:end]
  residual!(nlp.internal, x, @view c[1:ne])
  c[1:ne] .-= r
  if m > 0
    cons!(nlp.internal, x, @view c[ne+1:end])
  end
  return c
end

function jac_structure!(nlp :: FeasibilityFormNLS, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck nlp.meta.nnzj rows cols
  n, m, ne = nlp.internal.meta.nvar, nlp.internal.meta.ncon, nlp.internal.nls_meta.nequ
  nnzjF = nlp.internal.nls_meta.nnzj
  @views jac_structure_residual!(nlp.internal, rows[1:nnzjF], cols[1:nnzjF])
  if m > 0
    idx = nnzjF .+ (1:nlp.internal.meta.nnzj)
    @views jac_structure!(nlp.internal, rows[idx], cols[idx])
    rows[idx] .+= ne
  end
  rows[end-ne+1:end] .= 1:ne
  cols[end-ne+1:end] .= n .+ (1:ne)
  return rows, cols
end

function jac_coord!(nlp :: FeasibilityFormNLS, xr :: AbstractVector, vals :: AbstractVector)
  @lencheck nlp.meta.nvar xr
  @lencheck nlp.meta.nnzj vals
  n, m, ne = nlp.internal.meta.nvar, nlp.internal.meta.ncon, nlp.internal.nls_meta.nequ
  x = @view xr[1:n]
  nnzjF = nlp.internal.nls_meta.nnzj
  nnzjc = m > 0 ? nlp.internal.meta.nnzj : 0
  I = 1:nnzjF
  @views jac_coord_residual!(nlp.internal, x, vals[I])
  if m > 0
    I = nnzjF+1:nnzjF+nnzjc
    @views jac_coord!(nlp.internal, x, vals[I])
  end
  vals[nnzjF+nnzjc+1:nnzjF+nnzjc+ne] .= -1
  return vals
end

function jprod!(nlp :: FeasibilityFormNLS, xr :: AbstractVector, v :: AbstractVector, jv :: AbstractVector)
  @lencheck nlp.meta.nvar xr v
  @lencheck nlp.meta.ncon jv
  increment!(nlp, :neval_jprod)
  n, m, ne = nlp.internal.meta.nvar, nlp.internal.meta.ncon, nlp.internal.nls_meta.nequ
  x = @view xr[1:n]
  @views jprod_residual!(nlp.internal, x, v[1:n], jv[1:ne])
  @views jv[1:ne] .-= v[n+1:end]
  if m > 0
    @views jprod!(nlp.internal, x, v[1:n], jv[ne+1:end])
  end
  return jv
end

function jtprod!(nlp :: FeasibilityFormNLS, xr :: AbstractVector, v :: AbstractVector, jtv :: AbstractVector)
  @lencheck nlp.meta.nvar xr jtv
  @lencheck nlp.meta.ncon v
  increment!(nlp, :neval_jtprod)
  n, m, ne = nlp.internal.meta.nvar, nlp.internal.meta.ncon, nlp.internal.nls_meta.nequ
  x = @view xr[1:n]
  @views jtprod_residual!(nlp.internal, x, v[1:ne], jtv[1:n])
  if m > 0
    @views jtv[1:n] .+= jtprod(nlp.internal, x, v[ne+1:end])
  end
  @views jtv[n+1:end] .= -v[1:ne]
  return jtv
end

function hess_structure!(nlp :: FeasibilityFormNLS, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck nlp.meta.nnzh rows cols
  n, m, ne = nlp.internal.meta.nvar, nlp.internal.meta.ncon, nlp.internal.nls_meta.nequ
  nnzhF = nlp.internal.nls_meta.nnzh
  nnzhc = m > 0 ? nlp.internal.meta.nnzh : 0
  I = 1:nnzhF
  @views hess_structure_residual!(nlp.internal, rows[I], cols[I])
  if m > 0
    I = nnzhF+1:nnzhF+nnzhc
    @views hess_structure!(nlp.internal, rows[I], cols[I])
  end
  I = nnzhF+nnzhc+1:nnzhF+nnzhc+ne
  rows[I] .= n+1:n+ne
  cols[I] .= n+1:n+ne
  return rows, cols
end

function hess_structure!(nlp :: FeasibilityFormNLS{<: AbstractLLSModel}, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck nlp.meta.nnzh rows cols
  n, ne = nlp.internal.meta.nvar, nlp.internal.nls_meta.nequ
  rows .= n+1:n+ne
  cols .= n+1:n+ne
  return rows, cols
end

function hess_coord!(nlp :: FeasibilityFormNLS, xr :: AbstractVector, y :: AbstractVector, vals :: AbstractVector;
                     obj_weight :: Float64=1.0)
  @lencheck nlp.meta.nvar xr
  @lencheck nlp.meta.ncon y
  @lencheck nlp.meta.nnzh vals
  increment!(nlp, :neval_hess)
  n, m, ne = nlp.internal.meta.nvar, nlp.internal.meta.ncon, nlp.internal.nls_meta.nequ
  nnzhF = nlp.internal.nls_meta.nnzh
  nnzhc = m > 0 ? nlp.internal.meta.nnzh : 0
  x = @view xr[1:n]
  y1 = @view y[1:ne]
  y2 = @view y[ne+1:ne+m]
  I = 1:nnzhF
  @views hess_coord_residual!(nlp.internal, x, y1, vals[I])
  if m > 0
    I = nnzhF+1:nnzhF+nnzhc
    @views hess_coord!(nlp.internal, x, y2, vals[I], obj_weight=0.0)
  end
  vals[nnzhF+nnzhc+1:nnzhF+nnzhc+ne] .= obj_weight
  return vals
end

function hess_coord!(nlp :: FeasibilityFormNLS{<: AbstractLLSModel}, xr :: AbstractVector, y :: AbstractVector, vals :: AbstractVector;
                     obj_weight :: Float64=1.0)
  @lencheck nlp.meta.nvar xr
  @lencheck nlp.meta.ncon y
  @lencheck nlp.meta.nnzh vals
  increment!(nlp, :neval_hess)
  vals .= obj_weight
  return vals
end

function hess(nlp :: FeasibilityFormNLS, xr :: AbstractVector; obj_weight :: Float64=1.0)
  @lencheck nlp.meta.nvar xr
  increment!(nlp, :neval_hess)
  n, ne = nlp.internal.meta.nvar, nlp.internal.nls_meta.nequ
  return [spzeros(n, n + ne); spzeros(ne, n) obj_weight * I]
end

function hess(nlp :: FeasibilityFormNLS, xr :: AbstractVector, y :: AbstractVector;
              obj_weight :: Float64=1.0)
  @lencheck nlp.meta.nvar xr
  @lencheck nlp.meta.ncon y
  increment!(nlp, :neval_hess)
  n, m, ne = nlp.internal.meta.nvar, nlp.internal.meta.ncon, nlp.internal.nls_meta.nequ
  x = @view xr[1:n]
  @views Hx = m > 0 ? hess(nlp.internal, x, y[ne+1:end], obj_weight=0.0) : spzeros(n, n)
  Hx += hess_residual(nlp.internal, x, @view y[1:ne])
  return [Hx spzeros(n, ne); spzeros(ne, n) obj_weight * I]
end

function hprod!(nlp :: FeasibilityFormNLS, xr :: AbstractVector, y :: AbstractVector, v :: AbstractVector, hv :: AbstractVector;
    obj_weight :: Float64=1.0)
  @lencheck nlp.meta.nvar xr v hv
  @lencheck nlp.meta.ncon y
  n, m, ne = nlp.internal.meta.nvar, nlp.internal.meta.ncon, nlp.internal.nls_meta.nequ
  x = @view xr[1:n]
  if m > 0
    @views hprod!(nlp.internal, x, y[ne+1:end], v[1:n], hv[1:n], obj_weight=0.0)
  else
    fill!(hv, 0.0)
  end
  for i = 1:ne
    @views hv[1:n] .+= hprod_residual(nlp.internal, x, i, v[1:n]) * y[i]
  end
  @views hv[n+1:end] .= obj_weight * v[n+1:end]
  return hv
end

function residual!(nlp :: FeasibilityFormNLS, x :: AbstractVector, Fx :: AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.nls_meta.nequ Fx
  increment!(nlp, :neval_residual)
  n = nlp.internal.meta.nvar
  Fx .= @view x[n+1:end]
  return Fx
end

function jac_structure_residual!(nlp :: FeasibilityFormNLS, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck nlp.nls_meta.nnzj rows cols
  n, ne = nlp.internal.meta.nvar, nlp.internal.nls_meta.nequ
  rows .= 1:ne
  cols .= n .+ (1:ne)
  return rows, cols
end

function jac_coord_residual!(nlp :: FeasibilityFormNLS, x :: AbstractVector, vals :: AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.nls_meta.nnzj vals
  increment!(nlp, :neval_jac_residual)
  vals[1:nlp.nls_meta.nnzj] .= 1
  return vals
end

function jprod_residual!(nlp :: FeasibilityFormNLS, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  @lencheck nlp.meta.nvar x v
  @lencheck nlp.nls_meta.nequ Jv
  increment!(nlp, :neval_jprod_residual)
  n = nlp.internal.meta.nvar
  Jv .= @view v[n+1:end]
  return Jv
end

function jtprod_residual!(nlp :: FeasibilityFormNLS, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  @lencheck nlp.meta.nvar x Jtv
  @lencheck nlp.nls_meta.nequ v
  increment!(nlp, :neval_jtprod_residual)
  n, ne = nlp.internal.meta.nvar, nlp.internal.nls_meta.nequ
  Jtv[1:n] .= 0.0
  Jtv[n+1:end] .= v
  return Jtv
end

function hess_structure_residual!(nlp :: FeasibilityFormNLS, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck nlp.nls_meta.nnzh rows cols
  return rows, cols  # Hessian of residual is zero; do not change rows and cols
end

function hess_coord_residual!(nlp :: FeasibilityFormNLS, x :: AbstractVector, v :: AbstractVector, vals :: AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.nls_meta.nequ v
  @lencheck nlp.nls_meta.nnzh vals
  increment!(nlp, :neval_hess_residual)
  return vals
end

function jth_hess_residual(nlp :: FeasibilityFormNLS, x :: AbstractVector, i :: Int)
  @lencheck nlp.meta.nvar x
  increment!(nlp, :neval_jhess_residual)
  n = nlp.meta.nvar
  return spzeros(n, n)
end

function hprod_residual!(nlp :: FeasibilityFormNLS, x :: AbstractVector, i :: Int, v :: AbstractVector, Hiv :: AbstractVector)
  @lencheck nlp.meta.nvar x v Hiv
  increment!(nlp, :neval_hprod_residual)
  fill!(Hiv, 0.0)
  return Hiv
end
