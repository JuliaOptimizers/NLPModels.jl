export LLSMatrixModel, LLSOperatorModel, LLSTripletModel, LLSModel, AbstractLLSModel

abstract type AbstractLLSModel <: AbstractNLSModel end

show_header(io :: IO, nls :: AbstractLLSModel) = println(io, "AbstractLLSModel - Linear least-squares model")

mutable struct LLSMatrixModel <: AbstractLLSModel
  meta :: NLPModelMeta
  nls_meta :: NLSMeta
  counters :: NLSCounters

  A
  b :: Vector
  C
end

mutable struct LLSOperatorModel <: AbstractLLSModel
  meta :: NLPModelMeta
  nls_meta :: NLSMeta
  counters :: NLSCounters

  A :: LinearOperator
  b :: Vector
  C :: LinearOperator
end

mutable struct LLSTripletModel <: AbstractLLSModel
  meta :: NLPModelMeta
  nls_meta :: NLSMeta
  counters :: NLSCounters

  Arows :: Vector{Int}
  Acols :: Vector{Int}
  Avals :: Vector
  b :: Vector
  Crows :: Vector{Int}
  Ccols :: Vector{Int}
  Cvals :: Vector
end

"""
    nls = LLSModel(A, b; lvar, uvar, C, lcon, ucon, variant)
    nls = LLSModel(opA, b; lvar, uvar, opC, lcon, ucon)
    nls = LLSModel(Arows, Acols, Avals, b; lvar, uvar, Crows, Ccols, Cvals, lcon, ucon, variant)

Creates a Linear Least Squares model ``\\tfrac{1}{2}\\|Ax - b\\|^2`` with optional bounds
`lvar ≦ x ≦ uvar` and optional linear constraints `lcon ≦ Cx ≦ ucon`.
This problem is a linear least-squares problem with residual given by ``F(x) = Ax - b``.

There are three internal structures for `AbstractLLSModel`s, one storing the matrices explicitly, the
other storing `LinearOperator`s, and the last storing the triplet format of the matrices.
They correspond, respectively, to each of the three function calls shown.
Additionally, the `variant` can be used to transform the input into a different internal structure.
The option can be `:matrix`, `:operator`, and `:triplet`. Notice that the function signature
for LinearOperators input does not accept the variant option.
"""
function LLSModel end

function LLSMatrixModel(A :: AbstractMatrix, b :: AbstractVector;
                        x0 :: AbstractVector = zeros(eltype(A), size(A,2)),
                        lvar :: AbstractVector = fill(eltype(A)(-Inf), size(A, 2)),
                        uvar :: AbstractVector = fill(eltype(A)(Inf), size(A, 2)),
                        C :: AbstractMatrix  = Matrix{eltype(A)}(undef, 0, 0),
                        lcon :: AbstractVector = eltype(A)[],
                        ucon :: AbstractVector = eltype(A)[],
                        y0 :: AbstractVector = zeros(eltype(A), size(C,1)),
                        name :: String = "generic-LLSModel"
                       )
  nequ, nvar = size(A)
  ncon = size(C, 1)
  nnzjF = issparse(A) ? nnz(A) : nequ * nvar
  nnzh  = issparse(A) ? nnz(tril(A' * A)) : div(nvar * (nvar + 1), 2)
  nnzj  = issparse(C) ? nnz(C) : ncon * nvar
  meta = NLPModelMeta(nvar, x0=x0, lvar=lvar, uvar=uvar, ncon=ncon, y0=y0, lin=1:ncon,
                      nln=Int[], lcon=lcon, ucon=ucon, nnzj=nnzj, nnzh=nnzh, name=name)
  nls_meta = NLSMeta(nequ, nvar, nnzj=nnzjF, nnzh=0, lin=1:nequ, nln=Int[])

  LLSMatrixModel(meta, nls_meta, NLSCounters(), A, b, C)
end

function LLSOperatorModel(A :: LinearOperator, b :: AbstractVector;
                          x0 :: AbstractVector = zeros(eltype(A), size(A,2)),
                          lvar :: AbstractVector = fill(eltype(A)(-Inf), size(A, 2)),
                          uvar :: AbstractVector = fill(eltype(A)(Inf), size(A, 2)),
                          C :: LinearOperator  = opZeros(eltype(A), 0, size(A, 2)),
                          lcon :: AbstractVector = eltype(A)[],
                          ucon :: AbstractVector = eltype(A)[],
                          y0 :: AbstractVector = zeros(eltype(A), size(C,1)),
                          name :: String = "generic-LLSModel"
                         )
  nequ, nvar = size(A)
  ncon = size(C, 1)
  meta = NLPModelMeta(nvar, x0=x0, lvar=lvar, uvar=uvar, ncon=ncon, y0=y0, lin=1:ncon,
                      nln=Int[], lcon=lcon, ucon=ucon, nnzj=0, nnzh=0, name=name)
  nls_meta = NLSMeta(nequ, nvar, nnzj=0, nnzh=0, lin=1:nequ, nln=Int[])

  LLSOperatorModel(meta, nls_meta, NLSCounters(), A, b, C)
end

function LLSTripletModel(Arows :: AbstractVector{<: Integer},
                         Acols :: AbstractVector{<: Integer},
                         Avals :: AbstractVector,
                         nvar :: Integer,
                         b :: AbstractVector;
                         x0 :: AbstractVector = zeros(eltype(Avals), nvar),
                         lvar :: AbstractVector = fill(eltype(Avals)(-Inf), nvar),
                         uvar :: AbstractVector = fill(eltype(Avals)(Inf), nvar),
                         Crows :: AbstractVector{<: Integer} = eltype(Arows)[],
                         Ccols :: AbstractVector{<: Integer} = eltype(Arows)[],
                         Cvals :: AbstractVector = eltype(Avals)[],
                         lcon :: AbstractVector = eltype(Avals)[],
                         ucon :: AbstractVector = eltype(Avals)[],
                         y0 :: AbstractVector = zeros(eltype(Avals), length(lcon)),
                         name :: String = "generic-LLSModel"
                        )
  ncon = length(lcon)
  @lencheck ncon ucon y0
  nnzjF = length(Avals)
  @lencheck nnzjF Arows Acols
  nnzj = length(Cvals)
  @lencheck nnzj Crows Ccols
  nequ = length(b)

  meta = NLPModelMeta(nvar, x0=x0, lvar=lvar, uvar=uvar, ncon=ncon, y0=y0, lin=1:ncon,
                      nln=Int[], lcon=lcon, ucon=ucon, nnzj=nnzj, nnzh=0, name=name)

  nls_meta = NLSMeta(nequ, nvar, nnzj=nnzjF, nnzh=0, lin=1:nequ, nln=Int[])

  LLSTripletModel(meta, nls_meta, NLSCounters(), Arows, Acols, Avals, b, Crows, Ccols, Cvals)
end

function LLSModel(A :: AbstractMatrix, b :: AbstractVector;
                  variant = :matrix, kwargs...)
  if !(variant in [:matrix, :operator, :triplet])
    error("variant should be one of :matrix, :operator, :triplet")
  end
  LLSModel(Val(variant), A, b; kwargs...)
end

function LLSModel(::Val{:matrix}, A :: AbstractMatrix, b :: AbstractVector; kwargs...)
  LLSMatrixModel(A, b; kwargs...)
end

function LLSModel(::Val{:operator}, A :: AbstractMatrix, b :: AbstractVector;
                  C :: AbstractMatrix  = Matrix{eltype(A)}(undef, 0, 0),
                  kwargs...
                 )
  LLSOperatorModel(LinearOperator(A), b; C=LinearOperator(C), kwargs...)
end

function LLSModel(::Val{:triplet}, A :: AbstractMatrix, b :: AbstractVector;
                  C :: AbstractMatrix  = Matrix{eltype(A)}(undef, 0, 0),
                  kwargs...
                 )
  nvar = size(A, 2)
  Arows, Acols, Avals = if A isa AbstractSparseMatrix
    findnz(A)
  else
    m, n = size(A)
    I = ((i,j) for i = 1:m, j = 1:n)
    getindex.(I, 1)[:], getindex.(I, 2)[:], A[:]
  end
  Crows, Ccols, Cvals = if C isa AbstractSparseMatrix
    findnz(C)
  else
    m, n = size(C)
    I = ((i,j) for i = 1:m, j = 1:n)
    getindex.(I, 1)[:], getindex.(I, 2)[:], C[:]
  end
  LLSTripletModel(Arows, Acols, Avals, nvar, b; Crows=Crows, Ccols=Ccols, Cvals=Cvals, kwargs...)
end

function LLSModel(A :: LinearOperator, b :: AbstractVector; kwargs...)
  LLSOperatorModel(A, b; kwargs...)
end

function LLSModel(Arows :: AbstractVector{<: Integer},
                  Acols :: AbstractVector{<: Integer},
                  Avals :: AbstractVector,
                  nvar :: Integer,
                  b :: AbstractVector;
                  variant=:triplet, kwargs...)
  if !(variant in [:matrix, :operator, :triplet])
    error("variant should be one of :matrix, :operator, :triplet")
  end
  LLSModel(Val(variant), Arows, Acols, Avals, nvar, b; kwargs...)
end

function LLSModel(::Val{:matrix},
                  Arows :: AbstractVector{<: Integer},
                  Acols :: AbstractVector{<: Integer},
                  Avals :: AbstractVector,
                  nvar :: Integer,
                  b :: AbstractVector;
                  x0 :: AbstractVector = zeros(eltype(Avals), nvar),
                  lvar :: AbstractVector = fill(eltype(Avals)(-Inf), nvar),
                  uvar :: AbstractVector = fill(eltype(Avals)(Inf), nvar),
                  Crows :: AbstractVector{<: Integer} = eltype(Arows)[],
                  Ccols :: AbstractVector{<: Integer} = eltype(Arows)[],
                  Cvals :: AbstractVector = eltype(Avals)[],
                  lcon :: AbstractVector = eltype(Avals)[],
                  ucon :: AbstractVector = eltype(Avals)[],
                  y0 :: AbstractVector = zeros(eltype(Avals), length(lcon)),
                  name :: String = "generic-LLSModel"
                 )
  nequ = length(b)
  ncon = length(lcon)
  A = sparse(Arows, Acols, Avals, nequ, nvar)
  C = sparse(Crows, Ccols, Cvals, ncon, nvar)

  LLSMatrixModel(A, b, x0=x0, lvar=lvar, uvar=uvar,
                 C=C, lcon=lcon, ucon=ucon, y0=y0, name=name)
end

function LLSModel(::Val{:operator},
                  Arows :: AbstractVector{<: Integer},
                  Acols :: AbstractVector{<: Integer},
                  Avals :: AbstractVector,
                  nvar :: Integer,
                  b :: AbstractVector;
                  x0 :: AbstractVector = zeros(eltype(Avals), nvar),
                  lvar :: AbstractVector = fill(eltype(Avals)(-Inf), nvar),
                  uvar :: AbstractVector = fill(eltype(Avals)(Inf), nvar),
                  Crows :: AbstractVector{<: Integer} = eltype(Arows)[],
                  Ccols :: AbstractVector{<: Integer} = eltype(Arows)[],
                  Cvals :: AbstractVector = eltype(Avals)[],
                  lcon :: AbstractVector = eltype(Avals)[],
                  ucon :: AbstractVector = eltype(Avals)[],
                  y0 :: AbstractVector = zeros(eltype(Avals), length(lcon)),
                  name :: String = "generic-LLSModel"
                 )

  ncon = length(lcon)
  @lencheck ncon ucon y0
  nnzjF = length(Avals)
  @lencheck nnzjF Arows Acols
  nnzj = length(Cvals)
  @lencheck nnzj Crows Ccols
  nequ = length(b)

  TA, TC = eltype(Avals), eltype(Cvals)
  Av = zeros(TA, nequ)
  Atv = zeros(TA, nvar)
  Aprod(v)  = coo_prod!(Arows, Acols, Avals, v, Av)
  Atprod(v) = coo_prod!(Acols, Arows, Avals, v, Atv)
  Cv = zeros(TC, ncon)
  Ctv = zeros(TC, nvar)
  Cprod(v)  = coo_prod!(Crows, Ccols, Cvals, v, Cv)
  Ctprod(v) = coo_prod!(Ccols, Crows, Cvals, v, Ctv)
  A = LinearOperator{TA}(nequ, nvar, false, false, Aprod, Atprod, Atprod)
  C = LinearOperator{TC}(ncon, nvar, false, false, Cprod, Ctprod, Ctprod)

  LLSOperatorModel(A, b, x0=x0, lvar=lvar, uvar=uvar,
                   C=C, lcon=lcon, ucon=ucon, y0=y0, name=name)
end

function LLSModel(::Val{:triplet},
                  Arows :: AbstractVector{<: Integer},
                  Acols :: AbstractVector{<: Integer},
                  Avals :: AbstractVector,
                  nvar :: Integer,
                  b :: AbstractVector;
                  kwargs...)
  LLSTripletModel(Arows, Acols, Avals, nvar, b; kwargs...)
end

function residual!(nls :: AbstractLLSModel, x :: AbstractVector, Fx :: AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nequ Fx
  increment!(nls, :neval_residual)
  Fx .= nls.A * x - nls.b
  return Fx
end

function residual!(nls :: LLSTripletModel, x :: AbstractVector, Fx :: AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nequ Fx
  increment!(nls, :neval_residual)
  coo_prod!(nls.Arows, nls.Acols, nls.Avals, x, Fx)
  Fx .-= nls.b
  return Fx
end

function jac_structure_residual!(nls :: LLSMatrixModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck nls.nls_meta.nnzj rows
  @lencheck nls.nls_meta.nnzj cols
  if issparse(nls.A)
    I, J, V = findnz(nls.A)
    rows .= I
    cols .= J
  else
    m, n = size(nls.A)
    I = ((i,j) for i = 1:m, j = 1:n)
    rows .= getindex.(I, 1)[:]
    cols .= getindex.(I, 2)[:]
  end
  return rows, cols
end

function jac_structure_residual!(nls :: LLSTripletModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck nls.nls_meta.nnzj rows
  @lencheck nls.nls_meta.nnzj cols
  rows .= nls.Arows
  cols .= nls.Acols
  return rows, cols
end

function jac_coord_residual!(nls :: LLSMatrixModel, x :: AbstractVector, vals :: AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nnzj vals
  increment!(nls, :neval_jac_residual)
  if issparse(nls.A)
    vals .= nls.A.nzval
  else
    vals .= nls.A[:]
  end
  return vals
end

function jac_coord_residual!(nls :: LLSTripletModel, x :: AbstractVector, vals :: AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nnzj vals
  increment!(nls, :neval_jac_residual)
  vals .= nls.Avals
  return vals
end

function jac_residual(nls :: LLSMatrixModel, x :: AbstractVector)
  @lencheck nls.meta.nvar x
  increment!(nls, :neval_jac_residual)
  return nls.A
end

jac_op_residual(nls :: LLSOperatorModel, args...) = nls.A

function jprod_residual!(nls :: AbstractLLSModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.meta.nvar v
  @lencheck nls.nls_meta.nequ Jv
  increment!(nls, :neval_jprod_residual)
  Jv .= nls.A * v
  return Jv
end

function jprod_residual!(nls :: LLSTripletModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.meta.nvar v
  @lencheck nls.nls_meta.nequ Jv
  increment!(nls, :neval_jprod_residual)
  coo_prod!(nls.Arows, nls.Acols, nls.Avals, v, Jv)
  return Jv
end

function jtprod_residual!(nls :: AbstractLLSModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nequ v
  @lencheck nls.meta.nvar Jtv
  increment!(nls, :neval_jtprod_residual)
  Jtv .= nls.A' * v
  return Jtv
end

function jtprod_residual!(nls :: LLSTripletModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nequ v
  @lencheck nls.meta.nvar Jtv
  increment!(nls, :neval_jtprod_residual)
  coo_prod!(nls.Acols, nls.Arows, nls.Avals, v, Jtv)
  return Jtv
end

function hess_residual(nls :: AbstractLLSModel, x :: AbstractVector, v :: AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nequ v
  increment!(nls, :neval_hess_residual)
  n = nls.meta.nvar
  return spzeros(n, n)
end

function hess_structure_residual!(nls :: AbstractLLSModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck 0 rows
  @lencheck 0 cols
  return rows, cols
end

function hess_coord_residual!(nls :: AbstractLLSModel, x :: AbstractVector, v :: AbstractVector, vals :: AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nequ v
  @lencheck 0 vals
  increment!(nls, :neval_hess_residual)
  return vals
end

function jth_hess_residual(nls :: AbstractLLSModel, x :: AbstractVector, i :: Int)
  @lencheck nls.meta.nvar x
  increment!(nls, :neval_jhess_residual)
  n = nls.meta.nvar
  return spzeros(n, n)
end

function hprod_residual!(nls :: AbstractLLSModel, x :: AbstractVector, i :: Int, v :: AbstractVector, Hiv :: AbstractVector)
  @lencheck nls.meta.nvar x v Hiv
  increment!(nls, :neval_hprod_residual)
  fill!(Hiv, zero(eltype(x)))
  return Hiv
end

function cons!(nls :: AbstractLLSModel, x :: AbstractVector, c :: AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.meta.ncon c
  increment!(nls, :neval_cons)
  c .= nls.C * x
  return c
end

function cons!(nls :: LLSTripletModel, x :: AbstractVector, c :: AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.meta.ncon c
  increment!(nls, :neval_cons)
  coo_prod!(nls.Crows, nls.Ccols, nls.Cvals, x, c)
  return c
end

function jac_structure!(nls :: LLSMatrixModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck nls.meta.nnzj rows cols
  if issparse(nls.C)
    I, J, V = findnz(nls.C)
    rows .= I
    cols .= J
  else
    m, n = size(nls.C)
    I = ((i,j) for i = 1:m, j = 1:n)
    rows .= getindex.(I, 1)[:]
    cols .= getindex.(I, 2)[:]
  end
  return rows, cols
end

function jac_structure!(nls :: LLSTripletModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck nls.meta.nnzj rows cols
  rows .= nls.Crows
  cols .= nls.Ccols
  return rows, cols
end

function jac_coord!(nls :: LLSMatrixModel, x :: AbstractVector, vals :: AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.meta.nnzj vals
  increment!(nls, :neval_jac)
  if issparse(nls.C)
    vals .= nls.C.nzval
  else
    vals .= nls.C[:]
  end
  return vals
end

function jac_coord!(nls :: LLSTripletModel, x :: AbstractVector, vals :: AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.meta.nnzj vals
  increment!(nls, :neval_jac)
  vals .= nls.Cvals
  return vals
end

function jac(nls :: LLSMatrixModel, x :: AbstractVector)
  @lencheck nls.meta.nvar x
  increment!(nls, :neval_jac)
  return nls.C
end

jac_op(nls :: LLSOperatorModel, args...) = nls.C

function jprod!(nls :: AbstractLLSModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  @lencheck nls.meta.nvar x v
  @lencheck nls.meta.ncon Jv
  increment!(nls, :neval_jprod)
  Jv .= nls.C * v
  return Jv
end

function jprod!(nls :: LLSTripletModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  @lencheck nls.meta.nvar x v
  @lencheck nls.meta.ncon Jv
  increment!(nls, :neval_jprod)
  coo_prod!(nls.Crows, nls.Ccols, nls.Cvals, v, Jv)
  return Jv
end

function jtprod!(nls :: AbstractLLSModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  @lencheck nls.meta.nvar x Jtv
  @lencheck nls.meta.ncon v
  increment!(nls, :neval_jtprod)
  Jtv .= nls.C' * v
  return Jtv
end

function jtprod!(nls :: LLSTripletModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  @lencheck nls.meta.nvar x Jtv
  @lencheck nls.meta.ncon v
  increment!(nls, :neval_jtprod)
  coo_prod!(nls.Ccols, nls.Crows, nls.Cvals, v, Jtv)
  return Jtv
end

function hprod!(nls :: AbstractLLSModel, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight = 1.0)
  @lencheck nls.meta.nvar x v Hv
  increment!(nls, :neval_hprod)
  Hv .= obj_weight .* (nls.A' * (nls.A * v))
  return Hv
end

function hprod!(nls :: LLSTripletModel, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight = 1.0)
  @lencheck nls.meta.nvar x v Hv
  increment!(nls, :neval_hprod)
  Av = zeros(nls.nls_meta.nequ)
  coo_prod!(nls.Arows, nls.Acols, nls.Avals, v, Av)
  coo_prod!(nls.Acols, nls.Arows, nls.Avals, Av, Hv)
  Hv .*= obj_weight
  return Hv
end

function hprod!(nls :: AbstractLLSModel, x :: AbstractVector, y :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight = 1.0)
  @lencheck nls.meta.nvar x v Hv
  @lencheck nls.meta.ncon y
  hprod!(nls, x, v, Hv, obj_weight=obj_weight)
end

function hess(nls :: LLSMatrixModel, x :: AbstractVector; obj_weight = 1.0)
  @lencheck nls.meta.nvar x
  n = nls.meta.nvar
  obj_weight == 0 && return spzeros(eltype(nls.A), n, n)
  return tril(obj_weight * (nls.A' * nls.A))
end

hess(nls :: LLSMatrixModel, x :: AbstractVector, y :: AbstractVector; obj_weight = 1.0) = hess(nls, x, obj_weight=obj_weight)

function hess_structure!(nls :: LLSMatrixModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck nls.meta.nnzh rows cols
  AtA = tril(nls.A' * nls.A)
  if issparse(AtA)
    I, J, V = findnz(AtA)
    rows .= I
    cols .= J
  else
    n = size(nls.A, 2)
    I = ((i,j) for i = 1:n, j = 1:n if i ≥ j)
    rows .= getindex.(I, 1)
    cols .= getindex.(I, 2)
  end
  return rows, cols
end

function hess_coord!(nls :: LLSMatrixModel, x :: AbstractVector, vals :: AbstractVector; obj_weight = 1.0)
  @lencheck nls.meta.nvar x
  @lencheck nls.meta.nnzh vals
  increment!(nls, :neval_hess)
  AtA = tril(nls.A' * nls.A)
  if issparse(AtA)
    vals .= obj_weight * AtA.nzval
  else
    n = size(nls.A, 2)
    vals .= (obj_weight * AtA[i,j] for i = 1:n, j = 1:n if i ≥ j)
  end
  return vals
end
