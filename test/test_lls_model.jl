function lls_test()
  constructors = [("matrices", (A, b, C; kwargs...) -> LLSModel(A, b; C=C, kwargs...)),
                  ("triplet format", (A, b, C; kwargs...) -> begin
                     fA = findnz(sparse(A))
                     fC = findnz(sparse(C))
                     LLSModel(fA..., size(A, 2), b; Crows=fC[1], Ccols=fC[2], Cvals=fC[3], kwargs...)
                   end)
                 ]
  types_per_variant = Dict(:matrix => LLSMatrixModel,
                           :operator => LLSOperatorModel,
                           :triplet => LLSTripletModel)
  @testset "lls_test" begin
    for variant in [:matrix, :operator, :triplet], (name, con) in constructors
      @testset "Constructor from $name variant=$variant" begin
        for A = [Matrix(1.0I, 10, 3) .+ 1, sparse(1.0I, 10, 3) .+ 1],
            C = [zeros(0, 3), ones(1, 3), [ones(1,3); -I], sparse(ones(1,3))]

          b = collect(1:10)
          nequ, nvar = size(A)
          ncon = size(C,1)

          nls = con(A, b, C, variant=variant, lcon=zeros(ncon), ucon=zeros(ncon))
          @test typeof(nls) == types_per_variant[variant]

          x = [1.0; -1.0; 1.0]
          @test isapprox(A * x - b, residual(nls, x), rtol=1e-8)
          I, J = hess_structure_residual(nls)
          V = hess_coord_residual(nls, x, ones(nequ))
          @test sparse(I, J, V, nvar, nvar) == zeros(nvar, nvar)
          @test hess_residual(nls, x, ones(nequ)) == zeros(nvar,nvar)
          for i = 1:nequ
            @test isapprox(zeros(nvar, nvar), jth_hess_residual(nls, x, i), rtol=1e-8)
          end

          @test nls.meta.nlin == length(nls.meta.lin) == ncon
          @test nls.meta.nnln == length(nls.meta.nln) == 0

          if variant != :operator
            @test A == jac_residual(nls, x)
            I, J = jac_structure_residual(nls)
            V = jac_coord_residual(nls, x)
            @test A == sparse(I, J, V, nequ, nvar)

            @test C == jac(nls, x)
            I, J = jac_structure(nls)
            V = jac_coord(nls, x)
            @test C == sparse(I, J, V, ncon, nvar)
          end

          # Improving coverage
          @test jprod_residual(nls, x, ones(nvar)) == A * ones(nvar)
          @test jtprod_residual(nls, x, ones(nequ)) == A' * ones(nequ)
          @test hprod(nls, x, ones(nvar), obj_weight=0.5) == 0.5 * A' * A * ones(nvar)
          if ncon > 0
            @test jprod(nls, x, ones(nvar)) == C * ones(nvar)
            @test jtprod(nls, x, ones(ncon)) == C' * ones(ncon)
          end
        end
      end
    end

    @testset "Other constructors" begin
      A = LinearOperator(rand(10, 3))
      b = rand(10)
      C = LinearOperator(rand(1, 3))
      nls = LLSModel(A, b, C=C, lcon=[0.0], ucon=[0.0])
    end

    @testset "Hess related functions" begin
      # dense
      b = rand(10)
      A = rand(10, 5)
      AtA = A' * A
      lls = LLSModel(A, b)
      x = lls.meta.x0
      I, J = hess_structure(lls)
      V = hess_coord(lls, x, obj_weight=0.5)
      ijv = ((i,j,AtA[i,j]) for i = 1:5, j = 1:5 if i ≥ j)
      @test I == getindex.(ijv, 1)
      @test J == getindex.(ijv, 2)
      @test V == 0.5 * getindex.(ijv, 3)
      @test hess(lls, x, obj_weight=0.5) == 0.5 * tril(A' * A)
      @test hess(lls, x, [0.0], obj_weight=0.5) == 0.5 * tril(A' * A)
      # sparse
      A = sprand(10, 5, 0.2)
      lls = LLSModel(A, b)
      I, J = hess_structure(lls)
      V = hess_coord(lls, x, obj_weight=0.5)
      @test (I, J, 2V) == findnz(tril(A' * A))
      @test hess(lls, x, obj_weight=0.5) == 0.5 * tril(A' * A)
      @test hess(lls, x, [0.0], obj_weight=0.5) == 0.5 * tril(A' * A)
    end

    @testset "Wrong variant" begin
      @test_throws ErrorException LLSModel(rand(10,3), rand(10), variant=:wrong_variant)
      @test_throws ErrorException LLSModel([1,2], [1,1], ones(2), 1, [1.0; 2.0], variant=:wrong_variant)
    end
    
    @testset "jac_op of LLSOperatorModel" begin
      A = LinearOperator(rand(10, 3))
      C = LinearOperator(rand(2, 3))
      lls = LLSModel(A, ones(10), C=C, lcon=zeros(2), ucon=zeros(2))
      @test jac_op_residual(lls) == A
      @test jac_op(lls) == C
    end
  end
end

lls_test()
