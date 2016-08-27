# NLPModels.jl documentation

This package provides general guidelines to represent optimization problems in
Julia and a standardized API to evaluate the functions and their derivatives.
The main objective is to be able to rely on that API when designing optimization
solvers in Julia.

The general form of the optimization problem is
\begin{align*}
\min \quad & f(x) \\\\
& c_i(x) = 0, \quad i \in E, \\\\
& c_{L_i} \leq c_i(x) \leq c_{U_i}, \quad i \in I, \\\\
& \ell \leq x \leq u,
\end{align*}
where $f:\mathbb{R}^n\rightarrow\mathbb{R}$,
$c:\mathbb{R}^n\rightarrow\mathbb{R}^m$,
$E\cup I = \\{1,2,\dots,m\\}$, $E\cap I = \emptyset$,
and
$c_{L_i}, c_{U_i}, \ell_j, u_j \in \mathbb{R}\cup\\{\pm\infty\\}$
for $i = 1,\dots,m$ and $j = 1,\dots,n$.

For computational reasons, we write
\begin{align*}
\min \quad & f(x) \\\\
& c_L \leq c(x) \leq c_U \\\\
& \ell \leq x \leq u,
\end{align*}
defining $c_{L_i} = c_{U_i}$ for all $i \in E$.

Optimization problems are represented by an instance/subtype of `AbstractNLPModel`.
Such instances are composed of

- an instance of `NLPModelMeta`, which provides information about the problem,
  including the number of variables, constraints, bounds on the variables, etc.
- other data specific to the provenance of the problem.

## Internal Interfaces

 - [`SimpleNLPModel`](@ref): Uses
   [`ForwardDiff`](http://github.com/JuliaDiff/ForwardDiff.jl) to compute the
   derivatives. It has a very simple interface.
 - [`JuMPNLPModel`](@ref): Uses a [`JuMP`](https://github.com/JuliaOpt/JuMP.jl) model.
  - [`SlackModel`](@ref): Creates an equality constrained problem with bounds
    on the variables using an existing NLPModel.

## External Interfaces

 - `AmplModel`: Defined in
   [`AmplNLReader.jl`](https://github.com/JuliaSmoothOptimizers/AmplNLReader.jl)
   for problems modeled using [AMPL](http://www.ampl.com)
 - `CUTEstModel`: Defined in
   [`CUTEst.jl`](https://github.com/JuliaSmoothOptimizers/CUTEst.jl) for
   problems from [CUTEst](https://ccpforge.cse.rl.ac.uk/gf/project/cutest/wiki).

If you want your interface here, open a PR.
