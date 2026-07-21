# test.jl
using DifferentialEquations

println("Hello from Julia! Starting math execution...")

f(u, p, t) = 1.01 * u
u0 = 1/2
tspan = (0.0, 1.0)
prob = ODEProblem(f, u0, tspan)

sol = solve(prob)

global final_result = sol.u[end]
println("Julia finished computing.")
