import Base.Test.@test
using PyPlot
using DtProtos.pdf
include("../src/pdf.jl")


    
# SimplePdf
n = GaussianPdf(0,1,1)
println("eval $(evaluate(n,1.3))")
println("cdf $(cumulative(n,9))")
println("cdf $(cumulative(n,0))")

x = 1.3
fx = evaluate(n,x)
Fx = cumulative(n,x) # prob
invFp = inverseCumulative(n, Fx)
println("x = $x")
println("f(x) = $fx")
println("F(x) = $Fx = probability")
println("invF(prob) = invF(Fx) = $invFp")
println("invFp == fx : $(invFp == fx)" )

# Pdf
p1 = GaussianPdf( 0.0, 1.0, 1.0)
p2 = GaussianPdf( 0.0, 1.0, 1.0)
p3 = GaussianPdf( 0.0, 1.0, 1.0)
#p2 = SimplePdf( 1.0, 0.5, 1.0)
#p3 = SimplePdf(-0.5, 0.6, 1.0)

p = SimplePdf[]
push!(p,p1)
push!(p,p2)
push!(p,p3)

P = Pdf(p)
println("eval $(evaluate(P,1.3))")
println("cdf $(cumulative(P,9))")
println("cdf $(cumulative(P,0))")

# inverseCumulative - SimplePdf
n = GaussianPdf(0,1,1)
x = [-10:0.5:10]
fx = [evaluate(n,xi) for xi in x]
Fx = [cumulative(n,xi) for xi in x]# prob
#invFp = [inverseCumulative(n, Fxi) for Fxi in Fx]
invFp = [evaluate(n, inverseCumulative(n, Fxi)) for Fxi in Fx]
plot(x,fx)
plot(x,Fx)
plot(x,invFp,"o")

# inverseCumulative - Pdf
figure()
p1 = GaussianPdf( -3.0, 0.5, 1.0)
p2 = GaussianPdf( 0.0, 1.0, 1.0)
p3 = GaussianPdf( 7.0, 1.5, 1.0)

p = SimplePdf[]
push!(p,p1)
push!(p,p2)
push!(p,p3)

n = Pdf(p)
x = [-10:0.05:10]
fx = [evaluate(n,xi) for xi in x]
Fx = [cumulative(n,xi) for xi in x]# prob
#invFp = [inverseCumulative(n, Fxi) for Fxi in Fx]
invFp = [evaluate(n, inverseCumulative(n, Fxi)) for Fxi in Fx]
plot(x,fx)
plot(x,Fx)
plot(x,invFp,"o")
#invFp = inverseCumulative(n)
#plot(invFp[1],invFp[2],"o")

