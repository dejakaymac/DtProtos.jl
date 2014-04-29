import Base.Test.@test
using PyPlot
using DtProtos.pdfs
include("../src/pdfs.jl")


    
# SimplePdf
n = GaussianPdf(0,1,1)
println("eval $(pdf(n,1.3))")
println("cdf $(cdf(n,9))")
println("cdf $(cdf(n,0))")

x = 1.3
fx = pdf(n,x)
Fx = cdf(n,x) # prob
invFp = quantile(n, Fx)
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
println("eval $(pdf(P,1.3))")
println("cdf $(cdf(P,9))")
println("cdf $(cdf(P,0))")

# quantile - SimplePdf
n = GaussianPdf(0,1,1)
x = [-10:0.5:10]
fx = [pdf(n,xi) for xi in x]
Fx = [cdf(n,xi) for xi in x]# prob
#invFp = [quantile(n, Fxi) for Fxi in Fx]
invFp = [pdf(n, quantile(n, Fxi)) for Fxi in Fx]
plot(x,fx)
plot(x,Fx)
plot(x,invFp,"o")

# quantile - Pdf
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
fx = [pdf(n,xi) for xi in x]
Fx = [cdf(n,xi) for xi in x]# prob
#invFp = [quantile(n, Fxi) for Fxi in Fx]
invFp = [pdf(n, quantile(n, Fxi)) for Fxi in Fx]
plot(x,fx)
plot(x,Fx)
plot(x,invFp,"o")
#invFp = quantile(n)
#plot(invFp[1],invFp[2],"o")

# quantile - Bounded Pdf
figure()
lb, ub = -4.0, 8.0
dx = 0.05
p1 = BoundedGaussianPdf(-3.0, 0.5, 1.0, lb, ub)
p2 = BoundedGaussianPdf( 0.0, 1.0, 1.0, lb, ub)
p3 = BoundedGaussianPdf( 7.0, 1.5, 1.0, lb, ub)

p = SimplePdf[]
push!(p,p1)
push!(p,p2)
push!(p,p3)

n = Pdf(p)
x = [lb+dx/1.5:dx:ub-dx/2]
fx = [pdf(n,xi) for xi in x]
Fx = [cdf(n,xi) for xi in x]# prob
#invFp = [quantile(n, Fxi) for Fxi in Fx]
invFp = [pdf(n, quantile(n, Fxi)) for Fxi in Fx]
plot(x,fx)
plot(x,Fx)
plot(x,invFp,"o")
#invFp = quantile(n)
#plot(invFp[1],invFp[2],"o")

