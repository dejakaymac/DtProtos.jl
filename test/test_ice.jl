import Base.Test.@test
using PyPlot
using DtProtos.ice
using DtProtos.pdfs

#include("../src/DtProtos.jl")
#include("../src/ice.jl")


[close() for i = [1:10]]


## n = GaussianPdf(0,1,1)
#p1 = GaussianPdf( -3.0, 0.5, 1.0)
p1 = GaussianPdf( -3.0, 2.0, 1.0)
p2 = GaussianPdf( 0.0, 1.0, 1.0)
p3 = GaussianPdf( 7.0, 1.5, 1.0)
#p1 = BoundedGaussianPdf(-3.0, 2.0, 1.0, -6, 7)
#p2 = BoundedGaussianPdf( 0.0, 1.0, 1.0, -6, 7)
#p3 = BoundedGaussianPdf( 7.0, 1.5, 1.0, -6, 7)
p = SimplePdf[]
push!(p,p1); push!(p,p2); push!(p,p3)
#push!(p,p2); push!(p,p2); push!(p,p2)
n = Pdf(p)

#n = GaussianPdf(0,1,1)

nice = fromPdf(n)
#cps = [-3:0.5:3]
cps = [-6:0.5:7] # f, t
#cps = [-6:2.0:15]  # t, f
cps = [-15:0.5:15] # f, t

nice = fromPdfControlPoints(n, cps, true, true)
#nice = fromPdfControlPoints(n, cps, false, false)
#info("nice $nice")
info("controlPoints     : $(nice.controlPoints)")
info("logarithmOfDensity: $(nice.logarithmOfDensity)")
info("curvatures        : $(nice.curvatures)")
info("left, right       : $(nice.hasLeftTail), $(nice.hasRightTail)")

x = cps
fx = [pdf(n,xi) for xi in x]
Fx = [cdf(n,xi) for xi in x]# prob
#invFp = [inverseCumulative(n, Fxi) for Fxi in Fx]
invFp = [pdf(n, quantile(n, Fxi)) for Fxi in Fx]
plot(x,fx)
plot(x,Fx)
plot(x,invFp,"o")
