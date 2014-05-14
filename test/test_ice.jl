import Base.Test.@test

using Autoreload
using PyPlot
using Logging
Logging.configure(level=INFO)
#Logging.configure(level=DEBUG)

using DtProtos.ice
using DtProtos.pdfs

#arequire("../src/ice.jl")
#arequire("../src/pdfs.jl")

#include("../src/ice.jl")

#reload("DtProtos")

## reload("../src/DtProtos.jl")
## reload("../src/pdfs.jl")
## reload("../src/ice.jl")

#include("../src/DtProtos.jl")
#include("../src/ice.jl")


[close() for i = [1:10]]




lb, ub = -7.0, 7.0
lb, ub = -70.0, 70.0
lb, ub = -4.0, 8.0
lb, ub = -4.0, 8.0
## n = GaussianPdf(0,1,1)
#p1 = GaussianPdf( -3.0, 0.5, 1.0)
## p1 = GaussianPdf( -3.0, 2.0, 1.0)
## p2 = GaussianPdf( 0.0, 2.0, 1.0)
## p3 = GaussianPdf( 7.0, 1.5, 1.0)
p1 = GaussianPdf( 20.0, 7.0, 1.0)
p2 = GaussianPdf( 22.0, 6.0, 1.0)
p3 = GaussianPdf( 23.0, 7.5, 1.0)
#p1 = BoundedGaussianPdf(-3.0, 2.0, 1.0, lb, ub)
#p2 = BoundedGaussianPdf( 0.0, 1.0, 1.0, lb, ub)
#p3 = BoundedGaussianPdf( 7.0, 1.5, 1.0, lb, ub)
p = SimplePdf[]
#push!(p,p1); push!(p,p2); push!(p,p3)
push!(p,p2); push!(p,p2); push!(p,p2)
#push!(p,p3); push!(p,p3); push!(p,p3)
n = Pdf(p)

#n = GaussianPdf(0,1,1)
## x = [3*lb:(ub-lb)/20:2*ub]
## x = [0.0:60.0]
## fx = [pdf(n,xi) for xi in x]
## plot(x,fx,"o-")
## draw()
## show()

##########################
# Optimised controlpoints (fromPdfControlPoints)
##########################

#cps = [lb+0.02:0.5:ub-0.01]
#nice = fromPdfControlPoints(n, cps, false, false)
println("baz")
p1 = GaussianPdf( 20.0, 8.0, 1.0)
## 14.6041 25.3959 
## -3.22585 -3.22585 
## 0.227468 
## 1 1
## 20
## 64
p = SimplePdf[]
push!(p,p1);
n = Pdf(p)
#nice = fromPdfScale(n, 15, 0.001) #4
nice = fromPdfScale(n, 15, 0.001) #
info("controlPoints     : ", join(nice.controlPoints, " "))
info("logarithmOfDensity: ", join(nice.logarithmOfDensity, " "))
info("curvatures        : ", join(nice.curvatures, " "))
info("left, right       : $(nice.hasLeftTail), $(nice.hasRightTail)")
info("n                 : $(length(nice.controlPoints))")

figure()
x = [0.0:60.0]
fx = [pdf(n,xi) for xi in x]
plot(x,fx,"o-")

fx = [pdf(n,xi) for xi in nice.controlPoints]
plot(nice.controlPoints, fx, "o-")

figure()
plot(nice.controlPoints, "o-")



# ##########################
# # Hardcoded cnotrolpoints (fromPdfControlPoints)
# ##########################

# # nice = fromPdf(n)
# # #cps = [-3:0.5:3]
# # cps = [-6:0.5:7] # f, t
# #cps = [-6:0.5:15]  # t, f
# cps = [lb+0.02:0.5:ub-0.01]

# #cps = [lb:0.5:ub]
# #nice = fromPdfControlPoints(n, cps, true, true)
# nice = fromPdfControlPoints(n, cps, false, false)
# #info("nice $nice")
# info("controlPoints     : $(nice.controlPoints)")
# info("logarithmOfDensity: $(nice.logarithmOfDensity)")
# info("curvatures        : $(nice.curvatures)")
# info("left, right       : $(nice.hasLeftTail), $(nice.hasRightTail)")

# x = cps
# fx = [pdf(n,xi) for xi in x]
# Fx = [cdf(n,xi) for xi in x]# prob
# #invFp = [inverseCumulative(n, Fxi) for Fxi in Fx]
# invFp = [pdf(n, quantile(n, Fxi)) for Fxi in Fx]
# plot(x,fx)
# plot(x,Fx)
# plot(x,invFp,"o")
