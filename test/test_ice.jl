#import Base.Test.@test

using Base.Test
#using Autoreload
using PyPlot
#using Logging
#Logging.configure(level=INFO)
#Logging.configure(level=DEBUG)

using DtProtos.ice
using DtProtos.pdfs


function readPdf(fname::String)
    x = Float64[]
    y = Float64[]
    c = Float64[]
    ifs = open(fname, "r");
    push!(x, float64(split(readline(ifs)))...)
    push!(y, float64(split(readline(ifs)))...)
    push!(c, float64(split(readline(ifs)))...)
    ltail, rtail = bool(int(split(readline(ifs))))
    ipdf = IcePdf(x,y,c,ltail,rtail);
end


#function testI()
ifs = open("data/IcePDFtestI.txt", "r");
for i = [1:140]
    z, a, b, c, y = float64(split(readline(ifs)))
    if z == 1024
        z = Inf
    end
    if z == -1024
        z = -Inf
    end 
    y1 = ice.I(z,a,b,c);
    #println("$i $z, $a, $b, $c, $y, $y1")
    @test abs((y1-y)) <= 1e-4*abs(y)
end


#void testEvaluate()
pdf1 = readPdf("data/pdf1.dat");
pdf2 = readPdf("data/pdf2.dat");
pdf3 = readPdf("data/pdf3.dat");
pdf4 = readPdf("data/pdf4.dat");
pdf5 = readPdf("data/pdf5.dat");
ifs = open("data/densitytest.dat", "r");
for i = [1:50]
    x, y1, y2, y3, y4, y5 = float64(split(readline(ifs)))
    y = ice.pdf(pdf1, x);
    #//cout << x << ' ' << y1 << ' ' << y << endl;
    @test abs(y-y1) < 1e-8
    y = ice.pdf(pdf2, x)
    #//cout << x << ' ' << y2 << ' ' << y << endl;
    @test abs(y-y2) < 1e-8
    y = ice.pdf(pdf3, x)
    #//cout << x << ' ' << y3 << ' ' << y << endl;
    @test abs(y-y3) < 1e-8
    y = ice.pdf(pdf4, x)
    #//cout << x << ' ' << y4 << ' ' << y << endl;
    @test abs(y-y4) < 1e-8
    y = ice.pdf(pdf5, x)
    #//cout << x << ' ' << y5 << ' ' << y << endl;
    @test abs(y-y5) < 1e-8
end


#void testArea()
ipdf1 = readPdf("data/pdf1.dat")
ipdf2 = readPdf("data/pdf2.dat")
ipdf3 = readPdf("data/pdf3.dat")
ipdf4 = readPdf("data/pdf4.dat")
ipdf5 = readPdf("data/pdf5.dat")
@test abs(ice.area(pdf1) - 1) < 1e-4
@test abs(ice.area(pdf2) - 1) < 1e-4
@test abs(ice.area(pdf3) - 1) < 1e-4
@test abs(ice.area(pdf4) - 1) < 1e-4
@test abs(ice.area(pdf5) - 1) < 1e-4

#void testNormalise()
x = [0., 1.3, 2.]
y = [.3, .8, -.1]
c = [1., 3.]
ipdf = IcePdf(x, y, c, false, false)
ice.normalise!(ipdf)
@test abs(ice.area(ipdf) - 1) < 1e-4
            

#void testCumulative()
ipdf1 = readPdf("data/pdf1.dat")
ipdf2 = readPdf("data/pdf2.dat")
ipdf3 = readPdf("data/pdf3.dat")
ipdf4 = readPdf("data/pdf4.dat")
ipdf5 = readPdf("data/pdf5.dat")
ifs = open("data/cumulativetest.dat", "r")
for i = [1:50]
    x, y1, y2, y3, y4, y5 = float64(split(readline(ifs)))
    y = ice.cdf(ipdf1, x)
    #println(i, " ", x, " ", y1, " ", y)
    @test abs(y-y1) < 1e-4
    y = ice.cdf(ipdf2, x)
    #println(i, " ", x, " ", y2, " ", y)
    @test abs(y-y2) < 1e-4
    y = ice.cdf(ipdf3, x)
    #println(i, " ", x, " ", y3, " ", y)
    @test abs(y-y3) < 1e-4
    y = ice.cdf(ipdf4, x)
    #println(i, " ", x, " ", y4, " ", y)
    @test abs(y-y4) < 1e-4
    y = ice.cdf(ipdf5, x)
    #println(i, " ", x, " ", y5, " ", y)
    @test abs(y-y5) < 1e-4
end

# lb, ub = -7.0, 7.0
# lb, ub = -70.0, 70.0
# lb, ub = -4.0, 8.0
# lb, ub = -4.0, 8.0
# ## n = GaussianPdf(0,1,1)
# #p1 = GaussianPdf( -3.0, 0.5, 1.0)
# ## p1 = GaussianPdf( -3.0, 2.0, 1.0)
# ## p2 = GaussianPdf( 0.0, 2.0, 1.0)
# ## p3 = GaussianPdf( 7.0, 1.5, 1.0)
# p1 = GaussianPdf( 20.0, 7.0, 1.0)
# p2 = GaussianPdf( 22.0, 6.0, 1.0)
# p3 = GaussianPdf( 23.0, 7.5, 1.0)
# #p1 = BoundedGaussianPdf(-3.0, 2.0, 1.0, lb, ub)
# #p2 = BoundedGaussianPdf( 0.0, 1.0, 1.0, lb, ub)
# #p3 = BoundedGaussianPdf( 7.0, 1.5, 1.0, lb, ub)
# p = SimplePdf[]
# #push!(p,p1); push!(p,p2); push!(p,p3)
# push!(p,p2); push!(p,p2); push!(p,p2)
# #push!(p,p3); push!(p,p3); push!(p,p3)
# n = Pdf(p)

# #n = GaussianPdf(0,1,1)
# ## x = [3*lb:(ub-lb)/20:2*ub]
# ## x = [0.0:60.0]
# ## fx = [pdf(n,xi) for xi in x]
# ## plot(x,fx,"o-")
# ## draw()
# ## show()

# ##########################
# # Optimised controlpoints (fromPdfScale)
# ##########################

# #cps = [lb+0.02:0.5:ub-0.01]
# #nice = fromPdfControlPoints(n, cps, false, false)
# println("baz")
# p1 = GaussianPdf( 20.0, 8.0, 1.0)
# ## 14.6041 25.3959 
# ## -3.22585 -3.22585 
# ## 0.227468 
# ## 1 1
# ## 20
# ## 64
# p = SimplePdf[]
# push!(p,p1);
# n = Pdf(p)
# nice = fromPdfScale(n, 15, 0.001) #
# info("controlPoints     : ", join(nice.controlPoints, " "))
# info("logarithmOfDensity: ", join(nice.logarithmOfDensity, " "))
# info("curvatures        : ", join(nice.curvatures, " "))
# info("left, right       : $(nice.hasLeftTail), $(nice.hasRightTail)")
# info("n                 : $(length(nice.controlPoints))")
# # 14.6041 25.3959 
# # -3.22585 -3.22585 
# # 0.227468 
# # 1 1
# # 20
# # 64

# figure()
# x = [0.0:60.0]
# fx = [pdf(n,xi) for xi in x]
# plot(x,fx,"o-")

# fx = [pdf(n,xi) for xi in nice.controlPoints]
# plot(nice.controlPoints, fx, "o-")

# figure()
# plot(nice.controlPoints, "o-")



# # ##########################
# # # Hardcoded cnotrolpoints (fromPdfControlPoints)
# # ##########################

# # # nice = fromPdf(n)
# # # #cps = [-3:0.5:3]
# # # cps = [-6:0.5:7] # f, t
# # #cps = [-6:0.5:15]  # t, f
# # cps = [lb+0.02:0.5:ub-0.01]

# # #cps = [lb:0.5:ub]
# # #nice = fromPdfControlPoints(n, cps, true, true)
# # nice = fromPdfControlPoints(n, cps, false, false)
# # #info("nice $nice")
# # info("controlPoints     : $(nice.controlPoints)")
# # info("logarithmOfDensity: $(nice.logarithmOfDensity)")
# # info("curvatures        : $(nice.curvatures)")
# # info("left, right       : $(nice.hasLeftTail), $(nice.hasRightTail)")

# # x = cps
# # fx = [pdf(n,xi) for xi in x]
# # Fx = [cdf(n,xi) for xi in x]# prob
# # #invFp = [inverseCumulative(n, Fxi) for Fxi in Fx]
# # invFp = [pdf(n, quantile(n, Fxi)) for Fxi in Fx]
# # plot(x,fx)
# # plot(x,Fx)
# # plot(x,invFp,"o")
