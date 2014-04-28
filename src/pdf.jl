module pdf

# package code goes here
using PyPlot
using Distributions
using Logging
Logging.configure(level=OFF)
export SimplePdf, GaussianPdf, BoundedGaussianPdf, Pdf, evaluate, cumulative,
inverseCumulative

abstract SimplePdf # must have: mu::Float64, sigma::Float64, weight::Float64

type GaussianPdf <: SimplePdf # move us to pdf module
    mu::Float64
    sigma::Float64
    weight::Float64
    ## lower::Float64
    ## upper::Float64
end
    
type BoundedGaussianPdf <: SimplePdf
    mu::Float64
    sigma::Float64
    weight::Float64
    lower::Float64
    upper::Float64
end
    
type Pdf <:SimplePdf
    components::Array{SimplePdf,1}
end
## Alt use 
## julia> ?Distributions.MultivariateNormal
## DataType   : GenericMvNormal{PDMat} (constructor with 5 methods)
##   supertype: AbstractMvNormal
##   fields   : (:dim,:zeromean,:μ,:Σ)
## with identity corelation matrix


function evaluate(pdf::GaussianPdf, x::Float64)
    f = Distributions.Normal(pdf.mu, pdf.sigma)
    Distributions.pdf(f,x)
end

function evaluate(pdf::BoundedGaussianPdf, x::Float64)
    if pdf.lower <= x <= pdf.upper
        f = Distributions.Normal(pdf.mu, pdf.sigma)
        probability = Distributions.pdf(f,x)
    else
        probability = 0.0
    end
    debug("Need to rescale this??")
    return probability
end

function evaluate(pdf::Pdf, x::Float64)
    res = 0
    weights_sum = 0
    for cmp in pdf.components
        weights_sum += cmp.weight
        #f = Distributions.Normal(cmp.mu, cmp.sigma)
        #val = cmp.weight * Distributions.pdf(f,x)
        val = cmp.weight * evaluate(cmp, x) #Distributions.pdf(f,x)
        res += val
    end
    res / weights_sum
end

function cumulative(pdf::GaussianPdf, x::Number)
    F = Distributions.Normal(pdf.mu, pdf.sigma)
    Distributions.cdf(F,x) 
    end

function cumulative(pdf::BoundedGaussianPdf, x::Number)
    f = Distributions.Normal(pdf.mu, pdf.sigma)
    if  (pdf.lower <= x <= pdf.upper)
        F = Distributions.cdf(f,x) - Distributions.cdf(f, pdf.lower)
    elseif (x < pdf.lower)
        F = 0.0
    elseif (x > pdf.upper)
        F = 1.0
    else
        error("Crazy x")
    end
    #scale = Distributions.cdf(f, pdf.upper) - Distributions.cdf(f, pdf.lower)
    return F    
end

function cumulative(pdf::Pdf, x::Number)
    res = 0
    weights_sum = 0
    for cmp in pdf.components
        weights_sum += cmp.weight
        #f = Distributions.Normal(cmp.mu, cmp.sigma)
        res += cmp.weight * cumulative(cmp, x) #Distributions.cdf(f,x)
    end
    res / weights_sum
end

function inverseCumulative(pdf::GaussianPdf, probability::Number)
    f = Distributions.Normal(pdf.mu, pdf.sigma)
    x = Distributions.quantile(f,probability)
    return x
    #fx = Distributions.pdf(f,x)
    #return fx

    end

# function inverseCumulative(pdf::Pdf, probability::Number)
#     res = 0
#     weights_sum = 0
#     for cmp in pdf.components
#         weights_sum += cmp.weight
#         res += cmp.weight * inverseCumulative(cmp, probability)
#         #res += inverseCumulative(cmp, probability)
#     end
#     res / weights_sum
# end

function inverseCumulative(pdf::Pdf, probability::Number)
    # find x (or fx)for which F(x) = prob using bisection method
    tol = 0.001
    
    # Choose endpoints
    # # automatic ... 
    # a = 0 #-100
    # b = 0 #100 
    # for cmp in pdf.components
    #     a = min(a, inverseCumulative(cmp, tol))
    #     b = max(b, inverseCumulative(cmp, 1-tol))
    #     #println(a, " ", b)
    # end
    # # if probability < cumulative(pdf, a) || probability > cumulative(pdf, b)
    # #     do something
    # # end
    # err("Weird behaviour ... everything still ok when probability lies outside [a,b]")
    # dx = b - a
    # #println(a, " ", b)
    # a = a - dx
    # b = b + dx
    # debug("a = $a, b = $b")
    # Choose endpoints
    # hardcoded
    a = -100.0
    b = 100.0

    #
    nmax = 1000
    n = 0
    c = -99
    pc = -99
    while n <= nmax
        c = (a + b)/2
        pc = cumulative(pdf, c) - probability
        if abs(pc) < tol
            break #return c
        end
        n += 1
        pa = cumulative(pdf, a) - probability
        #pb = cumulative(b) - probability
        if sign(pc) == sign(pa)
            a = c
        else
            b = c
        end
    end
    if n == nmax + 1
        warn("Reached max iter $nmax, tol = $tol, pc = $pc, a, b = $a, $b")
    end
    debug("inverseCumulative niter = $n, c = $c")
    return c #,f
    #return evaluate(pdf, c)
end

# function inverseCumulative(pdf::Pdf)
#     #
#     lp = 0.05 # lower probability 
#     rp = 1 - lp
#     l = 0
#     r = 0
#     for cmp in pdf.components
#         l = min(l, inverseCumulative(cmp, lp))
#         r = max(r, inverseCumulative(cmp, rp))
#         println("l $lp ", inverseCumulative(cmp, lp))
#         println("r $rp ", inverseCumulative(cmp, rp))
#         println(l, " ", r)
#     end
#     dx = (r-l) / 100
#     x = [l:dx:r]
#     #
#     #f = [evaluate(pdf, xi) for xi=x]
#     return x #,f
# end

    
## # SimplePdf
## n = GaussianPdf(0,1,1)
## println("eval $(evaluate(n,1.3))")
## println("cdf $(cumulative(n,9))")
## println("cdf $(cumulative(n,0))")

## x = 1.3
## fx = evaluate(n,x)
## Fx = cumulative(n,x) # prob
## invFp = inverseCumulative(n, Fx)
## println("x = $x")
## println("f(x) = $fx")
## println("F(x) = $Fx = probability")
## println("invF(prob) = invF(Fx) = $invFp")
## println("invFp == fx : $(invFp == fx)" )

## # Pdf
## p1 = GaussianPdf( 0.0, 1.0, 1.0)
## p2 = GaussianPdf( 0.0, 1.0, 1.0)
## p3 = GaussianPdf( 0.0, 1.0, 1.0)
## #p2 = SimplePdf( 1.0, 0.5, 1.0)
## #p3 = SimplePdf(-0.5, 0.6, 1.0)

## p = SimplePdf[]
## push!(p,p1)
## push!(p,p2)
## push!(p,p3)

## P = Pdf(p)
## println("eval $(evaluate(P,1.3))")
## println("cdf $(cumulative(P,9))")
## println("cdf $(cumulative(P,0))")

## # inverseCumulative - SimplePdf
## n = GaussianPdf(0,1,1)
## x = [-10:0.5:10]
## fx = [evaluate(n,xi) for xi in x]
## Fx = [cumulative(n,xi) for xi in x]# prob
## #invFp = [inverseCumulative(n, Fxi) for Fxi in Fx]
## invFp = [evaluate(n, inverseCumulative(n, Fxi)) for Fxi in Fx]
## plot(x,fx)
## plot(x,Fx)
## plot(x,invFp,"o")

## # inverseCumulative - Pdf
## figure()
## p1 = GaussianPdf( -3.0, 0.5, 1.0)
## p2 = GaussianPdf( 0.0, 1.0, 1.0)
## p3 = GaussianPdf( 7.0, 1.5, 1.0)

## p = SimplePdf[]
## push!(p,p1)
## push!(p,p2)
## push!(p,p3)

## n = Pdf(p)
## x = [-10:0.05:10]
## fx = [evaluate(n,xi) for xi in x]
## Fx = [cumulative(n,xi) for xi in x]# prob
## #invFp = [inverseCumulative(n, Fxi) for Fxi in Fx]
## invFp = [evaluate(n, inverseCumulative(n, Fxi)) for Fxi in Fx]
## plot(x,fx)
## plot(x,Fx)
## plot(x,invFp,"o")
## #invFp = inverseCumulative(n)
## #plot(invFp[1],invFp[2],"o")


end # module
