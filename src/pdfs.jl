module pdfs

# package code goes here
using PyPlot
using Distributions
#import Distributions
using Logging
Logging.configure(level=WARNING)
export SimplePdf, GaussianPdf, BoundedGaussianPdf,Pdf, pdf, cdf, quantile

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


function pdf(dist::GaussianPdf, x::Float64)
    f = Normal(dist.mu, dist.sigma)
    Distributions.pdf(f,x)
end

function pdf(dist::BoundedGaussianPdf, x::Float64)
    if dist.lower <= x <= dist.upper
        f = Distributions.Normal(dist.mu, dist.sigma)
        probability = Distributions.pdf(f,x)
    else
        probability = 0.0
    end
    #debug("Need to rescale this??")
    return probability
end

function pdf(dist::Pdf, x::Float64)
    res = 0
    weights_sum = 0
    for cmp in dist.components
        weights_sum += cmp.weight
        #f = Distributions.Normal(cmp.mu, cmp.sigma)
        #val = cmp.weight * Distributions.pdf(f,x)
        val = cmp.weight * pdf(cmp, x) #Distributions.pdf(f,x)
        res += val
    end
    res / weights_sum
end

function cdf(dist::GaussianPdf, x::Number)
    F = Distributions.Normal(dist.mu, dist.sigma)
    Distributions.cdf(F,x) 
    end

function cdf(dist::BoundedGaussianPdf, x::Number)
    f = Distributions.Normal(dist.mu, dist.sigma)
    if  (dist.lower <= x <= dist.upper)
        F = Distributions.cdf(f,x) - Distributions.cdf(f, dist.lower)
    elseif (x < dist.lower)
        F = 0.0
        #F = Distributions.cdf(f, dist.lower)
    elseif (x > dist.upper)
        #F = 1.0
        F = Distributions.cdf(f,dist.upper) - Distributions.cdf(f, dist.lower)
    else
        error("Crazy x")
    end
    #scale = Distributions.cdf(f, dist.upper) - Distributions.cdf(f, dist.lower)
    return F    
end

function cdf(dist::Pdf, x::Number)
    res = 0
    weights_sum = 0
    for cmp in dist.components
        weights_sum += cmp.weight
        #f = Distributions.Normal(cmp.mu, cmp.sigma)
        res += cmp.weight * cdf(cmp, x) #Distributions.cdf(f,x)
    end
    res / weights_sum
end

function quantile(dist::GaussianPdf, probability::Number)
    f = Distributions.Normal(dist.mu, dist.sigma)
    x = Distributions.quantile(f,probability)
    return x
    #fx = Distributions.pdf(f,x)
    #return fx

    end

# function quantile(dist::Pdf, probability::Number)
#     res = 0
#     weights_sum = 0
#     for cmp in dist.components
#         weights_sum += cmp.weight
#         res += cmp.weight * quantile(cmp, probability)
#         #res += quantile(cmp, probability)
#     end
#     res / weights_sum
# end

function quantile(dist::Pdf, probability::Number)
    # find x (or fx)for which F(x) = prob using bisection method
    tol = 0.001
    
    # Choose endpoints
    # # automatic ... 
    # a = 0 #-100
    # b = 0 #100 
    # for cmp in dist.components
    #     a = min(a, quantile(cmp, tol))
    #     b = max(b, quantile(cmp, 1-tol))
    #     #println(a, " ", b)
    # end
    # # if probability < cdf(dist, a) || probability > cdf(dist, b)
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
        pc = cdf(dist, c) - probability
        if abs(pc) < tol
            println("xxx $(cdf(dist, c)) - $probability")
            break #return c
        end
        n += 1
        pa = cdf(dist, a) - probability
        #pb = cdf(b) - probability
        if sign(pc) == sign(pa)
            a = c
        else
            b = c
        end
    end
    if n == nmax + 1
        warn("Reached max iter $nmax, tol = $tol, pc = $pc, a, b = $a, $b")
    end
    debug("quantile niter = $n, c = $c")
    return c #,f
    #return pdf(dist, c)
end

# function quantile(dist::Pdf)
#     #
#     lp = 0.05 # lower probability 
#     rp = 1 - lp
#     l = 0
#     r = 0
#     for cmp in dist.components
#         l = min(l, quantile(cmp, lp))
#         r = max(r, quantile(cmp, rp))
#         println("l $lp ", quantile(cmp, lp))
#         println("r $rp ", quantile(cmp, rp))
#         println(l, " ", r)
#     end
#     dx = (r-l) / 100
#     x = [l:dx:r]
#     #
#     #f = [pdf(dist, xi) for xi=x]
#     return x #,f
# end


end # module
