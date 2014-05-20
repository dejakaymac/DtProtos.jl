module Pdfs

# package code goes here
using PyPlot
using Distributions
using Logging

export SimplePdf, GaussianPdf, BoundedGaussianPdf,Pdf, pdf, cdf, quantile, normalise!

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


function normalise!(dist::Pdf, N::Number)
    sumofweights = 0.0
    for i = [1:length(dist.components)]
        sumofweights += dist.components[i].weight
    end
    for i = [1:length(dist.components)]
        dist.components[i].weight = N * dist.components[i].weight / sumofweights
    end
    return dist
end
    ## void PDF::normalise(const float normalValue)
    ## {
    ##     if (size() > 0)
    ##     {
    ##         float weightsSum  = sumOfWeights();
    ##         for (size_t index = 0; index < size(); ++index)
    ##         {
    ##             weights[index] = normalValue * weights[index] / weightsSum;
    ##         }
    ##     }
    ## }


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

function cdf!(dist::GaussianPdf, x::Number)
    error("Not implelemented yet")
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
    #a = cdf(dist, tol)
    #b = cdf(dist, 1.0-tol)
    a = cdf(dist, 0.5)
    b = cdf(dist, 0.5)
    for cmp in dist.components
        a = min(a, quantile(cmp, tol/2))
        b = max(b, quantile(cmp, 1-tol/2))
        #a = min(a, quantile(cmp, 0.0))
        #b = max(b, quantile(cmp, 1.0))
    end
    nmax = 1000
    n = 0
    c = -99
    pc = -99
    while n <= nmax
        c = (a + b)/2
        pc = cdf(dist, c) - probability
        if abs(pc) <= tol
            info("cdf(c) - p = $(cdf(dist, c)) - $probability")
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
        warn("Reached max iter $nmax, tol = $tol, ",
             "cdf(c) = $(round(pc+probability,3)), p = $probability, ",
             "a, b, c = $a, $b, $c")
    end
    debug("quantile niter = $n, c = $c")
    return c #,f
    #return pdf(dist, c)
end


end # module
