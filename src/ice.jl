module ice

# package code goes here
using PyPlot
using Logging
Logging.configure(level=DEBUG)
#include("pdf.jl")
using DtProtos.pdf


# type IcePdf # move me to ice module
#     a::Float64
#     b::Float64
#     c::Float64
# end

type IcePdf # move me to ice module
    controlPoints::Array{Float64,1}
    logarithmOfDensity::Array{Float64,1}
    curvatures::Array{Float64,1}
    hasLeftTail::Bool
    hasRightTail::Bool
end

function normalise!(icepdf::IcePdf)
    # #assertNonEmpty();
    la = log(area(icepdf));
    for i = [1:length(icepdf.logarithmOfDensity)] #(unsigned i=0;i<myLogarithmOfDensity.size();++i)
        icepdf.logarithmOfDensity[i] -= la;
    end
    return icepdf
end

function area(icepdf::IcePdf)
    #assertNonEmpty();
    r = 0.0;
    for j = [1:1:length(icepdf.curvatures)] # (unsigned j=0;j<myCurvatures.size();++j)
        alpha = (icepdf.controlPoints[j+1] - icepdf.controlPoints[j])/2;
        r += A(icepdf, j)*alpha;
    end
    return r;
end

function A(icepdf::IcePdf, i::Number)
    # assertNonEmpty();
    a = (icepdf.logarithmOfDensity[i] + icepdf.logarithmOfDensity[i+1])/2;
    b = (icepdf.logarithmOfDensity[i+1] - icepdf.logarithmOfDensity[i])/2;
    if (i==0 && i==length(icepdf.curvatures) && icepdf.HasLeftTail && icepdf.HasRightTail)
        return I(Inf,a,b,icepdf.curvatures[i])-I(-Inf,a,b,icepdf.curvatures[i]);
    end
    if (i==0 && icepdf.hasLeftTail)
        return I(1,a,b,icepdf.curvatures[i])-I(-Inf,a,b,icepdf.curvatures[i]);
    end
    if (i==length(icepdf.curvatures) && icepdf.hasRightTail)
        return I(Inf,a,b,icepdf.curvatures[i])-I(-1,a,b,icepdf.curvatures[i]);
    end
    return I(1,a,b,icepdf.curvatures[i])-I(-1,a,b,icepdf.curvatures[i]);
end



# does fromPDFControlPoints use InverseCumulative() much?
# if so this is bad

function fromPdf(pdf::SimplePdf)
    println("From SimplePdf")
end

function fromPdf(pdf::GaussianPdf)
    l = cumulative(pdf, 0.25)
    r = cumulative(pdf, 0.75)
    x=0
end

function fromPdf(pdf::BoundedGaussianPdf)
    println("Not implemented")
    #raise/throw error
end

function C(b::Number, c::Number)
    if (abs(b*b/c)>30)
        return 0.0;
    end
    return c;
end

function I(z::Number,
           a::Number,
           b::Number,
           c::Number)
    if (c==0)
        if (b==0)
            @assert(isfinite(z), "z infinite, c=b=0");
            return (z*exp(a));
        end
        @assert(isinf(z)*b <= 0, "z infinite, z*b>0");
        return (exp(a+b*z)/b);
    end

    if (isinf(z))
        @assert(c > 0, "z infinite, c<=0");
        return exp(b*b/4/c+a+c)*sqrt(pi)*isinf(z)/2/sqrt(c);
    end

    @assert(isfinite(z), "z still infinite");
    if (c > 0)
        return exp(b*b/4/c+a+c)*sqrt(pi)*erf((2*c*z-b)/2/sqrt(c))/2/sqrt(c);
    end

    @assert(c < 0, "c should be less than 0");
        
    return exp(b*b/4/c+a+c)*sqrt(pi)*erfi((b-2*c*z)/2/sqrt(-c))/2/sqrt(-c);
end


function findC( x1::Float64, y1::Float64,
                x2::Float64, y2::Float64,
                A::Float64)

    a=(y1+y2)/2;
    b=(y2-y1)/2;
    l=-1;
    r=1;
    m = 0;
    localA = A/(x2-x1)*2;
    while I(1,a,b,C(b,l))-I(-1,a,b,C(b,l)) > localA
        if l < -100.0
            return C(b,-100);
        end
        l *= 2
    end

    while (I(1,a,b,C(b,r)) - I(-1,a,b,C(b,r)) < localA)
        if (r > 100)
            return C(b,100);
        end
        r*=2;
    end

    for i = [1:1:64] #(int i=0; i<64; ++i)
        m=(l+r)/2;
        if (I(1,a,b,C(b,m))-I(-1,a,b,C(b,m)) < localA)
            l = m
        else
            r=m;
        end
    end
    return C(b,m);
end


#function fromPdfControlPoints(pdf::GaussianPdf, 
function fromPdfControlPoints(pdf::SimplePdf,
#function fromPdfControlPoints(pdf::Pdf, 
                              controlPoints::Array{Float64,1},
                              leftTail::Bool,
                              rightTail::Bool)
    logarithmOfDensity = Float64[]
    curvatures = Float64[]
    lod = log(evaluate(pdf, controlPoints[1]));
    if lod < -20
        lod = -20;
    end
    push!(logarithmOfDensity, lod)
    
    for i = [2:1:length(controlPoints)]
        lod = log(evaluate(pdf, controlPoints[i]));
        if lod < -20
            lod = -20;
        end
        push!(logarithmOfDensity, lod)
        push!(curvatures, 
	      findC(
		    controlPoints[i-1],
		    logarithmOfDensity[i-1],
		    controlPoints[i],
		    logarithmOfDensity[i],
		    cumulative(pdf, controlPoints[i])
		    -cumulative(pdf, controlPoints[i-1])
		    )
	      )
    end
    hasLeftTail = curvatures[1] > 0;
    err(hasLeftTail)
    err(curvatures[1])
    err(curvatures)
    if (hasLeftTail) 
        alpha=(controlPoints[2]-controlPoints[1])/2;
        beta=(controlPoints[1]+controlPoints[2])/2;
        b=(logarithmOfDensity[2]-logarithmOfDensity[1])/2;
        err("baz")
        err(controlPoints[1])
        err( b/2/curvatures[1]*alpha+beta)
        hasLeftTail = (controlPoints[1] <= b/2/curvatures[1]*alpha+beta);
    end
    err(hasLeftTail)
    if (hasLeftTail)
        hasLeftTail = leftTail;
        debug("hasLeftTail $hasLeftTail")
    end
    
    hasRightTail = curvatures[length(curvatures)] > 0;
    if (hasRightTail) 
        k = length(curvatures)
        alpha = (controlPoints[k+1]-controlPoints[k])/2;
        beta = (controlPoints[k]+controlPoints[k+1])/2;
        b= (logarithmOfDensity[k+1]-logarithmOfDensity[k])/2;
        hasRightTail = (controlPoints[k+1] >= b/2/curvatures[k]*alpha+beta);
    end
    if (hasRightTail)
        hasRightTail = rightTail;
        debug("hasRightTail $hasRightTail")
    end
    
    #warn("controlPoints     : $(controlPoints)")
    #warn("logarithmOfDensity: $(logarithmOfDensity)")
    #warn("curvatures        : $(curvatures)")
    warn("left, right       : $(hasLeftTail), $(hasRightTail)")
    r = IcePdf(controlPoints,
               logarithmOfDensity,
               curvatures,
               hasLeftTail,
               hasRightTail);
    r = normalise!(r);

    #warn("controlPoints     : $(controlPoints)")
    #warn("logarithmOfDensity: $(logarithmOfDensity)")
    #warn("curvatures        : $(curvatures)")
    warn("left, right       : $(hasLeftTail), $(hasRightTail)")
    return r;
end


[close() for i = [1:10]]


## n = GaussianPdf(0,1,1)
p1 = GaussianPdf( -3.0, 0.5, 1.0)
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

nice = fromPdfControlPoints(n, cps, true, true)
#nice = fromPdfControlPoints(n, cps, false, false)
#info("nice $nice")
info("controlPoints     : $(nice.controlPoints)")
info("logarithmOfDensity: $(nice.logarithmOfDensity)")
info("curvatures        : $(nice.curvatures)")
info("left, right       : $(nice.hasLeftTail), $(nice.hasRightTail)")

x = cps
fx = [evaluate(n,xi) for xi in x]
Fx = [cumulative(n,xi) for xi in x]# prob
#invFp = [inverseCumulative(n, Fxi) for Fxi in Fx]
invFp = [evaluate(n, inverseCumulative(n, Fxi)) for Fxi in Fx]
plot(x,fx)
plot(x,Fx)
plot(x,invFp,"o")

end # module
