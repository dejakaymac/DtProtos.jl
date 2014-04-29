module ice

# package code goes here
using PyPlot
using Logging
Logging.configure(level=DEBUG)
#include("pdf.jl")
using DtProtos.pdfs


export normalise!, area, A, fromPdf, fromPdfControlPoints


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

function fromPdf(dist::SimplePdf)
    println("From SimplePdf")
end

function fromPdf(dist::GaussianPdf)
    l = cdf(dist, 0.25)
    r = cdf(dist, 0.75)
    x=0
end

function fromPdf(dist::BoundedGaussianPdf)
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


#function fromPdfControlPoints(dist::GaussianPdf, 
function fromPdfControlPoints(dist::SimplePdf,
#function fromPdfControlPoints(dist::Pdf, 
                              controlPoints::Array{Float64,1},
                              leftTail::Bool,
                              rightTail::Bool)
    logarithmOfDensity = Float64[]
    curvatures = Float64[]
    lod = log(pdf(dist, controlPoints[1]));
    if lod < -20
        lod = -20;
    end
    push!(logarithmOfDensity, lod)
    
    for i = [2:1:length(controlPoints)]
        lod = log(pdf(dist, controlPoints[i]));
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
		    cdf(dist, controlPoints[i])
		    -cdf(dist, controlPoints[i-1])
		    )
	      )
    end
    hasLeftTail = curvatures[1] > 0;
    warn("0 hasLeftTail $hasLeftTail")
    if (hasLeftTail) 
        alpha=(controlPoints[2]-controlPoints[1])/2;
        beta=(controlPoints[1]+controlPoints[2])/2;
        b=(logarithmOfDensity[2]-logarithmOfDensity[1])/2;
        hasLeftTail = (controlPoints[1] <= b/2/curvatures[1]*alpha+beta);
        warn("1 hasLeftTail $hasLeftTail")
    end
    if (hasLeftTail)
        hasLeftTail = leftTail;
        warn("2 hasLeftTail $hasLeftTail")
    end
    
    hasRightTail = curvatures[length(curvatures)] > 0;
    warn("0 hasRightTail $hasRightTail")
    if (hasRightTail) 
        k = length(curvatures)
        alpha = (controlPoints[k+1]-controlPoints[k])/2;
        beta = (controlPoints[k]+controlPoints[k+1])/2;
        b = (logarithmOfDensity[k+1]-logarithmOfDensity[k])/2;
        hasRightTail = (controlPoints[k+1] >= b/2/curvatures[k]*alpha+beta);
        warn("1 hasRightTail $hasRightTail")
    end
    if (hasRightTail)
        hasRightTail = rightTail;
        warn("2 hasRightTail $hasRightTail")
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



end # module
