module Ice

# package code goes here
using PyPlot
using Logging
using DtProtos.Pdfs

export fromPdf, fromPdfControlPoints, fromPdfScale, fromBoundedPdfScale, IcePdf, pdf

type IcePdf 
    controlPoints::Array{Float64,1}
    logarithmOfDensity::Array{Float64,1}
    curvatures::Array{Float64,1}
    hasLeftTail::Bool
    hasRightTail::Bool
end

function pdf(icepdf::IcePdf, z::Number)
    debug("== Ice.pdf(icepdf::IcePdf, x::Number)")
    #assertNonEmpty();
    i = segment(icepdf, z);
    if (i==-1)
        return 0;
    end

    localZ = (2 * z - 
              icepdf.controlPoints[i+1] - 
              icepdf.controlPoints[i] ) / 
              (icepdf.controlPoints[i+1] - 
               icepdf.controlPoints[i])
    a = (icepdf.logarithmOfDensity[i] + 
         icepdf.logarithmOfDensity[i+1]) / 2
    b = (icepdf.logarithmOfDensity[i+1] - 
         icepdf.logarithmOfDensity[i]) / 2
    return exp(a+b*localZ+icepdf.curvatures[i]*(1+localZ)*(1-localZ))
end


function quantile(icepdf::IcePdf, z::Number)
    # // For the moment, this uses bisection, for speed, replace with
    # // Brent's Method as is used in PDF.
    # assertNonEmpty();
    lower = icepdf.controlPoints[1]
    upper = icepdf.controlPoints[end]
    while (cdf(icepdf, lower) > z) 
        lower = 2*lower-upper
    end
    while (cdf(icepdf, upper) < z)
        upper = 2*upper-lower
    end
    for i = [1:32-1] #for (int i=0;i<32;++i) {
        if (cdf(icepdf, (lower+upper)/2.) > z)
            upper = (lower+upper)/2.
        else
            lower = (lower+upper)/2.
        end
    end
    return (lower+upper)/2.
end
    
function segment(icepdf::IcePdf, z::Number)
    debug("== Ice.segment(icepdf::IcePdf, z::Number)")
    #assertNonEmpty();
    nsegs = length(icepdf.curvatures)
    if (z < icepdf.controlPoints[1])
        if (icepdf.hasLeftTail) 
            return 1
        else
            return -1;
        end
    end
    for i = [1:nsegs] #for (int i=0;i<numberOfSegments;++i)
        if (z < icepdf.controlPoints[i+1]) 
            return i
        end
    end
    if (icepdf.hasRightTail) 
        #return (nsegs - 1)
        return nsegs
    else
        return -1
    end
end


function cdf(icepdf::IcePdf, z::Number)
    #assertNonEmpty();
    debug("== Ice.cdf(icepdf::IcePdf, z::Number)")
    i = segment(icepdf, z);
    debug("= segment done")
    if (i == -1) 
        if (z < icepdf.controlPoints[1]) 
            return 0.0
        else
            return 1.0
        end
    end
    r = 0.0
    for j = [1:i-1] # for (int j=0;j<i;++j)
        alpha = (icepdf.controlPoints[j+1] - 
                 icepdf.controlPoints[j]) / 2
        r += A(icepdf, j) * alpha
    end
    localZ = (2 * z - 
              icepdf.controlPoints[i+1] - 
              icepdf.controlPoints[i] ) / 
              (icepdf.controlPoints[i+1] - 
               icepdf.controlPoints[i])
    alpha = (icepdf.controlPoints[i+1] - 
             icepdf.controlPoints[i]) / 2
    a = (icepdf.logarithmOfDensity[i] + 
         icepdf.logarithmOfDensity[i+1]) / 2
    b = (icepdf.logarithmOfDensity[i+1] - 
         icepdf.logarithmOfDensity[i]) / 2
    if (i == 1 && icepdf.hasLeftTail)
        ltail = (I(localZ, a, b, icepdf.curvatures[i]) - 
                 I(-Inf,   a, b, icepdf.curvatures[i])) * alpha 
        return ltail
    end
    r += (I(localZ, a, b, icepdf.curvatures[i]) - 
          I(-1,     a, b, icepdf.curvatures[i])) * alpha
    return r
end


function normalise!(icepdf::IcePdf)
    debug("== Ice.normalise!(icepdf::IcePdf)")
    # #assertNonEmpty();
    la = log(area(icepdf))
    for i = [1:length(icepdf.logarithmOfDensity)] #(unsigned i=0;i<myLogarithmOfDensity.size();++i)
        icepdf.logarithmOfDensity[i] -= la;
    end
    return icepdf
end

function area(icepdf::IcePdf)
    debug("== Ice.area(icepdf::IcePdf)")
    #assertNonEmpty();
    r = 0.0;
    #for j = [1:1:length(icepdf.curvatures)] # (unsigned j=0;j<myCurvatures.size();++j)
    for j = [1:length(icepdf.curvatures)] # (unsigned j=0;j<myCurvatures.size();++j)
        alpha = (icepdf.controlPoints[j+1] - icepdf.controlPoints[j])/2;
        r += A(icepdf, j)*alpha;
    end
    # if sign(r) == -1
    #     #error("Negative area: ", r)
    #     if abs(r) < 1e-8
    #         r = 1e-20
    #     else
    #         error("Negative area: ", r)
    #     end
    # end  
    return r;
end

function A(icepdf::IcePdf, i::Number)
    debug("== Ice.A(icepdf::IcePdf, i::Number)")
    # assertNonEmpty();
    a = (icepdf.logarithmOfDensity[i] + icepdf.logarithmOfDensity[i+1])/2;
    b = (icepdf.logarithmOfDensity[i+1] - icepdf.logarithmOfDensity[i])/2;
    #if (i==0 && i==(length(icepdf.curvatures)) && icepdf.hasLeftTail && icepdf.hasRightTail)
    if (i==1 && i==(length(icepdf.curvatures)) && 
        icepdf.hasLeftTail && icepdf.hasRightTail)
        return I(Inf,a,b,icepdf.curvatures[i])-I(-Inf,a,b,icepdf.curvatures[i]);
    end
    if (i==1 && icepdf.hasLeftTail)
        return I(1,a,b,icepdf.curvatures[i])-I(-Inf,a,b,icepdf.curvatures[i]);
    end
    if (i==length(icepdf.curvatures) && icepdf.hasRightTail)
        return I(Inf,a,b,icepdf.curvatures[i])-I(-1,a,b,icepdf.curvatures[i]);
    end
    return I(1,a,b,icepdf.curvatures[i])-I(-1,a,b,icepdf.curvatures[i]);
end

function C(b::Number, c::Number)
    debug("== Ice.C(b::Number, c::Number)")
    if (abs(b*b/c)>30)
        return 0.0;
    end
    return c;
end

function I(z::Number,
           a::Number,
           b::Number,
           c::Number)
    debug("== Ice.I(z::Number, a::Number, b::Number, c::Number")
    if (c==0)
        if (b==0)
            @assert(isfinite(z), "z infinite, c=b=0");
            return (z*exp(a));
        end
        @assert(sign(z)*isinf(z)*b <= 0, "z infinite, z*b>0");
        return (exp(a+b*z)/b);
    end

    if (isinf(z))
        @assert(c > 0, "z infinite, c<=0");
        #return exp(b*b/4/c+a+c)*M_SQRTPI*isInfinite(z)/2/sqrt(c);
        #return exp(b*b/4/c+a+c)*sqrt(pi)*isinf(z)/2/sqrt(c);
        return exp(b*b/4/c+a+c)*sqrt(pi)*sign(z)*isinf(z)/2/sqrt(c);
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
    debug("== Ice.findC(x1::Float64, y1::Float64,",
          "x2::Float64, y2::Float64, A::Float64)")

    a = (y1+y2)/2;
    b = (y2-y1)/2;
    l = -1;
    r = 1;
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
#function fromPdfControlPoints(dist::SimplePdf,
function fromPdfControlPoints(dist::Pdf, 
                              controlPoints::Array{Float64,1},
                              leftTail::Bool,
                              rightTail::Bool)
    debug("== Ice.fromControlPoints",
          "(dist::Pdf, controlPoints::Array{Float64,1},",
          "leftTail::Bool,rightTail::Bool")
    logarithmOfDensity = Float64[]
    curvatures = Float64[]
    lod = log(pdfs.pdf(dist, controlPoints[1]));
    if lod < -20
        lod = -20;
    end
    push!(logarithmOfDensity, lod)
    
    for i = [2:1:length(controlPoints)]
        lod = log(pdfs.pdf(dist, controlPoints[i]));
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
		    pdfs.cdf(dist, controlPoints[i])
		    -pdfs.cdf(dist, controlPoints[i-1])
		    )
	      )
    end
    hasLeftTail = curvatures[1] > 0;
    debug("0 hasLeftTail $hasLeftTail")
    if (hasLeftTail) 
        alpha=(controlPoints[2]-controlPoints[1])/2;
        beta=(controlPoints[1]+controlPoints[2])/2;
        b=(logarithmOfDensity[2]-logarithmOfDensity[1])/2;
        hasLeftTail = (controlPoints[1] <= b/2/curvatures[1]*alpha+beta);
        debug("1 hasLeftTail $hasLeftTail")
    end
    if (hasLeftTail)
        hasLeftTail = leftTail;
        debug("2 hasLeftTail $hasLeftTail")
    end
    
    hasRightTail = curvatures[length(curvatures)] > 0;
    debug("0 hasRightTail $hasRightTail")
    if (hasRightTail) 
        k = length(curvatures)
        alpha = (controlPoints[k+1]-controlPoints[k])/2;
        beta = (controlPoints[k]+controlPoints[k+1])/2;
        b = (logarithmOfDensity[k+1]-logarithmOfDensity[k])/2;
        hasRightTail = (controlPoints[k+1] >= b/2/curvatures[k]*alpha+beta);
        debug("1 hasRightTail $hasRightTail")
    end
    if (hasRightTail)
        hasRightTail = rightTail;
        debug("2 hasRightTail $hasRightTail")
    end
    
    debug("left, right       : $(hasLeftTail), $(hasRightTail)")
    r = IcePdf(controlPoints,
               logarithmOfDensity,
               curvatures,
               hasLeftTail,
               hasRightTail);
    r = normalise!(r);
    debug("left, right       : $(hasLeftTail), $(hasRightTail)")
    return r;
end




# does fromPDFControlPoints use InverseCumulative() much?
# if so this is bad

function fromPdf(dist::SimplePdf)
    debug("== Ice.fromPdf(dist::SimplePdf)")
    println("From SimplePdf")
end

function fromPdf(dist::GaussianPdf)
    debug("== Ice.fromPdf(dist::GaussianPdf)")
    l = cdf(dist, 0.25)
    r = cdf(dist, 0.75)
    x=0
end

function fromPdf(dist::BoundedGaussianPdf)
    println("Not implemented")
    #raise/throw error
end


function fromPdfScale(dist::Pdf,
                      maxcontrolpoints::Number,
                      targeterror::Float64)
    x = Float64[]
    x0 = quantile(dist, 0.25)
    x1 = quantile(dist, 0.5)
    x1b = quantile(dist, 0.6)
    x2 = quantile(dist, 0.75)
    #push!(x, x0, x1, x2)
    #push!(x, x0, x1,x1b, x2)
    push!(x, x0, x2)
    #x = [float64(quantile(dist, i)) for i = [0.05:0.05:0.95] ]
    scale = x2 - x0
    #for cp = [length(x):]
    #cp = length(x) 
    cp = 3 #length(x) #+ 1
    z = 0.0
    step = 0.0
    l = 0.0
    r = 0.0
    while cp < maxcontrolpoints #   for (int controlPoints=2; controlPoints < maxControlPoints; ++controlPoints)
        info("cp, x : $cp, $x")
        sort!(x)
        ipdf = fromPdfControlPoints(dist,
                                     x,
                                     true,
                                     true)
        maxdelta = 0.0
        bestz = 0.0 
        for k = [1:cp]  #for k = [1:cp]  #for (int k=0; k < controlPoints+1 ; ++k)
            debug("k $k")
            if (k == 1)
                debug ("k == 1")
                l = quantile(dist, targeterror)
            else 
                debug (" l = x[k-1]")
                l = x[k-1]
            end
            if (k == cp)
                debug ("k == cp")
                r = quantile(dist, 1-targeterror)
            else
                debug("l = x[k]")
                r = x[k]
            end
            step = scale / 50
            if ((r-l)/4<step)
                step = ((r-l)/4)
            end
            z = l + step
            debug("z = l + step :  $z = $l + $step")
            debug("r-z = $(r-z)")
           
            while (r-z > step / 2.0) #for (z=l+step; r-z > step/2; z+=step)
                #z += step
                debug("r-z $(r-z)")
                delta = abs((pdfs.cdf(dist, z) - pdfs.cdf(dist, l)) -
                            (cdf(ipdf, z) - cdf(ipdf, l)))
                if ((delta > maxdelta) &&
                    (pdfs.pdf(dist, z) > ((5e-5)/scale)))
                    bestz = z;
                    maxdelta = delta;
                end
                z += step
            end # while r-z

            delta = abs((pdfs.cdf(dist, r) - pdfs.cdf(dist, l)) -
                        (cdf(ipdf, r) - cdf(ipdf, l)))
            if ((delta > maxdelta) && 
                (pdfs.pdf(dist, (l+r)/2) > ((5e-5)/scale)) )
                bestz = (l+r)/2
                maxdelta = delta
            end
        end # for k
        if (maxdelta < targeterror)
            debug("breaking")
            break
        end
        debug("bestz = $bestz")
        if typeof(bestz) == Int64
            error("bad type")
        end
        push!(x, bestz);
        cp += 1
    end  # while cp
    sort!(x)
    return fromPdfControlPoints(dist,x,true,true);
end

function fromBoundedPdfScale(dist::BoundedGaussianPdf)
    error("Not implemented")
end



end # module

      
