# TODO:
#  4 - compute skew from x-y correlation

using Distributions
using ImageView
using PyPlot
using DataFrames

function load_image_data(filename::String)
    # returns a Matrix{Float64}
    readdlm(filename, ',', Float64) :: Matrix{Float64}
end
function set_to_zero_one_range(image::Matrix{Float64})
    hi,lo = extrema(image)
    (image .- lo) ./ (hi-lo)
end
function normalize_image(image::Matrix{Float64})
    image2 = image + minimum(image)
    image2 ./= sum(image2)
    for i = 1 : length(image2)
        image2 = max(image2, 0.0)
    end
    image2
end

function display_image(image::Matrix{Float64})
    img = set_to_zero_one_range(image)
    ImageView.view(1-img) 
end
function display_image(image::Matrix{Float64}, overlay_pts::Vector{(Int,Int)};
    overlay_color = [255,0,255]/255.0
    )

    img = 1 - set_to_zero_one_range(image)

    n,m = size(img)

    grayimg = Array(Float64, n, m, 3)
    for i = 1 : n
        for j = 1 : m
            val = img[i,j]
            grayimg[i,j,1] = val
            grayimg[i,j,2] = val
            grayimg[i,j,3] = val
        end
    end

    for (i,j) in overlay_pts
        grayimg[i,j,:] = overlay_color
    end

    ImageView.view(grayimg)
end

function cost_function(image::Matrix{Float64}, G::MvNormal)
    # J(G) = ∑_(i,j) (P(x,y) - P(x,y))^2

    dat = set_to_zero_one_range(image)

    retval = 0.0
    for i = 1 : size(image,1)
        for j = 1 : size(image,2)
            P  = dat[i,j]
            P2 = pdf(G, Float64[i, j])
            retval += (P-P2)*(P-P2)
        end
    end

    retval
end

function fit_gaussian(image::Matrix{Float64})

    n,m = size(image)

    # brightest pixel
    brightest = -Inf
    i_best, j_best = 0, 0
    for i = 1 : n, j = 1 : m
        if image[i,j] > brightest
            brightest = image[i,j]
            i_best, j_best = i, j
        end
    end

    # println("brightest: ", i_best, "  ", j_best)

    # location of the gaussian at start
    x = float64(i_best)
    y = float64(j_best)

    # rectangle bounds
    x_lo, x_hi = -0.5, n+0.5
    y_lo, y_hi = -0.5, m+0.5

    # init gaussian params
    G = MvNormal([x,y],[1.0 0.0;0.0 1.0])

    # println("initial cost: ", cost_function(image, G))
end

function pts_above_threshold(image::Matrix{Float64}, percent_max::Float64=0.5)
    # given the image, computes collection of pixels that have value at least
    # that of half the max value

    # returns a list of tuples (Int,Int)[] corresponding to their indeces

    n = length(image)
    n_pixels = 0
    threshold = percent_max*maximum(image)
    for p in image
        if p > threshold
            n_pixels += 1
        end
    end 

    retval = Array((Int,Int), n_pixels)
    count = 0
    for x = 1 : size(image,1)
        for y = 1 : size(image,2)
            if image[x,y] > threshold
                retval[count+=1] = (x,y)
            end
        end
    end
    @assert(count == length(retval))
    retval
end

function calc_full_width_half_maximum(image::Matrix{Float64}, percent_max::Float64=0.5)
    # given the image, computes collection of pixels that have value at least
    # that of half the max value

    # computes a flood fill on the resulting pixels to find the 
    # largest continuous region

    # returns a list of tuples (Int,Int)[] corresponding to their indeces

    pts = pts_above_threshold(image, percent_max)

    # since the number of pixels is assumed low (~28), just do a flood fill directly
    # on the tuples

    # algorithm
    #  - while not all of the pts have been filled
    #      - create stack for new group
    #      - assign random pt to group and place on stack
    #      - while stack is not empty
    #               - add any unfilled neighbors (N/E/S/W) to the stack
    #               - assign to them group

    groups = Vector{Int}[]
    unfilled = [1:length(pts)]
    while !isempty(unfilled)
        group = Int[]
        stack = Int[]
        ind = rand(1:length(unfilled))
        ptind = unfilled[ind]
        deleteat!(unfilled, ind)

        push!(group, ptind)
        push!(stack, ptind)

        while !isempty(stack)
            ptind = pop!(stack)
            x,y = pts[ptind]

            ind = 0
            while ind < length(unfilled)
                ind += 1
                ptind = unfilled[ind]
                x2,y2 = pts[ptind]
                if abs(x-x2) ≤ 1 && abs(y-y2) ≤ 1
                    deleteat!(unfilled, ind)
                    push!(group, ptind)
                    push!(stack, ptind)
                    ind -= 1
                end
            end
        end

        push!(groups, group)
    end

    # println("ngroups: ", length(groups))
    # for (i,g) in enumerate(groups)
    #     println("\t", i, ": ", length(g))
    # end

    best_group = 0
    best_size = 0
    for (i,g) in enumerate(groups)
        n = length(g)
        if n > best_size
            best_size, best_group = n, i
        end
    end

    @assert(best_size > 0)

    return pts[groups[best_group]]
end

function midpt(pts::Vector{(Int,Int)})

    sumx, sumy = 0, 0
    for (x,y) in pts
        sumx += x
        sumy += y
    end

    n = length(pts)
    (sumx/n, sumy/n)
end

function calc_xy_correlation(image::Matrix{Float64}, center_x::Int, center_y::Int, desired_radius::Int=10)

    # compute the correlation between the subsection of the image given by the square area 
    # within radius of center
    # note that we weight on the pixel values

    n,m = size(image)
    rad = minimum([center_x-1,center_y-1,n-center_x-1, m-center_y-1, desired_radius])
    W   = set_to_zero_one_range(image)
    if rad < 0
        return NaN
    end

    sum_xy = 0.0
    sum_x  = 0.0
    sum_y  = 0.0


    for i = -rad : rad
        x = i + center_x
        for j = -rad : rad
            y = j + center_y
            w = W[x,y]
            sum_xy += w*i*j
            sum_x  += w*i*i
            sum_y  += w*j*j
        end
    end

    # println(sum_xy, " ", sum_x, " " , sum_y)

    sum_xy / sqrt(sum_x * sum_y)
end

function blob_subimage_boolean(pts::Vector{(Int,Int)})
    x_lo, x_hi = 10000, -10000
    y_lo, y_hi = 10000, -10000
    for (x,y) in pts
        x_lo = min(x, x_lo)
        x_hi = max(x, x_hi)
        y_lo = min(y, y_lo)
        y_hi = max(y, y_hi)
    end

    # note: adding a 1 pixel border for safety
    w = x_hi - x_lo + 3
    h = y_hi - y_lo + 3
    subimage = falses(w,h)
    for (x,y) in pts
        subimage[x-x_lo+2, y-y_lo+2] = true
    end

    subimage
end
function blob_subimage_raw(image::Matrix{Float64}, pts::Vector{(Int,Int)}, image_side_length::Int=24)

    W, H = size(image)

    center_x, center_y = midpt(pts)
    rad = minimum([center_x-1,center_y-1,W-center_x-1, H-center_y-1])

    if rad < image_side_length/2
        return Array(Float64, 1,1)
    end
    rad = int(image_side_length/2)

    x_lo = max(1,int(center_x-rad))
    y_lo = max(1,int(center_y-rad))

    image[x_lo:x_lo+image_side_length-1,
          y_lo:y_lo+image_side_length-1]
end
function blob_perimeter(subimage::BitMatrix)
    # perimeter is estimated as the number of pts that are not fully encircled
    # NOTE(tim): this does not attempt to handle blobs with missing internal pixels

    w,h = size(subimage)

    perimeter = 0
    for i = 2 : w-1
        for j = 2 : h-1
            if subimage[i,j]
                for (dx,dy) in ((1,1),(1,0),(1,-1),(0,1),(0,-1),(-1,1),(-1,0),(-1,-1))
                    if !subimage[i+dx, j+dy]
                        perimeter += 1
                        break
                    end
                end
            end
        end
    end

    perimeter
end
blob_area(subimage::BitMatrix) = sum(subimage)
function blob_roundness(subimage::BitMatrix)
    # roundness is estimated as 4π area / perimeter²

    perimeter = blob_perimeter(subimage) # perimeter in pixels
    area      = blob_area(subimage) # area in pixels

    4.0*π*area / (perimeter*perimeter)
end
function blob_entropy(subimage::Matrix{Float64})
    # entropy = -∑ P(x) log P(x)
    # NOTE(tim): assumes subimage was normalized

    retval = 0.0
    for v in subimage
        @assert(v ≥ 0.0)
        retval -= v * log(v)
    end
    retval
end
function blob_contrast(subimage::Matrix{Float64})
    
    m,n = size(subimage)

    z = 0.25
    μ = mean(subimage)
    σ = varm(subimage, μ)
    μ₄ = 0.0
    for i = 1 : m
        for j = 1 : n
            μ₄ += (subimage[i,j]-μ)^4
        end
    end
    μ₄ /= (n*m)

    α₄ = μ₄ / σ^4
    contrast = σ / α₄^z
end
function blob_directionality_histogram(subimage::Matrix)

    conv_H = [-1 0 1;
              -1 0 1;
              -1 0 1]
    conv_V = [-1 -1 -1;
               0  0  0;
               1  1  1]

    n_bins = 16

    histobins = ones(Int,n_bins) # for angles 0 to pi (adding uniform Laplace prior counts)

    m,n = size(subimage)
    for i = 2 : m-1
        for j = 2 : n-1
            Δh = 0.0
            Δv = 0.0
            
            for di = -1:1
                for dj = -1:1
                    Δh += subimage[i+di,j+dj]*conv_H[di+2,dj+2]
                    Δv += subimage[i+di,j+dj]*conv_V[di+2,dj+2]
                end
            end

            if isapprox(Δh, 0.0)
                θ = π/2
            else
                θ = π/2 + atan(Δv / Δh) # ∈ [0,π]
            end

            
            # println(θ)
            bin = int(θ * (n_bins-1) / π) + 1
            histobins[bin] += 1
        end
    end

    histobins ./ sum(histobins)
end

function dataset_index_matching_criteria(df::DataFrame)

    for i = size(df,1) : -1 : 1
        size = df[i, :size]
        # println(size)
        if size < 30
            return i
        end
    end
    return 0
end
function extract_dataset(df::DataFrame, i::Int)

    dirnumber = df[i, :directory]
    objnumber = df[i, :objectnumber]

    dirname  = "/media/tim/USB DISK/Extracted/Obj_" * string(dirnumber) * "/"
    filename = @sprintf("Obj_%d_1 - IC filter %d.csv", objnumber, objnumber)

    image = load_image_data(dirname * filename)

    pts = calc_full_width_half_maximum(image)

    display_image(image, pts)

    mid = midpt(pts)
    cor = calc_xy_correlation(image, int(mid[1]), int(mid[2]), 10)

    # println("skew: ", cor)
end

function process_data(dir::String;
    correlation_radius :: Int = 10
    )

    files = readdir(dir)
    nfiles = length(files)

    df = DataFrame()
    df[:objectnumber]           = Array(Int,     nfiles)
    df[:area]                   = Array(Int,     nfiles)
    df[:perimeter]              = Array(Int,     nfiles)
    df[:skew]                   = Array(Float64, nfiles)
    df[:peak]                   = Array(Float64, nfiles)
    df[:roundness]              = Array(Float64, nfiles)
    # df[:entropy]                = Array(Float64, nfiles)
    df[:contrast]               = Array(Float64, nfiles)
    df[:blob_mean]              = Array(Float64, nfiles)
    df[:blob_stdev]             = Array(Float64, nfiles)
    df[:subim_mean]             = Array(Float64, nfiles)
    df[:subim_stdev]            = Array(Float64, nfiles)
    df[:directionality_primary] = Array(Float64, nfiles)   # location of strongest directionality
    df[:directionality_entropy] = Array(Float64, nfiles) # entropy of directionality

    count = 0
    for (i,file) in enumerate(files)
        if ismatch(r"\d+", file)
            count += 1

            objectnumber = int(match(r"\d+", file).match)
            image = load_image_data(dir * file)

            pts = calc_full_width_half_maximum(image)
            mid = midpt(pts)
            pt_vals = Array(Float64, length(pts))
            for (i,pt) in enumerate(pts)
                pt_vals[i] = image[pt[1], pt[2]]
            end

            df[count,:objectnumber] = objectnumber
            df[count,:area]         = length(pts)
            df[count,:skew]         = calc_xy_correlation(image, int(mid[1]), int(mid[2]), correlation_radius)
            df[count,:blob_mean]    = mean(pt_vals)
            df[count,:blob_stdev]   = stdm(pt_vals,df[count,:blob_mean])

            subim_bool = blob_subimage_boolean(pts)

            df[count,:perimeter]    = blob_perimeter(subim_bool)
            df[count,:roundness]    = blob_roundness(subim_bool)

            subim = blob_subimage_raw(image, pts)

            if length(subim) != 1
                nsubim = normalize_image(subim)
                hist = blob_directionality_histogram(nsubim)

                df[count,:peak]                   = maximum(subim)
                df[count,:subim_mean]             = mean(subim)
                df[count,:subim_stdev]            = stdm(subim, df[count,:subim_mean])
                # df[count,:entropy]                = blob_entropy(nsubim)
                df[count,:contrast]               = blob_contrast(nsubim)
                df[count,:directionality_primary] = indmax(hist)
                df[count,:directionality_entropy] = -sum(hist .* log(hist))
            else
                df[count,:peak]                   = NaN
                df[count,:subim_mean]             = NaN
                df[count,:subim_stdev]            = NaN
                # df[count,:entropy]                = NaN
                df[count,:contrast]               = NaN
                df[count,:directionality_primary] = NaN
                df[count,:directionality_entropy] = NaN
            end

        end
    end

    df = df[1:count, :]

    writetable(dir*"features.csv", df)
end
function process_all_data()
    for i in [1:16]
        dirname = "/media/tim/Tim 1500 GB/extracted/Obj_" * string(i) * "/"
        println("processing ", dirname)
        process_data(dirname)
    end
end

function pull_aggregate_stats()

    file_numbers = [1:16]

    n_values = 0
    for i in file_numbers
        dirname = "/media/tim/Tim 1500 GB/extracted/Obj_" * string(i) * "/"
        df = readtable(dirname*"features.csv")
        n_values += size(df, 1)
    end

    data = DataFrame()
    data[:directory]       = Array(Int, n_values)
    data[:objectnumber]    = Array(Int, n_values)
    data[:area]            = Array(Int, n_values)
    data[:perimeter]       = Array(Int, n_values)
    data[:skew]            = Array(Float64, n_values)
    data[:peak]            = Array(Float64, n_values)
    data[:roundness]       = Array(Float64, n_values)
    data[:contrast]        = Array(Float64, n_values)
    data[:blob_mean]       = Array(Float64, n_values)
    data[:blob_stdev]      = Array(Float64, n_values)
    data[:subim_mean]      = Array(Float64, n_values)
    data[:subim_stdev]     = Array(Float64, n_values)
    data[:directionality_primary] = Array(Float64, n_values)
    data[:directionality_entropy] = Array(Float64, n_values)

    count = 0
    for i in file_numbers

        dirname = "/media/tim/Tim 1500 GB/extracted/Obj_" * string(i) * "/"
        df = readtable(dirname*"features.csv")
        println(names(df))
        data[count+1:count+size(df,1),2:end] = df
        data[count+1:count+size(df,1),:directory] = i
        count += size(df, 1)
    end

    writetable("/media/tim/Tim 1500 GB/extracted/all_image_features.csv", data)
    data
end
function plot_histograms(df::DataFrame)

    fig = PyPlot.figure(facecolor="white")

    ax1 = fig[:add_subplot](211)
    ax1[:hist](df[:size], 100, color=[0.0, 0.5, 0.5])
    ax1[:set_xlabel]("neuron size, FWHM [pixels]")
    ax1[:set_ylabel]("counts")

    arr = dropna(df[:skew])
    arr = filter!(x->!isinf(x) && !isnan(x), arr)
    ax2 = fig[:add_subplot](212)
    ax2[:hist](arr, 100, color=[0.5, 0.0, 0.5])
    ax2[:set_xlabel]("skew")
    ax2[:set_ylabel]("counts")
end