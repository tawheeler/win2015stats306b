# TODO:
#  4 - compute skew from x-y correlation

using Distributions
using ImageView
using PyPlot
using DataFrames

const INPUT_FOLDER = "/media/tim/USB DISK/Extracted/Obj_7/"

function load_image_data(filename::String)
    # returns a Matrix{Float64}
    readdlm(filename, ',', Float64) :: Matrix{Float64}
end
function normalize_image(image::Matrix{Float64})
    hi,lo = extrema(image)
    (image .- lo) ./ (hi-lo)
end

function display_image(image::Matrix{Float64})
    img = normalize_image(image)
    view(1-img) 
end
function display_image(image::Matrix{Float64}, overlay_pts::Vector{(Int,Int)};
    overlay_color = [255,0,255]/255.0
    )

    img = 1 - normalize_image(image)

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

    dat = normalize_image(image)

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

    println("brightest: ", i_best, "  ", j_best)

    # location of the gaussian at start
    x = float64(i_best)
    y = float64(j_best)

    # rectangle bounds
    x_lo, x_hi = -0.5, n+0.5
    y_lo, y_hi = -0.5, m+0.5

    # init gaussian params
    G = MvNormal([x,y],[1.0 0.0;0.0 1.0])

    println("initial cost: ", cost_function(image, G))
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

    println("ngroups: ", length(groups))
    for (i,g) in enumerate(groups)
        println("\t", i, ": ", length(g))
    end

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
    W   = normalize_image(image)
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

    println(sum_xy, " ", sum_x, " " , sum_y)

    sum_xy / sqrt(sum_x * sum_y)
end

function process_data(dir::String=INPUT_FOLDER;
    correlation_radius :: Int = 10
    )

    files = readdir(dir)
    nfiles = length(files)

    df = DataFrame(objectnumber=Array(Int, nfiles), size=Array(Int, nfiles), skew=Array(Float64, nfiles))

    count = 0
    for (i,file) in enumerate(files)
        if ismatch(r"\d+", file)
            count += 1

            objectnumber = int(match(r"\d+", file).match)
            image = load_image_data(dir * file)

            pts = calc_full_width_half_maximum(image)
            mid = midpt(pts)

            df[count,:objectnumber] = objectnumber
            df[count,:size] = length(pts)
            df[count,:skew] = calc_xy_correlation(image, int(mid[1]), int(mid[2]), correlation_radius)
        end
    end

    df = df[1:count, :]

    writetable(dir*"features.csv", df)
end

# for i in [1,2,3,4,5,6,8]
#     dirname = "/media/tim/USB DISK/Extracted/Obj_" * string(i) * "/"
#     println("processing ", dirname)
#     process_data(dirname)
# end

function pull_aggregate_stats()

    file_numbers = [1,2,3,4,5,6,7,8]

    n_values = 0
    for i in file_numbers
        dirname = "/media/tim/USB DISK/Extracted/Obj_" * string(i) * "/"
        df = readtable(dirname*"features.csv")
        n_values += size(df, 1)
    end

    data = DataFrame(directory=Array(Int, n_values), objectnumber=Array(Int, n_values), size=Array(Int, n_values), skew=Array(Float64, n_values))
    count = 0
    for i in file_numbers
        dirname = "/media/tim/USB DISK/Extracted/Obj_" * string(i) * "/"
        df = readtable(dirname*"features.csv")
        data[count+1:count+size(df,1),2:end] = df
        data[count+1:count+size(df,1),:directory] = i
        count += size(df, 1)
    end

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

function dataset_index_matching_criteria(df::DataFrame)

    for i = size(df,1) : -1 : 1
        size = df[i, :size]
        println(size)
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

    println("skew: ", cor)
end

