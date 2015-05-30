using Distributions
using ImageView
using PyPlot
using DataFrames

function get_data_matrix(df::DataFrame)
    m,n = size(df)

    X = Array(Float64, n, m)
    colnames = names(df)
    colind = 0
    for i = 1 : n
        if colnames[i] != :directory &&
           colnames[i] != :objectnumber &&
           colnames[i] != :isCell &&
           colnames[i] != :cluster_assignment_full &&
           colnames[i] != :cluster_assignment_binary

            colind += 1
            X[colind,:] = array(df[:,i])
        end
    end

    X[1:colind,:]
end
function get_data_names(df::DataFrame)
    colnames = names(df)
    keep = falses(length(colnames))
    colind = 0
    for i = 1 : length(colnames)
        if colnames[i] != :directory &&
           colnames[i] != :objectnumber &&
           colnames[i] != :isCell &&
           colnames[i] != :cluster_assignment_full

            keep[i] = true
        end
    end
    colnames[keep]
end
function whiten(X::Matrix{Float64})

    # de-mean
    Y = X - repmat(mean(X, 2), 1, size(X,2))

    # unit covariance
    for i = 1 : size(Y,1)
        σ = std(Y[i,:])
        Y[i,:] ./= σ
    end

    Y
end

const CLUSTER_ID_NAMES = ["A", "B", "C", "high intensity", "D", "low intensity"]
const COLOR_A = [0.0,1.0,0.0,0.4]
const COLOR_B = [1.0,0.0,0.0,0.4]
const COLOR_C = [0.2,0.7,0.6,0.4]
const COLOR_D = [0.7,0.2,0.6,0.4]

const TP = 1
const FP = 2
const TN = 3
const FN = 4

get_cluster_ids(df::DataFrame) = sort(unique(array(df[:cluster_assignment_full])))
get_cluster_inds(df::DataFrame, id::Int) = find(x->x==id, array(df[:cluster_assignment_full]))
get_cluster_size(df::DataFrame, id::Int) = length(get_cluster_inds(df, id))
cluster_id_to_name(id::Int) = CLUSTER_ID_NAMES[id]
function get_cluster_percent_neurons(df::DataFrame, id::Int)
    clusterinds = get_cluster_inds(df, id)
    size = length(clusterinds)
    n_neurons = sum(array(df[:isCell])[clusterinds])
    n_neurons / size * 100.0
end
function get_cluster_centroids(df::DataFrame, X::Matrix{Float64})

    # returns a matrix where the columns are the centroids
    # each row corresponds to a particular feature

    assignment = array(df[:cluster_assignment_full])
    ids = get_cluster_ids(df)

    nclust = length(ids)
    centroids = zeros(Float64, size(X, 1), length(ids))
    for i = 1 : nclust
        count = 0
        for j = 1 : size(X, 2)
            if assignment[j] == i
                count += 1
                centroids[:, i] += X[:, j]
            end
        end
        centroids[:, i] ./= count
    end

    centroids
end
function get_cluster_centroids_binary(df::DataFrame, X::Matrix{Float64})

    assignment = array(df[:cluster_assignment_binary])
    ids = sort(unique(array(df[:cluster_assignment_binary])))

    nclust = length(ids)
    centroids = zeros(Float64, size(X, 1), nclust)
    for i = 1 : nclust
        count = 0
        for j = 1 : size(X, 2)
            if assignment[j] == i+6
                count += 1
                centroids[:, i] += X[:, j]
            end
        end
        centroids[:, i] ./= count
    end

    centroids
end
function get_confusion_array(df::DataFrame)
    iscell = map(f->bool(f), array(df[:isCell]))
    isneuroncluster = map(1:size(df,1)) do i
        id = df[i, :cluster_assignment_full]
        name = cluster_id_to_name(id)
        name == "A" || name == "B" || name == "C" || name == "D"
    end

    confusion = map(1:size(df,1)) do i
        if iscell[i] && isneuroncluster[i]
            return TP
        elseif iscell[i]
            return FP
        elseif isneuroncluster[i]
            return FN
        else
            return TN
        end
    end
    confusion
end
function print_cluster_stats_table(df::DataFrame)
    # compute cluster size and neuron percent for each cluster

    @printf("%20s %10s %20s\n", "cluster", "size", "percent_neurons") 
    println("-"^52)

    for id in get_cluster_ids(df)
        cluster_size = get_cluster_size(df, id)
        percent_neurons = get_cluster_percent_neurons(df, id)
        @printf("%20s %10d %20f\n", cluster_id_to_name(id), cluster_size, percent_neurons)
    end
end

function calc_nearest_neighbors(X::Matrix{Float64})
    m = size(X, 2)
    nearest_neighbors = Array(Int, m)
    for i = 1 : m

        value = X[:, i]

        min_dist = Inf
        nearest_neighbors[i] = -1
        for j = 1 : m
            if j != i
                target = X[:,j]
                dist = norm(value - target)
                if dist < min_dist
                    min_dist = dist
                    nearest_neighbors[i] = j
                end
            end
        end
    end

    nearest_neighbors
end
function calc_assignment_to_closest_centroid(centroids::Matrix{Float64}, X::Matrix{Float64})
    p, m = size(X)
    n_clust = size(centroids, 2)
    assignment_2_to_closest_centroid = Array(Int, m)
    for i = 1 : m

        value = X[:, i]

        min_dist = Inf
        for j = 1 : n_clust
            target = centroids[:,j]

            dist = 0.0
            for k = 1 : p
                Δ = value[k] - target[k]
                dist += Δ*Δ
            end

            if dist < min_dist
                min_dist = dist
                assignment_2_to_closest_centroid[i] = j
            end
        end
    end

    assignment_2_to_closest_centroid
end
function calc_counts(vec::Vector{Int})

    counts = Dict{Int, Int}()
    for c in vec
        counts[c] = get(counts, c, 0) + 1
    end
    counts
end
function calc_in_group_proportions(
    centroids::Matrix{Float64}, # p×c
    X::Matrix{Float64},        # p×m
    nearest_neighbors :: Vector{Int}
    )
    # assign objects in set 2 to the closest centroids identified for set 1
    # in the "in-group-proportion" is the proportion of observations in the second group whose
    # nearest neighbor is also in that group

    
    assignment_2_to_closest_centroid = calc_assignment_to_closest_centroid(centroids, X)

    m = size(X, 2)
    c = size(centroids, 2)

    in_group_proportions = zeros(Int, c)
    counts = zeros(Int, c)
    for i = 1 : m
        neighbor = nearest_neighbors[i]

        # is it also in the same group?
        cind = assignment_2_to_closest_centroid[i]
        counts[cind] += 1
        if cind == assignment_2_to_closest_centroid[neighbor]
            in_group_proportions[cind] += 1
        end
    end

    in_group_proportions ./ counts

    # 0.9882692307692308
end
function calc_in_group_proportions_binary(
    centroids         :: Matrix{Float64},
    X2                :: Matrix{Float64},
    nearest_neighbors :: Vector{Int}
    )

    # assign objects in set 2 to the closest centroids identified for set 1
    # in the "in-group-proportion" is the proportion of observations in the second group whose
    # nearest neighbor is also in that group

    m2 = size(df2, 1)

    n_clust = size(centroids, 2)
    assignment_2_to_closest_centroid = Array(Int, m2)
    for i = 1 : m2

        value = X2[:, i]

        min_dist = Inf
        for j = 1 : n_clust
            target = centroids[:,j]
            dist = norm(value - target)
            if dist < min_dist
                min_dist = dist
                assignment_2_to_closest_centroid[i] = j
            end
        end
    end

    in_group_proportion = 0
    for i = 1 : m2

        neighbor = nearest_neighbors[i]

        # is it also in the same group?
        if assignment_2_to_closest_centroid[i] == assignment_2_to_closest_centroid[neighbor]
            in_group_proportion += 1
        end
    end

    in_group_proportion /= m2

    # 0.9978846153846154
end

function calc_in_group_proportions_pvalue(
    C   :: Matrix{Float64}, # p×c
    X   :: Matrix{Float64}, # p×m
    nearest_neighbors :: Vector{Int};
    nsamples :: Int = 1000
    )

    #=
    Compute the significance of the in-group-proportion by:
     - randomly selecting centroids in the feature space
     - compute the in-group-proportions using random centroids
     - p-value is the percentage of times that the in-group proportion
       from random centroids exceeds the observed in-group proportion

    C is the matrix of centroids with features in the rows
    Cₚ is C projected to the principle component orientation
    Permute within each row of Cₚ to get new centroids
    =#

    U,Σ,V = svd(C)
    Cp = C*V
    # Cp = deepcopy(C)
    p,c = size(C)

    in_group_proportions = calc_in_group_proportions(C, X, nearest_neighbors)

    counts_exceeded = zeros(Int, c)
    for i = 1 : nsamples

        # permute the columns
        for j = 1 : p
            Cp[j,:] = Cp[j,randperm(c)]
        end

        new_in_group_proportions = calc_in_group_proportions(Cp*V', X, nearest_neighbors)
        # new_in_group_proportions = calc_in_group_proportions(Cp, X, nearest_neighbors)
        for k = 1 : c
            if new_in_group_proportions[k] > in_group_proportions[k]
                counts_exceeded[k] += 1
            end
        end
    end

    counts_exceeded ./ (nsamples / 100)
end

function create_PCA_scatterplot(df::DataFrame, X::Matrix{Float64})

    # X is a p×m data matrix (p = #features, m = #samples)

    iscell = map(f->bool(f), array(df[:isCell]))

    U,Σ,V = svd(X)

    # data projected to subspace
    Z = U' * X

    inds_Neurons = iscell
    inds_NotNeur = !iscell

    fig = PyPlot.figure(facecolor="white")
    ax = fig[:add_subplot](111)

    ax2 = fig[:add_subplot](121)
    ax2[:scatter](Z[1,inds_Neurons], Z[2,inds_Neurons], color=COLOR_A, label="neuron")
    ax2[:set_xlim](-10,35)
    ax2[:set_ylim]( -5,20)
    ax2[:grid]()
    ax2[:legend]()

    ax3 = fig[:add_subplot](122)
    ax3[:scatter](Z[1,inds_NotNeur], Z[2,inds_NotNeur], color=COLOR_B, label="other")
    # ax3[:set_ylabel]("second principle component")
    ax3[:set_xlim](-10,35)
    ax3[:set_ylim]( -5,20)
    ax3[:grid]()
    ax3[:legend]()

    ax[:spines]["top"][:set_color]("none")
    ax[:spines]["bottom"][:set_color]("none")
    ax[:spines]["left"][:set_color]("none")
    ax[:spines]["right"][:set_color]("none")
    ax[:tick_params](labelcolor="w", top="off", bottom="off", left="off", right="off")
    ax[:set_xlabel]("first principle component")
    ax[:set_ylabel]("second principle component")

    confusion = get_confusion_array(df)
    inds_TP = find(x->x==TP, confusion)
    inds_TN = find(x->x==TN, confusion)
    inds_FP = find(x->x==FP, confusion)
    inds_FN = find(x->x==FN, confusion)

    fig = PyPlot.figure(facecolor="white")
    ax = fig[:add_subplot](111)
    ax[:scatter](Z[1,inds_TP], Z[2,inds_TP], color=COLOR_A, marker="s", label="True Positive")
    ax[:scatter](Z[1,inds_TN], Z[2,inds_TN], color=COLOR_B, marker="s", label="True Negative")
    ax[:scatter](Z[1,inds_FP], Z[2,inds_FP], color=COLOR_C, marker="^", label="False Positive")
    ax[:scatter](Z[1,inds_FN], Z[2,inds_FN], color=COLOR_D, marker="^", label="False Negative")
    ax[:set_xlabel]("first principle component")
    ax[:set_ylabel]("second principle component")
    ax[:legend]()

    println("TP: ", length(inds_TP))
    println("TN: ", length(inds_TN))
    println("FP: ", length(inds_FP))
    println("FN: ", length(inds_FN))

    variation = cumsum(Σ ./ sum(Σ))

    fig = PyPlot.figure(facecolor="white")
    ax = fig[:add_subplot](111)
    ax[:plot]([1:length(Σ)], cumsum(Σ ./ sum(Σ)), color="black")
    ax[:set_xlabel]("number of principle components")
    ax[:set_ylabel]("variation explained by principle components")
    ax[:set_xlim](1,length(Σ))
    ax[:set_ylim](0.0,1.0)
    ax[:grid]()

    println("variation explained by first k components")
    for i = 1 : length(variation)
        println(i, "  ", variation[i])
    end

    for i = 1 : 20
        println("top five features PC $i")
        p = sortperm(vec(U[:,i]), rev=true)
        colnames = get_data_names(df)
        for i = 1 : 5
            @printf("\t%20s \n", colnames[p[i]])
        end
    end
end

function export_centroids(centroids6::Matrix{Float64}, centroids2::Matrix{Float64})
    df = DataFrame()
    df[:cellA] = centroids6[:, 1]
    df[:cellB] = centroids6[:, 2]
    df[:cellC] = centroids6[:, 3]
    df[:cellD] = centroids6[:, 4]
    df[:background1]  = centroids6[:, 5]
    df[:background2] = centroids6[:, 6]
    writetable("centroids6.csv", df)

    df = DataFrame()
    df[:cell] = centroids2[:, 1]
    df[:structure] = centroids2[:, 2]
    writetable("centroids2.csv", df)
end

df = readtable("all_data_imputed.csv")
X = get_data_matrix(df)
Y = whiten(X)

df2 = readtable("all_data_imputed2.csv")
X2 = get_data_matrix(df2)
Y2 = whiten(X2)

centroids6 = get_cluster_centroids(df, X)
centroids2 = get_cluster_centroids_binary(df, X)
# export_centroids(centroids6, centroids2)

nearest_neighbors = calc_nearest_neighbors(X2)
# println(calc_in_group_proportions(centroids6, X2, nearest_neighbors))
# println(calc_in_group_proportions_binary(centroids2, X2, nearest_neighbors))
# println(calc_in_group_proportions_pvalue(centroids6, X2, nearest_neighbors))
println(calc_in_group_proportions_pvalue(centroids6, X2, nearest_neighbors))

# PyPlot.close("all")
# create_PCA_scatterplot(df, Y)

