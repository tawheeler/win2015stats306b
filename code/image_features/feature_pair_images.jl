using DataFrames
using Clustering
using PyPlot

function load_dataset()
    df = readtable("../all_data.csv")

    # remove entropy
    ind_entropy = findfirst(sym->sym == :entropy, names(df))
    df = df[:,[1:ind_entropy-1, ind_entropy+1:end]]
end
function export_dset(df::DataFrame, X::Matrix{Float64}, names::Vector{Symbol})

    df2 = deepcopy(df)
    for (i,sym) in enumerate(names)
        for j = 1 : size(df2,1)
            T = typeof(df2[j,sym])
            if Int == T
                df2[j,sym] = int(round(X[i,j]))
            else
                df2[j,sym] = X[i,j]
            end
        end
    end

    writetable("imputed_values.csv", df2)
end
function df_to_sample_matrix(df::DataFrame)

    m,n = size(df)

    X = Array(Float64, n-3, m)
    name_vec = names(df)

    column = 0
    for i = 1 : n
        
        column += 1
        sym = name_vec[column]
        if sym != :iscell && sym != :directory && sym != :objectnumber
            X[column,:] = array(df[sym])
        else
            deleteat!(name_vec, column)
            column -= 1
        end
    end

    X, name_vec
end
function whiten(X::Matrix{Float64})

    # de-mean & unit covariance

    Y = deepcopy(X)

    n = size(Y,1)
    means = Array(Float64, n)
    stdevs = Array(Float64, n)

    for i = 1 : n
        arr = filter(x->!isnan(x), X[i,:])
        μ = mean(arr)
        σ = stdm(arr, μ)
        Y[i,:] .-= μ
        Y[i,:] ./= σ
        means[i] = μ
        stdevs[i] = σ
    end

    (Y, means, stdevs)
end
function blacken(Y::Matrix{Float64}, means::Vector{Float64}, stdevs::Vector{Float64})
    X = deepcopy(Y)
    for i = 1 : size(X,1)
        X[i,:] .*= stdevs[i]
        X[i,:] .+= means[i]
    end
    X
end
function impute_SVD(X::Matrix{Float64}, k::Int)

    # initialize missing data with variable means
    # use a rank-k SVD of the data to impute the missing locations
    # repeat

    n,m = size(X)

    missing = falses(n,m)
    for i = 1 : n, j = 1 : m
        missing[i,j] = isnan(X[i,j]) || X[i,j] == -1
    end

    Z, means, stdevs = whiten(X)

    Z[missing] = 0.0

    iter = 0
    done = false
    while !done

        iter += 1

        U,Σ,V = svd(Z)
        Y = U * diagm([Σ[1:k], zeros(Float64, length(Σ)-k)]) * V'
        Z[missing] = Y[missing]

        if iter > 100
            done = true
        end
    end

    blacken(Z, means, stdevs)
end
function pair_image(X::Matrix{Float64}, iscell::Vector{Int}, xlabel::String, ylabel::String, ind1::Int, ind2::Int;
    color_A = [0.8,0.2,0.8,0.4],
    color_B = [0.2,0.8,0.7,0.4]
    )

    fig = PyPlot.figure(facecolor="white")

    x_arr = X[ind1,:]
    y_arr = X[ind2,:]

    cell_inds = find(i->i==1, iscell)
    vessel_inds = find(i->i==0, iscell)

    ax = fig[:add_subplot](111)
    ax[:scatter](x_arr[cell_inds],   y_arr[cell_inds],   color=color_A)
    ax[:scatter](x_arr[vessel_inds], y_arr[vessel_inds], color=color_B)
    ax[:set_xlabel](xlabel)
    ax[:set_ylabel](ylabel)
end

# df = load_dataset()
# X, name_vec = df_to_sample_matrix(df)
# X2 = impute_SVD(X, 10)
# println(X2[11,1:20])
# export_dset(df, X2, name_vec)


iscell = array(df[:iscell])
for i = 1 : 2 : length(name_vec)
    indA, indB = i,i+1
    pair_image(X2, iscell, string(name_vec[indA]), string(name_vec[indB]), indA, indB)
end


