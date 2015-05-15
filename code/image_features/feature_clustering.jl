using DataFrames
using Clustering
using PyPlot

function load_dataset()
    df = readtable("all_image_features.csv")
end

function df_to_sample_matrix(df::DataFrame)

    m,n = size(df)

    X = Array(Float64, 6, m)

    count = 0
    for j = 1 : m

        
        count += 1
        X[1,count] = df[j, :roundness]
        X[2,count] = df[j, :skew]
        X[3,count] = df[j, :area]
        X[4,count] = df[j, :perimeter]
        X[5,count] = df[j, :peak]
        X[6,count] = df[j, :contrast]
        # X[6,count] = df[j, :blob_mean]
        # X[7,count] = df[j, :blob_stdev]
        # X[8,count] = df[j, :entropy]
        # X[9,count] = df[j, :subim_mean]
        # X[10,count] = df[j,:subim_stdev]
        if any(x->isnan(x), X[:,count])
            count -= 1
        end
    end

    X[:,1:count]
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
    # println(Σ)
    # D,V = eig(Σ)

    # V' * Y

end

function cluster(X::Matrix{Float64})

    axis_roundness = 1
    axis_skew      = 2
    axis_area      = 3

    res = kmeans(X, 2)

    color_A = [0.8,0.2,0.8,0.4]
    color_B = [0.2,0.8,0.7,0.4]
    color_C = [0.1,0.7,0.1,0.4]

    pts_A = find(x->x==1, res.assignments)
    pts_B = find(x->x==2, res.assignments)
    # pts_C = find(x->x==3, res.assignments)

    fig = PyPlot.figure(facecolor="white")
    ax1 = fig[:add_subplot](131)
    ax1[:scatter](X[axis_roundness,pts_A], X[axis_skew,pts_A], color=color_A)
    ax1[:scatter](X[axis_roundness,pts_B], X[axis_skew,pts_B], color=color_B)
    # ax1[:scatter](X[axis_roundness,pts_C], X[axis_skew,pts_C], color=color_C)
    ax1[:set_xlabel]("roundness [-]")
    ax1[:set_ylabel]("skew [-]")

    ax2 = fig[:add_subplot](132)
    ax2[:scatter](X[axis_skew,pts_A], X[axis_area,pts_A], color=color_A)
    ax2[:scatter](X[axis_skew,pts_B], X[axis_area,pts_B], color=color_B)
    ax2[:set_xlabel]("skew [-]")
    ax2[:set_ylabel]("area [pix]")

    ax3 = fig[:add_subplot](133)
    ax3[:scatter](X[axis_area,pts_A], X[axis_roundness,pts_A], color=color_A)
    ax3[:scatter](X[axis_area,pts_B], X[axis_roundness,pts_B], color=color_B)
    ax3[:set_xlabel]("area [pix]")
    ax3[:set_ylabel]("roundness [-]")

    U,Σ,V = svd(X)
    S = U' * X

    fig = PyPlot.figure(facecolor="white")
    ax = fig[:add_subplot](111)
    ax[:scatter](S[1,pts_A], S[2,pts_A], color=color_A)
    ax[:scatter](S[1,pts_B], S[2,pts_B], color=color_B)
    ax[:set_xlabel]("first principle component")
    ax[:set_ylabel]("second principle component")
end

df = load_dataset()
X = df_to_sample_matrix(df)
Z = whiten(X)
cluster(Z)