library("sparcl")

mydata = read.csv("image_features/imputed_values.csv") # read csv file

y = mydata$isCell + 1 # ∈ [1,2]

# remove unneeded columns
mydata$isCell <- NULL
mydata$directory <- NULL
mydata$objectnumber <- NULL

# extract & scale n×p matrix (n = #samples, p  = #features)
mat = scale(as.matrix(mydata))

# principle components for reference
pc = princomp(mat)

# conduct standard hierarchical clustering
d <- dist(as.matrix(mat)) # find distance matrix 
hc <- hclust(d)           # apply hierarchical clustering 
# plot(hc,main="Standard Clustering", hang=-1)
# ColorDendrogram(hc,y=y)
heatmap(t(mat), scale="col")
dev.copy2pdf(file="heatmap.pdf")

# hc2 <- HierarchicalSparseCluster(x=as.matrix(mat))
# ColorDendrogram(hc2,y=y)