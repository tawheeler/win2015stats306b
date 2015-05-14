mydata = read.csv("all_data_nona.csv") # read csv file
mydata$iscell <- NULL
mydata$directory <- NULL
mydata$objectnumber <- NULL
mat = scale(as.matrix(mydata))

pc = princomp(mat)

# d <- dist(as.matrix(mat))   # find distance matrix 
# hc <- hclust(d)             # apply hirarchical clustering 
# plot(hc)                    # plot the dendrogram

heatmap(t(mat), scale="col")