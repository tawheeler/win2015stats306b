### Generate centroids with annotated rows
Centroids <- matrix(rnorm(30, sd = 10), 10)
rownames(Centroids) <- letters[1:nrow(Centroids)]

### Generate data with annotated rows
Data <- cbind(matrix(rep(Centroids[,1], 10), 10),
matrix(rep(Centroids[,2], 15), 10), matrix(rep(Centroids[,3], 20), 10))
Data <- Data + matrix(rnorm(length(Data), sd = 10), nrow(Data))
rownames(Data) <- letters[1:nrow(Data)]

### Classify the data and calculate the corresponding in-group
### proportions and group size
Result <- clusterRepro(Centroids, Data, Number.of.permutations = 1000)
Result$Actual.IGP
Result$Actual.Size