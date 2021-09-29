model_file = "C:\\Users\\Aurko\\source\\repos\\simple_ml_cpp\\simple_ml_cpp\\model.txt"
layers = c(0,12,2)
weights = vector("list", length(layers) - 1)
biases = vector("list", length(layers) - 1)
for(i in 2:length(layers)) {
  num_skip = 2 * layers[i - 1]
  num_lines = layers[i]
  weights[[i - 1]] = as.matrix(read.table(model_file, nrows = num_lines, skip = num_skip))
  biases[[i - 1]] = as.vector(read.table(model_file, nrows = num_lines, skip = num_skip + num_lines))
  layers[i] = layers[i] + layers[i - 1]
}

#network <- data.frame(I(weights),I(biases))

output <- function(input){
  for(i in 1:(length(layers) - 1)){
    input = tanh(as.matrix(weights[[i]]) %*% input + as.matrix(biases[[i]]))
  }
  #out = tanh(as.matrix(w3) %*% tanh(as.matrix(w2) %*% tanh(as.matrix(w1) %*% (tanh(as.matrix(w0) %*% input + as.matrix(b0))) + as.matrix(b1)) + as.matrix(b2)) + as.matrix(b3))
  return(input[2] > input[1])
}
total_output <- expand.grid(x=seq(-12,12,0.2), y=seq(-12,12,0.2))
total_output<- transform(total_output, cat=apply(cbind(total_output$x,total_output$y),1,output))
plot(total_output$x,total_output$y,xlim=c(-15,15), ylim=c(-15,15),col=ifelse(!total_output$cat, colours()[28],colours()[38]),pch=20)

