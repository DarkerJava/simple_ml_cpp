high = 10
low = -10
step = 5
plot(NULL,xlim=c(low * 1.5,high * 1.5), ylim=c(low * 1.5,high * 1.5))
categories = 2
px = vector(length = length(seq(low, high - step, step))^2 * 1000)
py = vector(length = length(px))
g = vector(length = length(px))
num = 1
for (i in seq(low, high - step, step)) {
  for(j in seq(low,high - step, step)){
    cx = rnorm(1,i + step/2, step / 4)
    cy = rnorm(1,j + step/2, step/ 4)
    x = rnorm(1000, cx, step/2.5)
    y = rnorm(1000, cy, step/2.5)
    cat = sample(1:categories, 1)
    points(x,y, col=colours()[cat * 10 + 18],pch=3)
    px[num:(num + 999)] = x
    py[num:(num + 999)] = y
    g[num:(num+999)] = rep(cat,1000)
    num = num + 1000
  }
}

index <- seq(1:length(px))
t <- index[index %% 10 != 0]
v <- index[index %% 10 == 0]
tx <- px[t]
vx <- px[v]
ty <- py[t]
vy <- py[v]
tg <- g[t]
vg <- g[v]

training <- data.frame(
  x = tx,
  y=ty,
  cat = tg
)

validation <- data.frame(
  x = vx,
  y=vy,
  cat = vg
)
write.table(training,"training.txt", append = FALSE, sep = " ", dec = ".", row.names = FALSE, col.names = FALSE)
write.table(validation,"validation.txt", append = FALSE, sep = " ", dec = ".", row.names = FALSE, col.names = FALSE)