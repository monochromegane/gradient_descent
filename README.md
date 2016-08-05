# Gradient descent in Golang

The sample implementaion of gradient descent in Golang.

- Polynomial regression
- Training set from sine function

![](images/gradient_descent.png)

## Usage

```sh
$ go run cmd/gradient_descent/main.go -eta 0.075 -m 3 -epoch 40000 -algorithm sgd -momentum 0.9
```

## Algorithms and optimization methods.

- Gradient descent
- Stochastic gradient descent (SGD)
- Mini-batch SGD
- Momentum
- AdaGrad
- AdaDelta

### Compare

![](images/errors_compare.png)
