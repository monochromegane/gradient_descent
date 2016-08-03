package gradient_descent

import (
	"fmt"
	"math"
)

func Run(dataset DataSet, opt Opt) ([]float64, []float64) {

	// initialize
	thetas := make([]float64, opt.Degree+1)
	velocities := make([]float64, opt.Degree+1)
	gradients := make([]float64, opt.Degree+1)
	updates := make([]float64, opt.Degree+1)
	for i, _ := range thetas {
		thetas[i] = 1.0
		velocities[i] = 0.0
		gradients[i] = 0.0
		updates[i] = 0.0
	}

	errors := []float64{}

	batchSize := len(dataset)
	if opt.Algorithm == "sgd" {
		if opt.BatchSize == -1 {
			batchSize = 1
		}
	}

	// learning (update parameters)
	for i := 0; i < opt.Epoch; i++ {

		// update parameter by gradient descent
		org_thetas := make([]float64, cap(thetas))
		copy(org_thetas, thetas)

		shuffled := dataset.Shuffle()
		for j, _ := range thetas {
			// compute gradient
			gradient := gradient(shuffled, org_thetas, j, batchSize)

			var update float64
			if opt.Optimization == "adagrad" {
				// optimize by AdaGrad
				gradients[j] += math.Pow(gradient, 2)
				learningRate := opt.LearingRate / (math.Sqrt(gradients[j] + opt.Epsilon))
				update = -(learningRate * gradient)
			} else if opt.Optimization == "adadelta" {
				// optimize by AdaDelta
				gradients[j] = (opt.DecayRate * gradients[j]) + (1.0-opt.DecayRate)*math.Pow(gradient, 2)
				update = -(math.Sqrt(updates[j]+opt.Epsilon) / math.Sqrt(gradients[j]+opt.Epsilon)) * gradient
				updates[j] = (opt.DecayRate * updates[j]) + (1.0-opt.DecayRate)*math.Pow(update, 2)
			} else {
				// no optimization
				update = -(opt.LearingRate * gradient)
			}

			// Use momentum if momentum option is passed
			velocities[j] = opt.Momentum*velocities[j] + update

			// update parameter
			thetas[j] = org_thetas[j] + velocities[j]
		}

		e := ObjectiveFunction(dataset, thetas)
		if i%100 == 0 {
			errors = append(errors, e)
		}
		fmt.Printf("%d (%f) %v\n", i, e, thetas)
		// if e < 0.8 {
		// 	break
		// }
	}
	return thetas, errors
}

// gradient
func gradient(dataset DataSet, thetas []float64, index int, batchSize int) float64 {
	result := 0.0
	for _, data := range dataset[0:batchSize] {
		result += ((PredictionFunction(data.X, thetas) - data.Y) * math.Pow(data.X, float64(index)))
	}
	return result
}

// fθ(x)
func PredictionFunction(x float64, thetas []float64) float64 {
	result := 0.0
	for i, theta := range thetas {
		result += theta * math.Pow(x, float64(i))
	}
	return result
}

// E(θ)
func ObjectiveFunction(trainings DataSet, thetas []float64) float64 {
	result := 0.0
	for _, training := range trainings {
		result += math.Pow((training.Y - PredictionFunction(training.X, thetas)), 2)
	}
	return result / 2.0
}

type Opt struct {
	Epoch        int
	Degree       int
	LearingRate  float64
	Optimization string
	Algorithm    string
	BatchSize    int
	Momentum     float64
	DecayRate    float64
	Epsilon      float64
}
