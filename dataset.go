package gradient_descent

import (
	"math"
	"math/rand"
	"time"
)

type DataSet []Data

func (ds DataSet) Shuffle() DataSet {
	newDs := make([]Data, len(ds))
	copy(newDs, ds)
	rand.Seed(time.Now().UnixNano())
	n := len(newDs)
	for i := n - 1; i >= 0; i-- {
		j := rand.Intn(i + 1)
		newDs[i], newDs[j] = newDs[j], newDs[i]
	}
	return newDs
}

type Data struct {
	X, Y float64
}

func LoadDataSet(count int, sigma float64) DataSet {

	dataset := []Data{}

	for i := 0; i < count; i++ {
		x := float64(i) / float64(count)
		dataset = append(dataset, Data{
			X: x,
			Y: math.Sin(2.0*math.Pi*x) + normalRand(0.0, sigma),
		})
	}

	return dataset
}

func normalRand(mu, sigma float64) float64 {
	z := math.Sqrt(-2.0*math.Log(rand.Float64())) * math.Sin(2.0*math.Pi*rand.Float64())
	return sigma*z + mu

}
