package main

import (
	"flag"
	"fmt"
	"image/color"
	"math"

	"github.com/gonum/plot"
	"github.com/gonum/plot/plotter"
	"github.com/gonum/plot/vg"
	gd "github.com/monochromegane/gradient_descent"
)

var m int
var eta float64
var epoch int
var algorithm string
var batchSize int
var momentum float64
var optimization string
var decayRate float64
var epsilon float64

func init() {
	flag.IntVar(&epoch, "epoch", 50000, "learning count")
	flag.IntVar(&m, "m", 0, "degree")
	flag.Float64Var(&eta, "eta", 0.001, "learning rate")
	flag.StringVar(&optimization, "o", "", "gradient_descent optimization algorithm [adagrad, adadelta]")
	flag.StringVar(&algorithm, "algorithm", "gd", "gradient_descent algorithm [gd, sgd]")
	flag.IntVar(&batchSize, "batch", -1, "Minibatch-SGD batch size")
	flag.Float64Var(&momentum, "momentum", 0.0, "momentum (recommendation rate 0.9~0.95)")
	flag.Float64Var(&decayRate, "rho", 0.95, "decay rate for adadelta optimization")
	flag.Float64Var(&epsilon, "epsilon", 0.0001, "Epsilon for adagrad and adadelta optimization")
	flag.Parse()
}

func main() {

	// load dataset
	dataset := gd.LoadDataSet(100, 0.1)

	// run gradient descent
	thetas, _ := gd.Run(dataset, gd.Opt{
		Epoch:        epoch,
		Degree:       m,
		LearingRate:  eta,
		Optimization: optimization,
		Algorithm:    algorithm,
		BatchSize:    batchSize,
		Momentum:     momentum,
		DecayRate:    decayRate,
		Epsilon:      epsilon,
	})
	testingSize := 20
	testing := gd.LoadDataSet(testingSize, 0.1)
	rms := math.Sqrt((2 * gd.ObjectiveFunction(testing, thetas)) / float64(testingSize))

	// plot graph
	p, _ := plot.New()
	p.Title.Text = "Gradient descent"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"
	p.X.Min = 0
	p.X.Max = 1
	p.Y.Min = -1.5
	p.Y.Max = 1.5

	// plot dataset
	pts := make(plotter.XYs, len(dataset))
	for i, data := range dataset {
		pts[i].X = data.X
		pts[i].Y = data.Y
	}

	d, _ := plotter.NewScatter(pts)
	d.GlyphStyle.Color = color.RGBA{R: 255, G: 123, B: 51, A: 255}

	// plot dataset function
	df := plotter.NewFunction(func(x float64) float64 { return math.Sin(2.0 * math.Pi * x) })
	df.Color = color.RGBA{R: 255, G: 102, B: 51, A: 255}

	// plot result
	f := plotter.NewFunction(func(x float64) float64 { return gd.PredictionFunction(x, thetas) })
	f.Color = color.RGBA{B: 255, A: 255}

	p.Add(d, df, f)
	p.Legend.Add("dataset", d)
	p.Legend.Add(fmt.Sprintf("result | RMS:%f", rms), f)
	p.Legend.Top = true

	if err := p.Save(5*vg.Inch, 5*vg.Inch, "gradient_descent.png"); err != nil {
		panic(err)
	}

}

func plots(errors []float64) plotter.XYs {
	pts := make(plotter.XYs, len(errors))
	for i, e := range errors {
		pts[i].X = float64(i * 100)
		pts[i].Y = e
	}
	return pts
}
