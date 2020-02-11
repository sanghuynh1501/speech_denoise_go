package batch_norm

import (
	"log"
	"math"
	"math/rand"
	"speech_denoise_go/dtype"

	_ "net/http/pprof"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var letterRunes = []rune("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

type BatchNormOp struct {
	momentum float64 // momentum for the moving average
	epsilon  float64 // small variance to be added to avoid dividing by 0

	// learnables
	mean, variance, ma *tensor.Dense

	// scratch space
	meanTmp, varianceTmp, tmpSpace, xNorm                *tensor.Dense
	batchSumMultiplier, numByChans, spatialSumMultiplier *tensor.Dense

	// training? if training then update movingMean and movingVar
	training bool
}

func RandStringRunes(n int) string {
	b := make([]rune, n)
	for i := range b {
		b[i] = letterRunes[rand.Intn(len(letterRunes))]
	}
	return string(b)
}

func initial_tensor(g *gorgonia.ExprGraph, shape [4]int, value []float64, name string) *gorgonia.Node {
	value_tensor := tensor.New(tensor.WithShape(shape[0], shape[1], shape[2], shape[3]), tensor.WithBacking(value))
	value_value := gorgonia.WithValue(value_tensor)

	value_node := gorgonia.NewTensor(g, dtype.Dt, 4, gorgonia.WithShape(shape[0], shape[1], shape[2], shape[3]), gorgonia.WithName(name+RandStringRunes(3)), value_value)
	return value_node
}

func pow(g *gorgonia.ExprGraph, X *gorgonia.Node, exp float64) *gorgonia.Node {
	return pow_caculate(g, X, X.Shape()[0], X.Shape()[1], X.Shape()[2], X.Shape()[3], exp, "add")
	// total_dim := X.Shape()[0] * X.Shape()[1] * X.Shape()[2] * X.Shape()[3]
	// var exp_array []float64
	// for i := 0; i < total_dim; i++ {
	// 	exp_array = append(exp_array, exp)
	// }
	// exp_node := initial_tensor(g, [4]int{X.Shape()[0], X.Shape()[1], X.Shape()[2], X.Shape()[3]}, exp_array, "pow")
	// result, _ := gorgonia.Pow(X, exp_node)
	// return result
}

func add(g *gorgonia.ExprGraph, X *gorgonia.Node, exp float64) *gorgonia.Node {
	return add_caculate(g, X, X.Shape()[0], X.Shape()[1], X.Shape()[2], X.Shape()[3], exp, "add")
	// total_dim := X.Shape()[0] * X.Shape()[1] * X.Shape()[2] * X.Shape()[3]
	// var exp_array []float64
	// for i := 0; i < total_dim; i++ {
	// 	exp_array = append(exp_array, exp)
	// }
	// exp_node := initial_tensor(g, [4]int{X.Shape()[0], X.Shape()[1], X.Shape()[2], X.Shape()[3]}, exp_array, "pow")
	// result, _ := gorgonia.Add(X, exp_node)
	// return result
}

func Mean(data []float64) float64 {
	var sum float64
	sum = 0
	for i := 0; i < len(data); i++ {
		sum += data[i]
	}
	return sum / float64(len(data))
}

func inintialData(N, H, W, C int) [][][][]float64 {
	mean := make([][][][]float64, N)
	for i := range mean {
		mean[i] = make([][][]float64, H)
		for j := range mean[i] {
			mean[i][j] = make([][]float64, W)
			for k := range mean[i][j] {
				mean[i][j][k] = make([]float64, C)
			}
		}
	}
	return mean
}

func inintialMeanW(N, H, W int) [][][]float64 {
	mean := make([][][]float64, N)
	for i := range mean {
		mean[i] = make([][]float64, H)
		for j := range mean[i] {
			mean[i][j] = make([]float64, W)
		}
	}
	return mean
}

func inintialMeanH(N, H int) [][]float64 {
	mean := make([][]float64, N)
	for i := range mean {
		mean[i] = make([]float64, H)
	}
	return mean
}

func inintialMeanN(N int) []float64 {
	mean := make([]float64, N)
	for i := range mean {
		mean[i] = 0
	}
	return mean
}

func mean_array(data [][][][]float64, N int, H int, W int, C int) []float64 {
	var mean0 = inintialMeanW(N, H, W)
	for n := 0; n < N; n++ {
		for h := 0; h < H; h++ {
			for w := 0; w < W; w++ {
				mean0[n][h][w] = Mean(data[n][h][w])
			}
		}
	}
	var mean1 = inintialMeanH(N, H)
	for n := 0; n < N; n++ {
		for h := 0; h < H; h++ {
			mean1[n][h] = Mean(mean0[n][h])
		}
	}
	var mean2 = inintialMeanN(H)
	for h := 0; h < H; h++ {
		mean2[h] = 0
		for n := 0; n < N; n++ {
			mean2[h] += mean1[n][h]
		}
		mean2[h] = mean2[h] / float64(N)
	}
	return mean2
}

func array_inintial(shape tensor.ConsOpt, input []float64) gorgonia.NodeConsOpt {
	image := tensor.New(shape, tensor.WithBacking(input))
	return gorgonia.WithValue(image)
}

func add_caculate(g *gorgonia.ExprGraph, X *gorgonia.Node, N int, H int, W int, C int, exp float64, name string) *gorgonia.Node {
	mv := gorgonia.NewTapeMachine(g)
	defer mv.Close()
	if err := mv.RunAll(); err != nil {
		log.Fatal(err)
	}
	data := X.Value().Data().([]float64)
	var sum []float64
	i := 0
	for n := 0; n < N; n++ {
		for h := 0; h < H; h++ {
			for w := 0; w < W; w++ {
				for c := 0; c < C; c++ {
					sum = append(sum, data[i]+exp)
					i += 1
				}
			}
		}
	}
	return gorgonia.NewTensor(g, dtype.Dt, 4, gorgonia.WithShape(N, H, W, C), gorgonia.WithName(name+RandStringRunes(3)), array_inintial(tensor.WithShape(N, H, W, C), sum))
}

func pow_caculate(g *gorgonia.ExprGraph, X *gorgonia.Node, N int, H int, W int, C int, exp float64, name string) *gorgonia.Node {
	mv := gorgonia.NewTapeMachine(g)
	defer mv.Close()
	if err := mv.RunAll(); err != nil {
		log.Fatal(err)
	}
	data := X.Value().Data().([]float64)
	var sum []float64
	i := 0
	for n := 0; n < N; n++ {
		for h := 0; h < H; h++ {
			for w := 0; w < W; w++ {
				for c := 0; c < C; c++ {
					sum = append(sum, math.Pow(data[i], exp))
					i += 1
				}
			}
		}
	}
	return gorgonia.NewTensor(g, dtype.Dt, 4, gorgonia.WithShape(N, H, W, C), gorgonia.WithName(name+RandStringRunes(3)), array_inintial(tensor.WithShape(N, H, W, C), sum))
}

func mean_caculate(g *gorgonia.ExprGraph, X *gorgonia.Node, N int, H int, W int, C int, name string) *gorgonia.Node {
	mv := gorgonia.NewTapeMachine(g)
	defer mv.Close()
	if err := mv.RunAll(); err != nil {
		log.Fatal(err)
	}
	data := X.Value().Data().([]float64)
	var mean = inintialData(N, H, W, C)
	i := 0
	for n := 0; n < N; n++ {
		for h := 0; h < H; h++ {
			for w := 0; w < W; w++ {
				for c := 0; c < C; c++ {
					mean[n][h][w][c] = data[i]
					i += 1
				}
			}
		}
	}
	return gorgonia.NewTensor(g, dtype.Dt, 1, gorgonia.WithShape(H), gorgonia.WithName(name), array_inintial(tensor.WithShape(H), mean_array(mean, N, H, W, C)))
}

// mean = nd.mean(X, axis=(0,2,3))
func mean_cal(g *gorgonia.ExprGraph, X *gorgonia.Node, N, C, H, W int, name string) *gorgonia.Node {
	return mean_caculate(g, X, N, C, H, W, name)
}

// variance = nd.mean((X - mean.reshape((1, C, 1, 1))) ** 2, axis=(0, 2, 3))
func variance_cal(g *gorgonia.ExprGraph, X *gorgonia.Node, mean *gorgonia.Node, N, C, H, W int, name string) *gorgonia.Node {
	mean_reshape, _ := gorgonia.Reshape(mean, tensor.Shape{1, C, 1, 1})
	X_norm, _ := gorgonia.BroadcastSub(X, mean_reshape, nil, []byte{0, 2, 3})
	// X_norm, _ = gorgonia.Square(X_norm)
	X_norm = pow(g, X_norm, 2)
	variance := mean_cal(g, X_norm, N, C, H, W, "variance"+name)
	return variance
}

// X_hat = (X - mean.reshape((1, C, 1, 1))) * 1.0 / nd.sqrt(variance.reshape((1, C, 1, 1)) + eps)
func xhat_cal(g *gorgonia.ExprGraph, X *gorgonia.Node, mean *gorgonia.Node, variance *gorgonia.Node, C int, eps float64) *gorgonia.Node {
	mean_reshape, _ := gorgonia.Reshape(mean, tensor.Shape{1, C, 1, 1})
	X_norm, _ := gorgonia.BroadcastSub(X, mean_reshape, nil, []byte{0, 2, 3})
	variance_reshape, _ := gorgonia.Reshape(variance, tensor.Shape{1, C, 1, 1})
	variance_eps := add(g, variance_reshape, eps)
	// variance_sqrt, _ := gorgonia.Sqrt(variance_eps)
	variance_sqrt := pow(g, variance_eps, 0.5)
	xhat, _ := gorgonia.BroadcastHadamardDiv(X_norm, variance_sqrt, nil, []byte{0, 3})
	return xhat
}

func Batch_normalization(g *gorgonia.ExprGraph, X *gorgonia.Node, beta *gorgonia.Node, eps float64, name string) *gorgonia.Node {
	N := X.Shape()[0]
	C := X.Shape()[1]
	H := X.Shape()[2]
	W := X.Shape()[3]
	mean := mean_cal(g, X, N, C, H, W, "mean"+name)
	variance := variance_cal(g, X, mean, N, C, H, W, name)
	xhat := xhat_cal(g, X, mean, variance, C, eps)
	result, _ := gorgonia.BroadcastAdd(xhat, beta, nil, []byte{0, 2, 3})

	mv := gorgonia.NewTapeMachine(g)
	defer mv.Close()
	if err := mv.RunAll(); err != nil {
		log.Fatal(err)
	}

	return gorgonia.NewTensor(g, dtype.Dt, 4, gorgonia.WithShape(result.Shape()[0], result.Shape()[1], result.Shape()[2], result.Shape()[3]), gorgonia.WithName("batch_norm"+name), array_inintial(tensor.WithShape(result.Shape()[0], result.Shape()[1], result.Shape()[2], result.Shape()[3]), result.Value().Data().([]float64)))
}

// func BatchNorm(x, scale, bias *gorgonia.Node, momentum, epsilon float64) (retVal, γ, β *gorgonia.Node, op *gorgonia.BatchNormOp, err error) {
// 	batches := x.Shape()[0]
// 	channels := x.Shape()[1]
// 	spatialDim := x.Shape().TotalSize() / (channels * batches)

// 	mean := tensor.New(tensor.Of(dtype.Dt), tensor.WithShape(channels))
// 	variance := tensor.New(tensor.Of(dtype.Dt), tensor.WithShape(channels))
// 	ma := tensor.New(tensor.Of(dtype.Dt), tensor.WithShape(1))

// 	meanTmp := tensor.New(tensor.Of(dtype.Dt), tensor.WithShape(channels))
// 	varianceTmp := tensor.New(tensor.Of(dtype.Dt), tensor.WithShape(channels))
// 	tmp := tensor.New(tensor.Of(dtype.Dt), tensor.WithShape(x.Shape().Clone()...))
// 	xNorm := tensor.New(tensor.Of(dtype.Dt), tensor.WithShape(x.Shape().Clone()...))
// 	batchSumMultiplier := tensor.New(tensor.Of(dtype.Dt), tensor.WithShape(batches))

// 	var uno interface{}
// 	uno = float64(1)
// 	spatialSumMultiplier := tensor.New(tensor.Of(dtype.Dt), tensor.WithShape(spatialDim))
// 	if err = spatialSumMultiplier.Memset(uno); err != nil {
// 		return nil, nil, nil, nil, err
// 	}

// 	numByChans := tensor.New(tensor.Of(dtype.Dt), tensor.WithShape(channels*batches))
// 	if err = batchSumMultiplier.Memset(uno); err != nil {
// 		return nil, nil, nil, nil, err
// 	}

// 	op = &gorgonia.BatchNormOp{
// 		momentum: momentum,
// 		epsilon:  epsilon,

// 		mean:     mean,
// 		variance: variance,
// 		ma:       ma,

// 		meanTmp:              meanTmp,
// 		varianceTmp:          varianceTmp,
// 		tmpSpace:             tmp,
// 		xNorm:                xNorm,
// 		batchSumMultiplier:   batchSumMultiplier,
// 		numByChans:           numByChans,
// 		spatialSumMultiplier: spatialSumMultiplier,

// 		training: true,
// 	}
// 	g := x.Graph()
// 	dims := x.Shape().Dims()

// 	if scale == nil {
// 		scale = gorgonia.NewTensor(g, dtype.Dt, dims, gorgonia.WithShape(x.Shape().Clone()...), gorgonia.WithName(x.Name()+"_γ"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
// 	}
// 	if bias == nil {
// 		bias = gorgonia.NewTensor(g, dtype.Dt, dims, gorgonia.WithShape(x.Shape().Clone()...), gorgonia.WithName(x.Name()+"_β"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
// 	}

// 	if retVal, err = gorgonia.ApplyOp(op, x); err != nil {
// 		return nil, nil, nil, nil, err
// 	}
// 	if retVal, err = gorgonia.BroadcastHadamardProd(scale, retVal, nil, []byte{0, 2, 3}); err != nil {
// 		return nil, nil, nil, nil, err
// 	}
// 	retVal, err = gorgonia.BroadcastAdd(retVal, bias, nil, []byte{0, 2, 3})

// 	return retVal, scale, bias, op, err
// }
