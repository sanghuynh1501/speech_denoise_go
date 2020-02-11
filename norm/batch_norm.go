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
var Mean_4 [][][][]float64
var Mean_3 [][][]float64
var Mean_2 [][]float64
var Mean_1 []float64

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

func mean(data []float64) float64 {
	var sum float64
	sum = 0
	for i := 0; i < len(data); i++ {
		sum += data[i]
	}
	return sum / float64(len(data))
}

func inintialData(N, H, W, C int) [][][][]float64 {
	Mean_4 = make([][][][]float64, N)
	for i := range Mean_4 {
		Mean_4[i] = make([][][]float64, H)
		for j := range Mean_4[i] {
			Mean_4[i][j] = make([][]float64, W)
			for k := range Mean_4[i][j] {
				Mean_4[i][j][k] = make([]float64, C)
			}
		}
	}
	return Mean_4
}

func inintialMeanW(N, H, W int) [][][]float64 {
	Mean_3 = make([][][]float64, N)
	for i := range Mean_3 {
		Mean_3[i] = make([][]float64, H)
		for j := range Mean_3[i] {
			Mean_3[i][j] = make([]float64, W)
		}
	}
	return Mean_3
}

func inintialMeanH(N, H int) [][]float64 {
	Mean_2 = make([][]float64, N)
	for i := range Mean_2 {
		Mean_2[i] = make([]float64, H)
	}
	return Mean_2
}

func inintialMeanN(N int) []float64 {
	Mean_1 = make([]float64, N)
	for i := range Mean_1 {
		Mean_1[i] = 0
	}
	return Mean_1
}

func mean_array(data [][][][]float64, N int, H int, W int, C int) []float64 {
	var mean0 = inintialMeanW(N, H, W)
	for n := 0; n < N; n++ {
		for h := 0; h < H; h++ {
			for w := 0; w < W; w++ {
				mean0[n][h][w] = mean(data[n][h][w])
			}
		}
	}
	var mean1 = inintialMeanH(N, H)
	for n := 0; n < N; n++ {
		for h := 0; h < H; h++ {
			mean1[n][h] = mean(mean0[n][h])
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
	for i := 0; i < N*H*W*C; i++ {
		sum = append(sum, data[i]+exp)
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
	for i := 0; i < N*H*W*C; i++ {
		sum = append(sum, math.Pow(data[i], exp))
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
