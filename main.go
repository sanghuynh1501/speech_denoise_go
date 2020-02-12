package main

import (
	"bytes"
	"encoding/base64"
	"io"
	"log"
	"math"
	"math/rand"
	_ "net/http/pprof"
	"speech_denoise_go/dtype"
	"strings"
	"syscall/js"

	// batch_norm "speech_denoise_go/norm"
	"speech_denoise_go/weight"
	"strconv"
	"time"

	"github.com/pkg/errors"
	"github.com/youpy/go-wav"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type convnet struct {
	g           *gorgonia.ExprGraph
	w1Weights   []*gorgonia.Node // weights. the number at the back indicates which layer it's used for
	w2Weights   []*gorgonia.Node
	convWeights []*gorgonia.Node // weights. the number at the back indicates which layer it's used for
	normWeights []*gorgonia.Node
	out         *gorgonia.Node
	predVal     gorgonia.Value
}

const charset = "abcdefghijklmnopqrstuvwxyz" +
	"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

var seededRand *rand.Rand = rand.New(
	rand.NewSource(time.Now().UnixNano()))

func StringWithCharset(length int, charset string) string {
	b := make([]byte, length)
	for i := range b {
		b[i] = charset[seededRand.Intn(len(charset))]
	}
	return string(b)
}

func String(length int) string {
	return StringWithCharset(length, charset)
}

func initial_tensor(g *gorgonia.ExprGraph, X *gorgonia.Node, value float64, name string) *gorgonia.Node {
	total_dim := X.Shape()[0] * X.Shape()[1] * X.Shape()[2] * X.Shape()[3]
	var exp_array []float64
	for i := 0; i < total_dim; i++ {
		exp_array = append(exp_array, value)
	}
	value_tensor := tensor.New(tensor.WithShape(X.Shape()[0], X.Shape()[1], X.Shape()[2], X.Shape()[3]), tensor.WithBacking(exp_array))
	value_value := gorgonia.WithValue(value_tensor)
	value_node := gorgonia.NewTensor(g, dtype.Dt, 4, gorgonia.WithShape(X.Shape()[0], X.Shape()[1], X.Shape()[2], X.Shape()[3]), gorgonia.WithName(name+String(3)), value_value)
	return value_node
}

func array_inintial(shape tensor.ConsOpt, input []float64) gorgonia.NodeConsOpt {
	image := tensor.New(shape, tensor.WithBacking(input))
	return gorgonia.WithValue(image)
}

func newConvNet(g *gorgonia.ExprGraph, n_layers int, n_channels int, ksz int, weights_array []weight.Weight) *convnet {
	var convWeights []*gorgonia.Node
	var normWeights []*gorgonia.Node
	var w1Weights []*gorgonia.Node
	var w2Weights []*gorgonia.Node

	in_channels := 1
	for i := 0; i < n_layers+1; i++ {
		var w *gorgonia.Node
		w = gorgonia.NewTensor(g, dtype.Dt, 4, gorgonia.WithShape(n_channels, in_channels, 1, ksz), gorgonia.WithName("conv_w"+strconv.FormatInt(int64(i), 10)), array_inintial(tensor.WithShape(n_channels, in_channels, 1, ksz), weights_array[i].Conv2D))
		convWeights = append(convWeights, w)

		w = gorgonia.NewTensor(g, dtype.Dt, 4, gorgonia.WithShape(1, n_channels, 1, 1), gorgonia.WithName("norm_w"+strconv.FormatInt(int64(i), 10)), array_inintial(tensor.WithShape(1, n_channels, 1, 1), weights_array[i].BatchNorm))
		normWeights = append(normWeights, w)

		w = gorgonia.NewTensor(g, dtype.Dt, 4, gorgonia.WithShape(1, 1, 1, 1), gorgonia.WithName("w1_w"+strconv.FormatInt(int64(i), 10)), array_inintial(tensor.WithShape(1, 1, 1, 1), weights_array[i].W0))
		w1Weights = append(w1Weights, w)

		w = gorgonia.NewTensor(g, dtype.Dt, 4, gorgonia.WithShape(1, 1, 1, 1), gorgonia.WithName("w2_w"+strconv.FormatInt(int64(i), 10)), array_inintial(tensor.WithShape(1, 1, 1, 1), weights_array[i].W1))
		w2Weights = append(w2Weights, w)

		in_channels = n_channels
	}

	var w *gorgonia.Node
	w = gorgonia.NewTensor(g, dtype.Dt, 4, gorgonia.WithShape(1, in_channels, 1, 1), gorgonia.WithName("conv_w"+strconv.FormatInt(int64(n_layers+1), 10)), array_inintial(tensor.WithShape(1, in_channels, 1, 1), weights_array[n_layers+1].Conv2D))
	convWeights = append(convWeights, w)

	w = gorgonia.NewTensor(g, dtype.Dt, 4, gorgonia.WithShape(1, 1, 1, 1), gorgonia.WithName("norm_w"+strconv.FormatInt(int64(n_layers+1), 10)), array_inintial(tensor.WithShape(1, 1, 1, 1), weights_array[n_layers+1].BatchNorm))
	normWeights = append(normWeights, w)

	w = gorgonia.NewTensor(g, dtype.Dt, 4, gorgonia.WithShape(1, 1, 1, 1), gorgonia.WithName("w1_w"+strconv.FormatInt(int64(n_layers+1), 10)), array_inintial(tensor.WithShape(1, 1, 1, 1), weights_array[n_layers+1].W0))
	w1Weights = append(w1Weights, w)

	w = gorgonia.NewTensor(g, dtype.Dt, 4, gorgonia.WithShape(1, 1, 1, 1), gorgonia.WithName("w2_w"+strconv.FormatInt(int64(n_layers+1), 10)), array_inintial(tensor.WithShape(1, 1, 1, 1), weights_array[n_layers+1].W1))
	w2Weights = append(w2Weights, w)

	return &convnet{
		g:           g,
		convWeights: convWeights,
		normWeights: normWeights,
		w1Weights:   w1Weights,
		w2Weights:   w2Weights,
	}
}

// This function is particularly verbose for educational reasons. In reality, you'd wrap up the layers within a layer struct type and perform per-layer activations
func (m *convnet) fwd(g *gorgonia.ExprGraph, x *gorgonia.Node, n_layers int, ksz int) (err error) {
	for i := 0; i < n_layers; i++ {
		if i == 0 {
			log.Println("m.convWeights[i].Shape() ", m.convWeights[i].Shape())
			if x, err = gorgonia.Conv2d(x, m.convWeights[i], tensor.Shape{1, ksz}, []int{0, 1}, []int{1, 1}, []int{1, 1}); err != nil {
				return errors.Wrap(err, "Layer 0 Convolution failed")
			}
			// norm := batch_norm.Batch_normalization(g, x, m.normWeights[i], 0.001, "layer"+strconv.FormatInt(int64(i), 10))
			_, norm_weight, _ := gorgonia.Broadcast(x, m.normWeights[i], gorgonia.NewBroadcastPattern(nil, []byte{0, 2, 3}))
			norm, _, _, _, _ := gorgonia.BatchNorm(x, initial_tensor(g, x, 1, "scale"), norm_weight, 0, 0.001)
			x_w1, _ := gorgonia.BroadcastHadamardProd(x, m.w1Weights[i], nil, []byte{0, 1, 2, 3})
			w2_norm, _ := gorgonia.BroadcastHadamardProd(norm, m.w2Weights[i], nil, []byte{0, 1, 2, 3})
			x, _ = gorgonia.Add(x_w1, w2_norm)
			if x, err = gorgonia.LeakyRelu(x, 0.2); err != nil {
				return errors.Wrap(err, "Layer 0 activation failed")
			}

		} else {
			if x, err = gorgonia.Conv2d(x, m.convWeights[i], tensor.Shape{1, ksz}, []int{0, int(math.Pow(2, float64(i)))}, []int{1, 1}, []int{1, int(math.Pow(2, float64(i)))}); err != nil {
				return errors.Wrap(err, "Layer 0 Convolution failed")
			}
			// norm := batch_norm.Batch_normalization(g, x, m.normWeights[i], 0.001, "layer"+strconv.FormatInt(int64(i), 10))
			_, norm_weight, _ := gorgonia.Broadcast(x, m.normWeights[i], gorgonia.NewBroadcastPattern(nil, []byte{0, 2, 3}))
			norm, _, _, _, _ := gorgonia.BatchNorm(x, initial_tensor(g, x, 1, "scale"), norm_weight, 0, 0.001)
			x_w1, _ := gorgonia.BroadcastHadamardProd(x, m.w1Weights[i], nil, []byte{0, 1, 2, 3})
			w2_norm, _ := gorgonia.BroadcastHadamardProd(norm, m.w2Weights[i], nil, []byte{0, 1, 2, 3})
			x, _ = gorgonia.Add(x_w1, w2_norm)
			if x, err = gorgonia.LeakyRelu(x, 0.2); err != nil {
				return errors.Wrap(err, "Layer 0 activation failed")
			}
		}
	}

	if x, err = gorgonia.Conv2d(x, m.convWeights[n_layers], tensor.Shape{1, ksz}, []int{0, 1}, []int{1, 1}, []int{1, 1}); err != nil {
		return errors.Wrap(err, "Layer 0 Convolution failed")
	}
	// norm := batch_norm.Batch_normalization(g, x, m.normWeights[n_layers], 0.001, "layer"+strconv.FormatInt(int64(n_layers), 10))
	_, norm_weight, _ := gorgonia.Broadcast(x, m.normWeights[n_layers], gorgonia.NewBroadcastPattern(nil, []byte{0, 2, 3}))
	norm, _, _, _, _ := gorgonia.BatchNorm(x, initial_tensor(g, x, 1, "scale"), norm_weight, 0, 0.001)
	x_w1, _ := gorgonia.BroadcastHadamardProd(x, m.w1Weights[n_layers], nil, []byte{0, 1, 2, 3})
	w2_norm, _ := gorgonia.BroadcastHadamardProd(norm, m.w2Weights[n_layers], nil, []byte{0, 1, 2, 3})
	x, _ = gorgonia.Add(x_w1, w2_norm)
	if x, err = gorgonia.LeakyRelu(x, 0.2); err != nil {
		return errors.Wrap(err, "Layer 0 activation failed")
	}

	if x, err = gorgonia.Conv2d(x, m.convWeights[n_layers+1], tensor.Shape{1, 1}, []int{0, 0}, []int{1, 1}, []int{1, 1}); err != nil {
		return errors.Wrap(err, "Layer 0 Convolution failed")
	}
	x, err = gorgonia.BroadcastAdd(x, m.normWeights[n_layers+1], nil, []byte{0, 2, 3})

	m.out = x
	gorgonia.Read(m.out, &m.predVal)

	return
}

func floatToString(input_num float64, number int) string {
	// to convert a float number to a string
	return strconv.FormatFloat(input_num, 'f', number, 64)
}

func getAudio() []float64 {
	doc := js.Global().Get("document")
	image_element := doc.Call("getElementById", "predict_source")
	audio_base64 := image_element.Get("src")
	coI := strings.Index(audio_base64.String(), ",")
	raw_audio := audio_base64.String()[coI+1:]

	unbased, _ := base64.StdEncoding.DecodeString(string(raw_audio))
	res := bytes.NewReader(unbased)
	reader := wav.NewReader(res)
	var audio_array []float64
	for {
		samples, err := reader.ReadSamples()
		if err == io.EOF {
			break
		}

		for _, sample := range samples {
			audio_array = append(audio_array, reader.FloatValue(sample, 0)*2)
		}
	}
	return audio_array
}

var xValue *gorgonia.Node
var g *gorgonia.ExprGraph
var m *convnet

func predict(weights_array []weight.Weight, n_layers int, ksz int, input []float64) []float64 {
	g = gorgonia.NewGraph()
	m = newConvNet(g, n_layers, 64, ksz, weights_array)
	xValue = gorgonia.NewTensor(g, dtype.Dt, 4, gorgonia.WithShape(1, 1, 1, len(input)), gorgonia.WithName("xValue"), array_inintial(tensor.WithShape(1, 1, 1, len(input)), input))
	log.Println("xValue.Shape() ", xValue.Shape())
	if err := m.fwd(g, xValue, n_layers, ksz); err != nil {
		log.Fatalf("%+v", err)
	}
	mv := gorgonia.NewTapeMachine(g)
	defer mv.Close()
	if err := mv.RunAll(); err != nil {
		log.Fatal(err)
	}

	return m.out.Value().Data().([]float64)
}

func update_percent(percent float64) {
	doc := js.Global().Get("document")
	percent_elements := doc.Call("getElementsByClassName", "ant-progress-bg")
	percent_elements.Index(0).Get("style").Set("width", floatToString(percent*100, 1)+"%")
	text_elements := doc.Call("getElementsByClassName", "ant-progress-text")
	text_elements.Index(0).Set("innerHTML", floatToString(percent*100, 1)+"%")
}

func main() {
	dtype.Init_dtype()

	weights_array := weight.Load()
	n_layers := 13
	ksz := 3

	audio_array := getAudio()

	// file, _ := os.Open("p232_001.wav")
	// reader := wav.NewReader(file)

	// defer file.Close()

	// var audio_array []float64
	// for {
	// 	samples, err := reader.ReadSamples()
	// 	if err == io.EOF {
	// 		break
	// 	}

	// 	for _, sample := range samples {
	// 		audio_array = append(audio_array, reader.FloatValue(sample, 0)*2)
	// 	}
	// }

	var predict_array []float64
	var result_array []string
	var sub_array []float64
	var sample float64

	for i := 0; i < len(audio_array); i += 2000 {
		if i+2000 < len(audio_array) {
			sub_array = audio_array[i : i+2000]
		} else {
			sub_array = audio_array[i:len(audio_array)]
		}
		predict_array = predict(weights_array, n_layers, ksz, sub_array)
		for _, sample = range predict_array {
			result_array = append(result_array, floatToString(sample, 18))
		}
		percent := float64(len(result_array)) / float64(len(audio_array))
		update_percent(percent)
	}

	log.Println("result_array ", result_array)
	js.Global().Set("result", strings.Join(result_array, ","))
}
