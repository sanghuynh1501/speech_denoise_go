package main

import (
	"bytes"
	"encoding/base64"
	"io"
	"log"
	"math"
	_ "net/http/pprof"
	"os"
	"speech_denoise_go/dtype"
	"strings"
	"syscall/js"

	"speech_denoise_go/weight"
	"strconv"

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
	gamma       *gorgonia.Node
	out         *gorgonia.Node
	predVal     gorgonia.Value
}

var image tensor.Tensor

func array_inintial(shape tensor.ConsOpt, input []float64) gorgonia.NodeConsOpt {
	image = tensor.New(shape, tensor.WithBacking(input))
	return gorgonia.WithValue(image)
}

var in_channels int
var w *gorgonia.Node

var convWeights []*gorgonia.Node
var normWeights []*gorgonia.Node
var w1Weights []*gorgonia.Node
var w2Weights []*gorgonia.Node
var gamma *gorgonia.Node

func newConvNet(g *gorgonia.ExprGraph, n_layers int, n_channels int, ksz int, weights_array []weight.Weight) *convnet {
	convWeights = []*gorgonia.Node{}
	normWeights = []*gorgonia.Node{}
	w1Weights = []*gorgonia.Node{}
	w2Weights = []*gorgonia.Node{}

	in_channels = 1
	gamma = gorgonia.NewTensor(g, dtype.Dt, 4, gorgonia.WithShape(1, n_channels, 1, 2000), gorgonia.WithName("gamma"+strconv.FormatInt(int64(0), 10)), gorgonia.WithInit(gorgonia.Ones()))
	for i := 0; i < n_layers+1; i++ {
		w = gorgonia.NewTensor(g, dtype.Dt, 4, gorgonia.WithShape(n_channels, in_channels, 1, ksz), gorgonia.WithName("conv_w"+strconv.FormatInt(int64(i), 10)), array_inintial(tensor.WithShape(n_channels, in_channels, 1, ksz), weights_array[i].Conv2D))
		convWeights = append(convWeights, w)

		w = gorgonia.NewTensor(g, dtype.Dt, 4, gorgonia.WithShape(1, n_channels, 1, 2000), gorgonia.WithName("norm_w"+strconv.FormatInt(int64(i), 10)), array_inintial(tensor.WithShape(1, n_channels, 1, 2000), weights_array[i].BatchNorm))
		normWeights = append(normWeights, w)

		w = gorgonia.NewTensor(g, dtype.Dt, 4, gorgonia.WithShape(1, n_channels, 1, 2000), gorgonia.WithName("w1_w"+strconv.FormatInt(int64(i), 10)), array_inintial(tensor.WithShape(1, n_channels, 1, 2000), weights_array[i].W0))
		w1Weights = append(w1Weights, w)

		w = gorgonia.NewTensor(g, dtype.Dt, 4, gorgonia.WithShape(1, n_channels, 1, 2000), gorgonia.WithName("w2_w"+strconv.FormatInt(int64(i), 10)), array_inintial(tensor.WithShape(1, n_channels, 1, 2000), weights_array[i].W1))
		w2Weights = append(w2Weights, w)

		in_channels = n_channels
	}

	w = gorgonia.NewTensor(g, dtype.Dt, 4, gorgonia.WithShape(1, in_channels, 1, 1), gorgonia.WithName("conv_w"+strconv.FormatInt(int64(n_layers+1), 10)), array_inintial(tensor.WithShape(1, in_channels, 1, 1), weights_array[n_layers+1].Conv2D))
	convWeights = append(convWeights, w)

	w = gorgonia.NewTensor(g, dtype.Dt, 4, gorgonia.WithShape(1, 1, 1, 2000), gorgonia.WithName("norm_w"+strconv.FormatInt(int64(n_layers+1), 10)), array_inintial(tensor.WithShape(1, 1, 1, 2000), weights_array[n_layers+1].BatchNorm))
	normWeights = append(normWeights, w)

	w = gorgonia.NewTensor(g, dtype.Dt, 4, gorgonia.WithShape(1, 1, 1, 2000), gorgonia.WithName("w1_w"+strconv.FormatInt(int64(n_layers+1), 10)), array_inintial(tensor.WithShape(1, 1, 1, 2000), weights_array[n_layers+1].W0))
	w1Weights = append(w1Weights, w)

	w = gorgonia.NewTensor(g, dtype.Dt, 4, gorgonia.WithShape(1, 1, 1, 2000), gorgonia.WithName("w2_w"+strconv.FormatInt(int64(n_layers+1), 10)), array_inintial(tensor.WithShape(1, 1, 1, 2000), weights_array[n_layers+1].W1))
	w2Weights = append(w2Weights, w)

	return &convnet{
		g:           g,
		convWeights: convWeights,
		normWeights: normWeights,
		w1Weights:   w1Weights,
		w2Weights:   w2Weights,
		gamma:       gamma,
	}
}

var norm, x_w1, w2_norm *gorgonia.Node

func (m *convnet) fwd(g *gorgonia.ExprGraph, x *gorgonia.Node, n_layers int, ksz int) (err error) {

	for i := 0; i < n_layers; i++ {
		if i == 0 {
			x, _ = gorgonia.Conv2d(x, m.convWeights[i], tensor.Shape{1, ksz}, []int{0, 1}, []int{1, 1}, []int{1, 1})
			norm, _, _, _, _ = gorgonia.BatchNorm(x, m.gamma, m.normWeights[i], 0, 0.001)
			x_w1, _ = gorgonia.HadamardProd(x, m.w1Weights[i])
			w2_norm, _ = gorgonia.HadamardProd(norm, m.w2Weights[i])
			x, _ = gorgonia.Add(x_w1, w2_norm)
			x, _ = gorgonia.LeakyRelu(x, 0.2)

		} else {
			x, _ = gorgonia.Conv2d(x, m.convWeights[i], tensor.Shape{1, ksz}, []int{0, int(math.Pow(2, float64(i)))}, []int{1, 1}, []int{1, int(math.Pow(2, float64(i)))})
			norm, _, _, _, _ = gorgonia.BatchNorm(x, m.gamma, m.normWeights[i], 0, 0.001)
			x_w1, _ = gorgonia.HadamardProd(x, m.w1Weights[i])
			w2_norm, _ = gorgonia.HadamardProd(norm, m.w2Weights[i])
			x, _ = gorgonia.Add(x_w1, w2_norm)
			x, err = gorgonia.LeakyRelu(x, 0.2)
		}
	}

	x, _ = gorgonia.Conv2d(x, m.convWeights[n_layers], tensor.Shape{1, ksz}, []int{0, 1}, []int{1, 1}, []int{1, 1})
	norm, _, _, _, _ = gorgonia.BatchNorm(x, m.gamma, m.normWeights[n_layers], 0, 0.001)
	x_w1, _ = gorgonia.HadamardProd(x, m.w1Weights[n_layers])
	w2_norm, _ = gorgonia.HadamardProd(norm, m.w2Weights[n_layers])
	x, _ = gorgonia.Add(x_w1, w2_norm)
	x, _ = gorgonia.LeakyRelu(x, 0.2)

	x, _ = gorgonia.Conv2d(x, m.convWeights[n_layers+1], tensor.Shape{1, 1}, []int{0, 0}, []int{1, 1}, []int{1, 1})
	x, _ = gorgonia.Add(x, m.normWeights[n_layers+1])

	m.out = x
	gorgonia.Read(m.out, &m.predVal)

	return
}

func floatToString(input_num float64, number int) string {
	return strconv.FormatFloat(input_num, 'f', number, 64)
}

var audio_array []float64

func getAudio() {
	doc := js.Global().Get("document")
	image_element := doc.Call("getElementById", "predict_source")
	audio_base64 := image_element.Get("src")
	coI := strings.Index(audio_base64.String(), ",")
	raw_audio := audio_base64.String()[coI+1:]

	unbased, _ := base64.StdEncoding.DecodeString(string(raw_audio))
	res := bytes.NewReader(unbased)
	reader := wav.NewReader(res)
	for {
		samples, err := reader.ReadSamples()
		if err == io.EOF {
			break
		}

		for _, sample := range samples {
			audio_array = append(audio_array, reader.FloatValue(sample, 0)*2)
		}
	}
}

var xValue *gorgonia.Node
var g *gorgonia.ExprGraph
var m *convnet
var err error

func predict(weights_array []weight.Weight, n_layers int, ksz int, input []float64) []float64 {
	g = gorgonia.NewGraph()
	xValue = gorgonia.NewTensor(g, dtype.Dt, 4, gorgonia.WithShape(1, 1, 1, len(input)), gorgonia.WithName("xValue"), array_inintial(tensor.WithShape(1, 1, 1, len(input)), input))
	m = newConvNet(g, n_layers, 64, ksz, weights_array)
	if err = m.fwd(g, xValue, n_layers, ksz); err != nil {
		log.Fatalf("%+v", err)
	}
	var mv = gorgonia.NewTapeMachine(g)
	defer mv.Close()
	if err = mv.RunAll(); err != nil {
		log.Fatal(err)
	}
	xValue = nil
	return m.out.Value().Data().([]float64)
	// return []float64{}
}

func readAudio() {
	file, _ := os.Open("p232_001.wav")
	reader := wav.NewReader(file)

	defer file.Close()

	for {
		samples, err := reader.ReadSamples()
		if err == io.EOF {
			break
		}

		for _, sample := range samples {
			audio_array = append(audio_array, reader.FloatValue(sample, 0)*2)
		}
	}

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

	weight.Load()
	n_layers := 13
	ksz := 3

	getAudio()
	// readAudio()

	var predict_array []float64
	var result_array []string
	var sub_array []float64
	var sample float64

	for len(audio_array) > 0 {
		log.Println("start")
		if len(audio_array) > 2000 {
			sub_array = audio_array[:2000]
			audio_array = audio_array[2000:]
		} else {
			sub_array = audio_array[:len(audio_array)]
			audio_array = []float64{}
		}
		for len(sub_array) < 2000 {
			sub_array = append(sub_array, 0)
		}
		predict_array = predict(weight.W_Data.Weights, n_layers, ksz, sub_array)
		for _, sample = range predict_array {
			result_array = append(result_array, floatToString(sample, 18))
		}
		// percent := float64(len(result_array)) / float64(len(audio_array))
		// update_percent(percent)

		predict_array = nil
		sub_array = nil
		m = nil
	}

	log.Println("result_array ", result_array)
	js.Global().Set("result", strings.Join(result_array, ","))
}
