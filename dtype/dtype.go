package dtype

import (
	"flag"
	"log"
	"math/rand"
	_ "net/http/pprof"

	"gorgonia.org/tensor"
)

var (
	dtype = flag.String("dtype", "float32", "Which dtype to use")
)

var Dt tensor.Dtype

func parseDtype() {
	switch *dtype {
	case "float64":
		Dt = tensor.Float64
	case "float32":
		Dt = tensor.Float32
	default:
		log.Fatalf("Unknown dtype: %v", *dtype)
	}
}

func Init_dtype() {
	flag.Parse()
	parseDtype()
	rand.Seed(1337)
}
