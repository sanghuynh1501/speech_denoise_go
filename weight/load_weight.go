package weight

import (
	"encoding/gob"
	"fmt"
	"log"
	"os"

	"github.com/utahta/go-openuri"
)

type Weight struct {
	Conv2D    []float64
	W0        []float64
	W1        []float64
	BatchNorm []float64
}

type Data struct {
	Weights []Weight
}

func Load() []Weight {
	dataFile, err_file_out := openuri.Open("http://localhost:3000/integerdata.gob")
	// dataFile, err_file_out := os.Open("integerdata.gob")
	if err_file_out != nil {
		fmt.Println(err_file_out)
		os.Exit(1)
	}
	dec := gob.NewDecoder(dataFile)
	var m Data
	err := dec.Decode(&m)
	if err != nil {
		log.Fatal("decode error:", err)
	}
	dataFile.Close()

	return m.Weights
}
