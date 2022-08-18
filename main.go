package main

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
	"time"

	"github.com/dwkim606/test_lattigo/ckks"
	"github.com/dwkim606/test_lattigo/rlwe"
)

var err error

const log_c_scale = 30
const log_in_scale = 30
const log_out_scale = 30

type context struct {
	logN    int
	N       int
	ECD_LV  int   // LV of input ctxt and kernels (= 1)
	in_wids []int // input widths including padding
	kp_wids []int // keep widths among input widths
	// pads           map[int]int
	ext_idx        map[int][][]int         // ext_idx for keep_vec (saved for each possible input width) map: in_wid, [up/low]
	r_idx          map[int][]map[int][]int // r_idx for compr_vec (or ext_vec) map: in_wid [pos] map: rot
	r_idx_l        map[int][]map[int][]int // low, r_idx for compr_vec (or ext_vec) map: in_wid [pos] map: rot
	m_idx          map[int][]map[int][]int // m_idx , map: in_wid [pos] map: rot
	m_idx_l        map[int][]map[int][]int // low, m_idx , map: in_wid [pos] map: rot
	pl_idx         []*ckks.Plaintext
	params         ckks.Parameters
	encoder        ckks.Encoder
	encryptor      ckks.Encryptor
	decryptor      ckks.Decryptor
	evaluator      ckks.Evaluator
	pack_evaluator ckks.Evaluator
	btp            *ckks.Bootstrapper
}

func newContext(logN, ker_wid int, in_wids, kp_wids []int, boot bool, kind string) *context {
	cont_start := time.Now()
	cont := context{N: (1 << logN), logN: logN, ECD_LV: 1}
	cont.in_wids = make([]int, len(in_wids))
	copy(cont.in_wids, in_wids)
	cont.kp_wids = make([]int, len(kp_wids))
	copy(cont.kp_wids, kp_wids)

	btpParams := ckks.DefaultBootstrapParams[6]
	if kind == "BL_Conv" {
		btpParams = ckks.DefaultBootstrapParams[7]
	}
	cont.params, err = btpParams.Params()
	if err != nil {
		panic(err)
	}

	fmt.Printf("CKKS parameters: logN = %d, logSlots = %d, h = %d, logQP = %d, levels = %d, scale= 2^%f, sigma = %f \n",
		cont.params.LogN(), cont.params.LogSlots(), btpParams.H, cont.params.LogQP(), cont.params.QCount(), math.Log2(cont.params.Scale()), cont.params.Sigma())
	if cont.params.N() != cont.N {
		fmt.Println("Set Boot logN to", logN)
		panic("Boot N != N")
	}
	// Gen rotations
	var rotations []int
	cont.ext_idx = make(map[int][][]int)
	cont.r_idx = make(map[int][]map[int][]int)
	cont.r_idx_l = make(map[int][]map[int][]int)
	cont.m_idx = make(map[int][]map[int][]int)
	cont.m_idx_l = make(map[int][]map[int][]int)
	var iter int

	switch kind {
	case "BL_Conv":
		for _, elt := range cont.in_wids {
			for k := -(ker_wid / 2); k <= ker_wid/2; k++ {
				for k2 := -(ker_wid / 2); k2 <= ker_wid/2; k2++ {
					rotations = append(rotations, k*elt+k2)
				}
			}
			out_batch := (cont.N / 2) / (elt * elt)
			for k := 1; k < out_batch; k++ {
				rotations = append(rotations, k*elt*elt)
			}
		}
	case "Conv": // we assume manual padding using kp_wid
		if boot {
			iter = 2 // we assume full padding, i.e., up and low is both nonzero
			for i, elt := range cont.in_wids {
				cont.ext_idx[elt] = make([][]int, iter)
				for ul := 0; ul < iter; ul++ {
					cont.ext_idx[elt][ul] = gen_keep_vec(cont.N/2, elt, cont.kp_wids[i], ul)
				}
			}
		}
	case "Resnet_crop_fast": // Generate ext_idx for extracting valid values from conv with "same" padding
		iter = 2 // since we use full padding,
		for i := range cont.in_wids {
			step := 1 << i
			raw_in_wid_odd := true
			if cont.kp_wids[i]%2 == 0 {
				raw_in_wid_odd = false
			}
			// println("i, odd? ", i, raw_in_wid_odd)
			cont.ext_idx[step] = make([][]int, iter)
			for ul := 0; ul < iter; ul++ {
				cont.ext_idx[step][ul] = gen_keep_vec_stride(cont.N/2, cont.in_wids[0], cont.kp_wids[i], step, ul, raw_in_wid_odd)
			}
		}
	case "Resnet_crop_fast_wide2": // Generate ext_idx for extracting valid values from conv with "same" padding
		iter = 2 // since we use full padding,
		for i := 1; i <= 2; i++ {
			step := 1 << (i - 1)
			raw_in_wid_odd := true
			if cont.kp_wids[i]%2 == 0 {
				raw_in_wid_odd = false
			}
			cont.ext_idx[step] = make([][]int, iter)
			for ul := 0; ul < iter; ul++ {
				cont.ext_idx[step][ul] = gen_keep_vec_stride(cont.N/2, cont.in_wids[1], cont.kp_wids[i], step, ul, raw_in_wid_odd)
			}
		}

		// for first block (normal conv)
		elt := cont.in_wids[0]
		cont.ext_idx[elt] = make([][]int, iter)
		for ul := 0; ul < iter; ul++ {
			cont.ext_idx[elt][ul] = gen_keep_vec(cont.N/2, elt, cont.kp_wids[0], ul)
		}

		// for first block (stride)
		cont.r_idx[elt] = make([]map[int][]int, 1)
		cont.r_idx_l[elt] = make([]map[int][]int, 1)
		cont.m_idx[elt] = make([]map[int][]int, 1)
		cont.m_idx_l[elt] = make([]map[int][]int, 1)
		pos := 0
		cont.m_idx[elt][pos], cont.r_idx[elt][pos] = gen_comprs_fast(cont.N/2, elt, 2*cont.kp_wids[1], pos, 0)
		cont.m_idx_l[elt][pos], cont.r_idx_l[elt][pos] = gen_comprs_fast(cont.N/2, elt, 2*cont.kp_wids[1], pos, 1)
		for k := range cont.r_idx[elt][pos] {
			rotations = append(rotations, k)
		}
		for k := range cont.r_idx_l[elt][pos] {
			rotations = append(rotations, k)
		}
		for k := range cont.m_idx[elt][pos] {
			rotations = append(rotations, k)
		}
		for k := range cont.m_idx_l[elt][pos] {
			rotations = append(rotations, k)
		}
	case "Resnet_crop_fast_wide3": // the same as wide2 but additional stride// Generate ext_idx for extracting valid values from conv with "same" padding
		iter = 2 // since we use full padding,
		for i := 1; i <= 2; i++ {
			step := 1 << (i - 1)
			raw_in_wid_odd := true
			if cont.kp_wids[i]%2 == 0 {
				raw_in_wid_odd = false
			}
			cont.ext_idx[step] = make([][]int, iter)
			for ul := 0; ul < iter; ul++ {
				cont.ext_idx[step][ul] = gen_keep_vec_stride(cont.N/2, cont.in_wids[1], cont.kp_wids[i], step, ul, raw_in_wid_odd)
			}
		}

		// for first block (normal conv)
		elt := cont.in_wids[0]
		cont.ext_idx[elt] = make([][]int, iter)
		for ul := 0; ul < iter; ul++ {
			cont.ext_idx[elt][ul] = gen_keep_vec(cont.N/2, elt, cont.kp_wids[0], ul)
		}

		// for first block (stride)
		cont.r_idx[elt] = make([]map[int][]int, 4)
		cont.r_idx_l[elt] = make([]map[int][]int, 4)
		cont.m_idx[elt] = make([]map[int][]int, 4)
		cont.m_idx_l[elt] = make([]map[int][]int, 4)
		for pos := 0; pos < 4; pos += 2 {
			cont.m_idx[elt][pos], cont.r_idx[elt][pos] = gen_comprs_fast(cont.N/2, elt, 2*cont.kp_wids[1], pos, 0)
			cont.m_idx_l[elt][pos], cont.r_idx_l[elt][pos] = gen_comprs_fast(cont.N/2, elt, 2*cont.kp_wids[1], pos, 1)
			for k := range cont.r_idx[elt][pos] {
				rotations = append(rotations, k)
			}
			for k := range cont.r_idx_l[elt][pos] {
				rotations = append(rotations, k)
			}
			for k := range cont.m_idx[elt][pos] {
				rotations = append(rotations, k)
			}
			for k := range cont.m_idx_l[elt][pos] {
				rotations = append(rotations, k)
			}
		}
	case "Imagenet_final": // Generate ext_idx for extracting valid values from conv with "same" padding
		iter = 2 // since we use half padding, i.e., lower part is all zero
		for i, elt := range cont.in_wids {
			cont.ext_idx[elt] = make([][]int, iter)
			for ul := 0; ul < iter; ul++ {
				cont.ext_idx[elt][ul] = gen_keep_vec(cont.N/2, elt, cont.kp_wids[i], ul)
			}
			cont.r_idx[elt] = make([]map[int][]int, 4)
			cont.r_idx_l[elt] = make([]map[int][]int, 4)
			cont.m_idx[elt] = make([]map[int][]int, 4)
			cont.m_idx_l[elt] = make([]map[int][]int, 4)

			if i == 0 {
				for pos := 0; pos < 4; pos += 2 {
					cont.m_idx[elt][pos], cont.r_idx[elt][pos] = gen_comprs_fast(cont.N/2, elt, cont.kp_wids[i], pos, 0)
					cont.m_idx_l[elt][pos], cont.r_idx_l[elt][pos] = gen_comprs_fast(cont.N/2, elt, cont.kp_wids[i], pos, 1)
					for k := range cont.r_idx[elt][pos] {
						rotations = append(rotations, k)
					}
					for k := range cont.m_idx[elt][pos] {
						rotations = append(rotations, k)
					}
					for k := range cont.r_idx_l[elt][pos] {
						rotations = append(rotations, k)
					}
					for k := range cont.m_idx_l[elt][pos] {
						rotations = append(rotations, k)
					}
				}
			}
		}
	case "Imagenet_final_fast": // Generate ext_idx for extracting valid values from conv with "same" padding
		iter = 2 // since we use full padding
		for i, elt := range cont.in_wids {
			cont.ext_idx[elt] = make([][]int, iter)
			for ul := 0; ul < iter; ul++ {
				cont.ext_idx[elt][ul] = gen_keep_vec(cont.N/2, elt, cont.kp_wids[i], ul)
			}
			cont.r_idx[elt] = make([]map[int][]int, 4)
			cont.r_idx_l[elt] = make([]map[int][]int, 4)
			cont.m_idx[elt] = make([]map[int][]int, 4)
			cont.m_idx_l[elt] = make([]map[int][]int, 4)

			if i == 0 {
				for pos := 0; pos < 2; pos += 1 {
					cont.r_idx[elt][pos] = gen_comprs_full(cont.N/2, elt, 2*cont.kp_wids[1], pos, 0)
					cont.r_idx_l[elt][pos] = gen_comprs_full(cont.N/2, elt, 2*cont.kp_wids[1], pos, 1)
					for k := range cont.r_idx[elt][pos] {
						rotations = append(rotations, k)
					}
					for k := range cont.r_idx_l[elt][pos] {
						rotations = append(rotations, k)
					}
				}
			}
		}
	default:
		panic("Wrong kinds!")
	}
	rotations = removeDuplicateInt(rotations)

	// Scheme context and keys for evaluation (no Boot)
	kgen := ckks.NewKeyGenerator(cont.params)
	sk, _ := kgen.GenKeyPairSparse(btpParams.H)
	rlk := kgen.GenRelinearizationKey(sk, 2)
	fmt.Println("Num Rotations: ", len(rotations))
	var rotkeys *rlwe.RotationKeySet
	// additional optimization with 2 P
	if kind == "BL_Conv" {
		new_params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
			LogN: logN,
			Q:    cont.params.Q(),
			P:    []uint64{0x1fffffffffe00001, 0x1fffffffffc80001}, // Pi 61
			// Pi 61
			Sigma:    rlwe.DefaultSigma,
			LogSlots: logN - 1,
			Scale:    float64(1 << 30),
		})
		if err != nil {
			panic(err)
		}
		new_kgen := ckks.NewKeyGenerator(new_params)
		rotkeys = new_kgen.GenRotationKeysForRotations(rotations, true, sk)
		cont.pack_evaluator = ckks.NewEvaluator(new_params, rlwe.EvaluationKey{Rlk: rlk, Rtks: rotkeys})

		cont.encoder = ckks.NewEncoder(cont.params)
		cont.decryptor = ckks.NewDecryptor(cont.params, sk)
		cont.encryptor = ckks.NewEncryptor(cont.params, sk)
		cont.evaluator = ckks.NewEvaluator(cont.params, rlwe.EvaluationKey{Rlk: rlk})
	} else {
		rotkeys = kgen.GenRotationKeysForRotations(rotations, false, sk)
		cont.encoder = ckks.NewEncoder(cont.params)
		cont.decryptor = ckks.NewDecryptor(cont.params, sk)
		cont.encryptor = ckks.NewEncryptor(cont.params, sk)
		cont.evaluator = ckks.NewEvaluator(cont.params, rlwe.EvaluationKey{Rlk: rlk, Rtks: rotkeys})
	}

	if !(kind == "BL_Conv") {
		// we use smaller keys for rotations for pack_ctxts
		new_params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
			LogN: logN,
			Q:    cont.params.Q(),
			P:    []uint64{0x1fffffffffe00001}, // Pi 61
			// Pi 61
			Sigma:    rlwe.DefaultSigma,
			LogSlots: logN - 1,
			Scale:    float64(1 << 30),
		})
		if err != nil {
			panic(err)
		}
		new_kgen := ckks.NewKeyGenerator(new_params)
		new_encoder := ckks.NewEncoder(new_params)

		cont.pl_idx, cont.pack_evaluator = gen_idxNlogs(0, new_kgen, sk, new_encoder, new_params)
	}

	if boot {
		fmt.Println("Generating bootstrapping keys...")
		rotations = btpParams.RotationsForBootstrapping(cont.params.LogSlots())
		rotkeys = kgen.GenRotationKeysForRotations(rotations, true, sk)
		btpKey := ckks.BootstrappingKey{Rlk: rlk, Rtks: rotkeys}

		if kind == "BL_Conv" {
			if cont.btp, err = ckks.NewBootstrapper(cont.params, btpParams, btpKey); err != nil {
				panic(err)
			}
		} else {
			if cont.btp, err = ckks.NewBootstrapper_mod(cont.params, btpParams, btpKey); err != nil {
				panic(err)
			}
		}
		fmt.Printf("Done in %s \n", time.Since(cont_start))
	}

	return &cont
}

func main() {
	// st, _ := strconv.Atoi(os.Args[1])
	// end, _ := strconv.Atoi(os.Args[2])
	// ker, _ := strconv.Atoi(os.Args[3])
	// testImagenet_final_fast_in(st, end, ker)

	// Test Conv Boot & NoBoot FINAL!
	batchs := [5]int{4, 16, 64, 256, 1024}
	widths := [5]int{128, 64, 32, 16, 8}

	test_name := os.Args[1]
	ker_wid, _ := strconv.Atoi(os.Args[2])
	i_batch, _ := strconv.Atoi(os.Args[3])
	num_tests, _ := strconv.Atoi(os.Args[4])

	if !((ker_wid == 3) || (ker_wid == 5) || (ker_wid == 7)) {
		panic("Wrong kernel wid (not in 3,5,7)")
	}
	var boot, resnet bool
	switch test_name {
	case "conv":
		boot = false
		resnet = false
		if (num_tests > 10) || (i_batch > 3) {
			panic("Too many tests (>10) or too many batch index (>3)")
		}
	case "convReLU":
		boot = true
		resnet = false
		if (num_tests > 10) || (i_batch > 3) {
			panic("Too many tests (>10) or too many batch index (>3)")
		}
	case "resnet":
		resnet = true
	default:
		panic("wrong test type")
	}

	if resnet {
		// // latest version for resnet crop cifar10
		ker_wid, _ := strconv.Atoi(os.Args[2])
		depth, _ := strconv.Atoi(os.Args[3])
		wide_case, _ := strconv.Atoi(os.Args[4])
		test_num, _ := strconv.Atoi(os.Args[5])
		cf100, _ := strconv.ParseBool(os.Args[6])

		debug := false // if turned on, it shows all intermediate input
		if wide_case == 1 {
			testResNet_crop_fast_in(0, test_num, ker_wid, depth, debug, cf100)
		} else {
			testResNet_crop_fast_wide_in(0, test_num, ker_wid, depth, wide_case, debug, cf100)
		}

	} else {
		if boot {
			fmt.Println("Convolution followed by ReLU (& Bootstrapping) test start!")
		} else {
			fmt.Println("Convolution test start! (No Bootstrapping)")
		}
		fmt.Println("Ker: ", ker_wid, "batches: ", batchs[i_batch], "widths: ", widths[i_batch])

		fmt.Println("Base Line start.")
		testConv_BL_in(batchs[i_batch], widths[i_batch], ker_wid, num_tests, boot)

		fmt.Println("Ours start.")
		testConv_in(batchs[i_batch], widths[i_batch], ker_wid, num_tests, boot)
	}
}

func printDebugCfs(params ckks.Parameters, ciphertext *ckks.Ciphertext, valuesWant []float64, decryptor ckks.Decryptor, encoder ckks.Encoder) (valuesTest []float64) {
	total_size := make([]int, 15)
	valuesTest = encoder.DecodeCoeffs(decryptor.DecryptNew(ciphertext))

	fmt.Println()
	fmt.Printf("Level: %d (logQ = %d)\n", ciphertext.Level(), params.LogQLvl(ciphertext.Level()))
	fmt.Printf("Scale: 2^%f\n", math.Log2(ciphertext.Scale))
	fmt.Printf("ValuesTest:")
	for i := range total_size {
		fmt.Printf("%6.10f, ", valuesTest[i])
	}
	fmt.Printf("... \n")
	fmt.Printf("ValuesWant:")
	for i := range total_size {
		fmt.Printf("%6.10f, ", valuesWant[i])
	}
	fmt.Printf("... \n")

	valuesWantC := make([]complex128, len(valuesWant))
	valuesTestC := make([]complex128, len(valuesTest))
	for i := range valuesWantC {
		valuesWantC[i] = complex(valuesWant[i], 0)
		valuesTestC[i] = complex(valuesTest[i], 0)
	}
	precStats := ckks.GetPrecisionStats(params, encoder, nil, valuesWantC[:params.Slots()], valuesTestC[:params.Slots()], params.LogSlots(), 0)

	fmt.Println(precStats.String())

	precStats = ckks.GetPrecisionStats(params, encoder, nil, valuesWantC[params.Slots():], valuesTestC[params.Slots():], params.LogSlots(), 0)

	fmt.Println(precStats.String())
	fmt.Println()

	return
}

func printDebugCfsPlain(valuesTest, valuesWant []float64) {
	total_size := make([]int, 10)

	fmt.Printf("ValuesTest:")
	for i := range total_size {
		fmt.Printf("%6.10f, ", valuesTest[i])
	}
	fmt.Printf("... \n")
	fmt.Printf("ValuesWant:")
	for i := range total_size {
		fmt.Printf("%6.10f, ", valuesWant[i])
	}
	fmt.Printf("... \n")

	valuesWantC := make([]complex128, len(valuesWant))
	valuesTestC := make([]complex128, len(valuesTest))
	for i := range valuesWantC {
		valuesWantC[i] = complex(valuesWant[i], 0)
		valuesTestC[i] = complex(valuesTest[i], 0)
	}
	precStats := ckks.GetPrecisionStatsPlain(valuesWantC, valuesTestC, len(valuesWantC), 0)
	fmt.Println(precStats.String())
	fmt.Println()
}

func printDebug(params ckks.Parameters, ciphertext *ckks.Ciphertext, valuesWant []complex128, decryptor ckks.Decryptor, encoder ckks.Encoder) (valuesTest []complex128) {
	total_size := make([]int, 15)
	valuesTest = encoder.Decode(decryptor.DecryptNew(ciphertext), params.LogSlots())

	fmt.Println()
	fmt.Printf("Level: %d (logQ = %d)\n", ciphertext.Level(), params.LogQLvl(ciphertext.Level()))
	fmt.Printf("Scale: 2^%f\n", math.Log2(ciphertext.Scale))
	fmt.Printf("ValuesTest:")
	for i := range total_size {
		fmt.Printf("%6.5f, ", real(valuesTest[i]))
	}
	fmt.Printf("... \n")
	fmt.Printf("ValuesWant:")
	for i := range total_size {
		fmt.Printf("%6.5f, ", real(valuesWant[i]))
	}
	fmt.Printf("... \n")

	precStats := ckks.GetPrecisionStats(params, encoder, nil, valuesWant, valuesTest, params.LogSlots(), 0)

	fmt.Println(precStats.String())
	fmt.Println()

	return
}

// print slice (back and forth prt_size elements)
// scaled by 2N
func prt_vecc(vec []complex128) {
	prt_size := 5
	total_size := len(vec)

	if total_size <= 2*prt_size {
		fmt.Print("    [")
		for i := 0; i < total_size; i++ {
			fmt.Printf("  %4.5f + %1.2f i, ", real(vec[i]), imag(vec[i]))
		}
		fmt.Print(" ]\n")
	} else {
		fmt.Print("    [")
		for i := 0; i < prt_size; i++ {
			fmt.Printf(" %4.5f + %1.2f i, ", real(vec[i]), imag(vec[i]))
		}
		fmt.Printf(" ...,")
		for i := total_size - prt_size; i < total_size; i++ {
			fmt.Printf(" %4.5f + %1.2f i, ", real(vec[i]), imag(vec[i]))
		}
		fmt.Print(" ]\n")
	}
	fmt.Println()
}

// print slice (back and forth prt_size elements)
func prt_vec(vec []float64) {
	prt_size := 32
	total_size := len(vec)

	if total_size <= 2*prt_size {
		fmt.Print("    [")
		for i := 0; i < total_size; i++ {
			fmt.Printf("  %4.4f, ", vec[i])
		}
		fmt.Print(" ]\n")
	} else {
		fmt.Print("    [")
		for i := 0; i < prt_size; i++ {
			fmt.Printf(" %4.4f, ", vec[i])
		}
		fmt.Printf(" ...,")
		for i := total_size - prt_size; i < total_size; i++ {
			fmt.Printf(" %4.4f", vec[i])
		}
		fmt.Print(" ]\n")
	}
	fmt.Println()
}

// vec = (B0 input, B1 input, ... ) format for BaseLine Test, each batch is sqr-sized
// print (i,j)-th position in [batches], only shows (show, show) entries show = 0 : print all
func prt_mat_BL(vec []complex128, batch, show int) {
	in_wid := int(math.Sqrt(float64(len(vec) / batch)))
	tmp := make([]float64, batch)
	for i := 1; i < in_wid+1; i++ {
		for j := 1; j < in_wid+1; j++ {
			if (show == 0) || (((i <= show) || (i+show > in_wid)) && ((j <= show) || (j+show > in_wid))) {
				fmt.Printf("(%d, %d): ", i, j)
				for b := 0; b < batch; b++ {
					tmp[b] = real(vec[in_wid*in_wid*b+(i-1)*in_wid+(j-1)])
				}
				prt_vec(tmp)
			}
		}
	}
}

// vec = arrgvec with batch batches, each batch is sqr-sized
// print (i,j)-th position in [batches], only shows (show, show) entries show = 0 : print all
func prt_mat(vec []float64, batch, show int) {
	mat_size := len(vec) / batch
	row := int(math.Sqrt(float64(mat_size)))
	j, k := 1, 1
	for i := 0; i < len(vec); i += batch {
		if (show == 0) || (((j <= show) || (j > row-show)) && ((k <= show) || (k > (row - show)))) {
			fmt.Printf("(%d, %d): ", j, k)
			prt_vec(vec[i : i+batch])
		}
		k++
		if k*k > mat_size {
			k = 1
			j++
		}
	}
}

// vec = arrgvec with batch batches, each batch is sqr-sized
// print (i,j)-th position in [batches], only shows (show, show) entries show = 0 : print all
func prt_mat_norm(vec []float64, batch, norm, show int, half bool) {
	mat_size := len(vec) / batch
	row := int(math.Sqrt(float64(mat_size)))
	if half {
		row = row / 2
	}
	tmp := make([]float64, batch/norm)
	j, k := 1, 1
	for i := 0; i < len(vec); i += batch {
		if (show == 0) || (((j <= show) || ((j > row-show) && (j <= row))) && ((k <= show) || ((k > row-show) && (k <= row)))) {
			fmt.Printf("(%d, %d): ", j, k)
			for idx := range tmp {
				tmp[idx] = vec[i+norm*idx]
			}
			prt_vec(tmp)
		}
		k++
		if k*k > mat_size {
			k = 1
			j++
		}
	}
}

// vec = arrgvec with batch batches, each batch is sqr-sized
// print (i,j)-th position in [batches], only shows (show, show) entries show = 0 : print all
// input is strided with steps, read from start
func prt_mat_norm_step(vec []float64, batch, norm, step, start, show int, half bool) {
	mat_size := len(vec) / batch
	row := int(math.Sqrt(float64(mat_size)))
	if half {
		row = row / 2
	}
	tmp := make([]float64, batch/norm)
	j, k := 1, 1
	for i := 0; i < len(vec); i += batch {
		if (show == 0) || (((j <= show*step) || ((j > row-show*step) && (j <= row))) && ((k <= show*step) || ((k > row-show*step) && (k <= row)))) {
			if ((j-start)%step == 0) && ((k-start)%step == 0) {
				fmt.Printf("(%d, %d): ", (j-start)/step+1, (k-start)/step+1)
				for idx := range tmp {
					tmp[idx] = vec[i+norm*idx]
				}
				prt_vec(tmp)
			}
		}
		k += 1
		if k*k > mat_size {
			k = 1
			j += 1
		}
	}
}

// only (sj,sk) element in all batches
func prt_mat_one(vec []float64, batch, sj, sk int) (out []float64) {
	mat_size := len(vec) / batch
	j, k := 1, 1
	for i := 0; i < len(vec); i += batch {
		if (j == sj) && (k == sk) {
			fmt.Print(vec[i : i+batch])
			out = vec[i : i+batch]
		}
		k++
		if k*k > mat_size {
			k = 1
			j++
		}
	}
	return out
}

// only (sj,sk) element in all batches
func prt_mat_one_norm(vec []float64, batch, norm, sj, sk int) (out []float64) {
	mat_size := len(vec) / batch
	tmp := make([]float64, batch/norm)
	j, k := 1, 1
	for i := 0; i < len(vec); i += batch {
		if (j == sj) && (k == sk) {
			for idx := range tmp {
				tmp[idx] = vec[i+norm*idx]
			}
			prt_vec(tmp)
			out = tmp
		}
		k++
		if k*k > mat_size {
			k = 1
			j++
		}
	}
	return out
}

// only out_num, (1,1) element in all batches (1,0,0,0,0,0,0,0,2,0,0,0,0,0,...)
func prt_mat_one_BL(vec []complex128, max_bat, out_num int) (out []float64) {
	mat_size := len(vec) / max_bat
	out = make([]float64, out_num)

	for i := range out {
		out[i] = real(vec[i*mat_size*8])
	}

	return out
}

// only out_num, (1,1) element in all batches (1,0,0,0,0,0,0,0,2,0,0,0,0,0,...)
func prt_mat_one_BL_img(vec []complex128, max_bat, out_num int) (out []float64) {
	mat_size := len(vec) / max_bat
	out = make([]float64, out_num)

	for i := range out {
		out[i] = real(vec[i*mat_size])
	}

	return out
}

func check(e error) {
	if e != nil {
		panic(e)
	}
}

func readTxt(name_file string, size int) (input []float64) {

	file, err := os.Open(name_file)
	check(err)
	scanner := bufio.NewScanner(file)
	scanner.Split(bufio.ScanWords)

	for scanner.Scan() {
		add, _ := strconv.ParseFloat(scanner.Text(), 64)
		input = append(input, add)
	}
	file.Close()
	// fmt.Print(input)

	if (size != 0) && (len(input) != size) {
		panic("input size inconsistent!")
	}

	return input
}

func writeTxt(name_file string, input []float64) {
	file, err := os.OpenFile(name_file, os.O_TRUNC|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Fatalf("failed creating file: %s", err)
	}

	datawriter := bufio.NewWriter(file)
	for _, data := range input {
		_, _ = datawriter.WriteString(strconv.FormatFloat(data, 'e', -1, 64) + "\n")
	}

	datawriter.Flush()
	file.Close()
}

func prep_Input(input []float64, raw_in_wid, in_wid, N, norm int, trans, printResult bool) (out []float64) {
	out = make([]float64, N)
	batch := N / (in_wid * in_wid)
	k := 0

	if trans {
		for i := 0; i < in_wid/2; i++ {
			for j := 0; j < in_wid/2; j++ {
				for b := 0; b < batch/norm; b++ {
					if (i < raw_in_wid) && (j < raw_in_wid) {
						out[(2*i+1)*in_wid*batch+(2*j+1)*batch+b*norm] = input[k]
						k++
					}
				}
			}
		}
	} else {
		for i := 0; i < in_wid; i++ {
			for j := 0; j < in_wid; j++ {
				for b := 0; b < batch/norm; b++ {
					if (i < raw_in_wid) && (j < raw_in_wid) {
						out[i*in_wid*batch+j*batch+b*norm] = input[k]
						k++
					}
				}
			}
		}
	}

	if printResult {
		fmt.Println("Input matrix: ")
		prt_mat(out, batch, 3)
	}

	return out
}

func removeDuplicateInt(intSlice []int) []int {
	allKeys := make(map[int]bool)
	list := []int{}
	for _, item := range intSlice {
		if _, value := allKeys[item]; !value {
			allKeys[item] = true
			list = append(list, item)
		}
	}
	return list
}

// only returns valid values from
func post_process(in_cfs []float64, raw_in_wid, in_wid int) []float64 {
	batch := len(in_cfs) / (in_wid * in_wid)
	out := make([]float64, raw_in_wid*raw_in_wid*batch)

	for i := 0; i < raw_in_wid; i++ {
		for j := 0; j < raw_in_wid; j++ {
			for b := 0; b < batch; b++ {
				out[i*raw_in_wid*batch+batch*j+b] = in_cfs[i*in_wid*batch+batch*j+b]
			}
		}
	}

	return out
}

// from 8*8 with 1 pad -> 7*7
func post_trim_BL(in_vals []complex128, raw_in_wid, in_wid int) []float64 {
	batch := len(in_vals) / (in_wid * in_wid)
	out := make([]float64, raw_in_wid*raw_in_wid*batch)

	for b := 0; b < batch; b++ {
		for i := 0; i < raw_in_wid; i++ {
			for j := 0; j < raw_in_wid; j++ {
				out[b*raw_in_wid*raw_in_wid+i*raw_in_wid+j] = real(in_vals[b*in_wid*in_wid+i*in_wid+j])
			}
		}
	}

	return out
}

// only returns valid values from
func post_process_BL(in_vals []float64, raw_in_wid int) []float64 {
	batch := len(in_vals) / (raw_in_wid * raw_in_wid)
	out := make([]float64, raw_in_wid*raw_in_wid*batch)

	for i := 0; i < raw_in_wid; i++ {
		for j := 0; j < raw_in_wid; j++ {
			for b := 0; b < batch; b++ {
				out[i*raw_in_wid*batch+j*batch+b] = in_vals[b*raw_in_wid*raw_in_wid+i*raw_in_wid+j]
			}
		}
	}

	return out
}
