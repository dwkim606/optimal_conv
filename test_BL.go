package main

import (
	"fmt"
	"math"
	"strconv"
	"time"

	"github.com/dwkim606/test_lattigo/ckks"
)

// BaseLine Conv without boot, Assume full batch with Po2 in_wid & N
// Normal Conv without output modification (e.g., trimming or expanding)
// Input does not need padding
// use imaginary part to save before boot
func testConv_BL_in(real_batch, in_wid, ker_wid, total_test_num int, boot bool) {
	in_kind := "Conv"
	test_dir := "test_conv_data/"
	if (in_kind != "TransConv") && (in_kind != "Conv") && (in_kind != "StrConv") {
		panic("Wrong in_kind!")
	}
	pad := ker_wid / 2
	raw_in_wid := in_wid - pad // = in_wid

	in_size := in_wid * in_wid
	ker_size := ker_wid * ker_wid
	slots := real_batch / 2 * in_size
	log_slots := 0
	for ; (1 << log_slots) < slots; log_slots++ {
	}
	out_batch := real_batch
	if in_kind == "TransConv" {
		out_batch = real_batch / 4
	}
	kp_wid := 0
	kind := "BL_" + in_kind
	in_batch := real_batch

	// generate Context: params, Keys, rotations, general plaintexts
	cont := newContext(log_slots+1, ker_wid, []int{in_wid}, []int{kp_wid}, boot, kind)
	fmt.Println("vec size: log2 = ", cont.logN)
	fmt.Println("raw input width: ", raw_in_wid)
	fmt.Println("kernel width: ", ker_wid)
	fmt.Println("num batches in & out: ", real_batch, ", ", out_batch)

	for test_iter := 0; test_iter < total_test_num; test_iter++ {
		fmt.Println(test_iter+1, "-th iter...start")
		input := readTxt(test_dir+"test_conv"+strconv.Itoa(ker_wid)+"_batch_"+strconv.Itoa(in_batch)+"_in_"+strconv.Itoa(test_iter)+".csv", raw_in_wid*raw_in_wid*in_batch)
		ker_in := readTxt(test_dir+"test_conv"+strconv.Itoa(ker_wid)+"_batch_"+strconv.Itoa(in_batch)+"_ker_"+strconv.Itoa(test_iter)+".csv", in_batch*in_batch*ker_wid*ker_wid)
		bn_a := readTxt(test_dir+"test_conv"+strconv.Itoa(ker_wid)+"_batch_"+strconv.Itoa(in_batch)+"_bna_"+strconv.Itoa(test_iter)+".csv", in_batch)
		bn_b := readTxt(test_dir+"test_conv"+strconv.Itoa(ker_wid)+"_batch_"+strconv.Itoa(in_batch)+"_bnb_"+strconv.Itoa(test_iter)+".csv", in_batch)
		var real_out []float64
		if boot {
			real_out = readTxt(test_dir+"test_conv"+strconv.Itoa(ker_wid)+"_batch_"+strconv.Itoa(in_batch)+"_reluout_"+strconv.Itoa(test_iter)+".csv", raw_in_wid*raw_in_wid*in_batch)
		} else {
			real_out = readTxt(test_dir+"test_conv"+strconv.Itoa(ker_wid)+"_batch_"+strconv.Itoa(in_batch)+"_out_"+strconv.Itoa(test_iter)+".csv", raw_in_wid*raw_in_wid*in_batch)
		}

		pad_input1 := make([]float64, in_wid*in_wid*real_batch/2)
		pad_input2 := make([]float64, in_wid*in_wid*real_batch/2)
		for i := 0; i < raw_in_wid; i++ {
			for j := 0; j < raw_in_wid; j++ {
				for b := 0; b < real_batch/2; b++ {
					pad_input1[b+j*real_batch/2+i*real_batch/2*in_wid] = input[b+j*real_batch+i*real_batch*raw_in_wid]
					pad_input2[b+j*real_batch/2+i*real_batch/2*in_wid] = input[b+real_batch/2+j*real_batch+i*real_batch*raw_in_wid]
				}
			}
		}

		bn_a_sep := make([][]float64, 2)
		bn_b_sep := make([][]float64, 2)
		zeros := make([]float64, real_batch/2)
		for out := 0; out < 2; out++ {
			bn_a_sep[out] = make([]float64, real_batch/2)
			bn_b_sep[out] = make([]float64, real_batch/2)
			for i := 0; i < real_batch/2; i++ {
				bn_a_sep[out][i] = bn_a[i+out*real_batch/2]
				bn_b_sep[out][i] = bn_b[i+out*real_batch/2]
			}
		}

		ker_in_sep := make([][][]float64, 2) // number of output ctxts
		for out := 0; out < 2; out++ {
			ker_in_sep[out] = make([][]float64, 2) // number of input ctxts
			for in := 0; in < 2; in++ {
				ker_in_sep[out][in] = make([]float64, len(ker_in)/(2*2))
				for k := 0; k < ker_size; k++ {
					for i := 0; i < real_batch/2; i++ { // in
						for j := 0; j < real_batch/2; j++ { // out
							ker_in_sep[out][in][k*real_batch*real_batch/4+i*real_batch/2+j] =
								ker_in[k*real_batch*real_batch+(i+in*real_batch/2)*real_batch+out*real_batch/2+j] // [i][4*j]
						}
					}
				}
			}
		}

		input1_rs := reshape_input_BL(pad_input1, in_wid)
		input2_rs := reshape_input_BL(pad_input2, in_wid)
		start := time.Now()
		ct_input1 := cont.encryptor.EncryptNew(cont.encoder.EncodeAtLvlNew(cont.ECD_LV, input1_rs, cont.logN-1))
		ct_input2 := cont.encryptor.EncryptNew(cont.encoder.EncodeAtLvlNew(cont.ECD_LV, input2_rs, cont.logN-1))
		fmt.Printf("Encryption done in %s \n", time.Since(start))

		start_eval := time.Now()
		ct_res := make([]*ckks.Ciphertext, 2)
		for pos := 0; pos < 2; pos++ {
			ct_res[pos] = cont.evaluator.AddNew(evalConv_BN_BL_test(cont, ct_input1, ker_in_sep[pos][0], bn_a_sep[pos], bn_b_sep[pos], in_wid, ker_wid, real_batch/2, real_batch/2, 0, 1, pad, false, false),
				evalConv_BN_BL_test(cont, ct_input2, ker_in_sep[pos][1], bn_a_sep[pos], zeros, in_wid, ker_wid, real_batch/2, real_batch/2, 0, 1, pad, false, false))
		}
		fmt.Printf("Evaluation total done in %s \n", time.Since(start_eval))

		if boot {
			img_eval := time.Now()
			for pos := 0; pos < 2; pos++ {
				ct_res[pos] = cont.evaluator.AddNew(cont.pack_evaluator.ConjugateNew(ct_res[pos]), ct_res[pos])
				if pos == 1 {
					ct_res[pos] = cont.evaluator.MultByiNew(ct_res[pos])
				}
				// fmt.Println("before Boot: LV = ", ct_res[pos].Level(), " Scale = ", math.Log2(ct_res[pos].Scale))
			}
			ct_res[0] = cont.evaluator.AddNew(ct_res[0], ct_res[1])
			img_eval_part := time.Since(img_eval)

			pos := 0
			alpha := 0.0
			pow := 4.0
			ct_res[pos].Scale = ct_res[pos].Scale * math.Pow(2, pow+2) // +1 for 1/2 * (a + conj(a)), +1 for 1/2 after boot
			// vals_preB := cont.encoder.Decode(cont.decryptor.DecryptNew(ct_res[pos]), cont.logN-1)
			fmt.Println("\n ========= Bootstrapping... (original) ========= ")
			start_boot := time.Now()
			// fmt.Println("initial (before boot): LV = ", ct_res[pos].Level(), " Scale = ", math.Log2(ct_res[pos].Scale))
			ct_boot := cont.btp.Bootstrapp(ct_res[pos])
			fmt.Printf("Boot Done in %s \n", time.Since(start_boot))

			// fmt.Println("after Boot: LV = ", ct_boot.Level(), " Scale = ", math.Log2(ct_boot.Scale))

			// Only for checking the correctness (for Boot)
			// vals_postB := printDebug(cont.params, ct_boot, vals_preB, cont.decryptor, cont.encoder)
			// vals_relu := make([]complex128, len(vals_postB))
			// for i, elt := range vals_postB {
			// 	vals_relu[i] = complex((math.Max(0, real(elt))+math.Min(0, real(elt)*alpha))*math.Pow(2, pow), 0)
			// }
			img_eval = time.Now()
			pl_scale := ckks.NewPlaintext(cont.params, ct_boot.Level(), math.Pow(2, 30)*float64(cont.params.Q()[14])*float64(cont.params.Q()[13])/ct_boot.Scale)
			val_scale := make([]complex128, cont.N/2)
			for i := range val_scale {
				val_scale[i] = complex(1.0, 0) // val_scale[i] = complex(1.0/math.Pow(2, pow), 0)
			}
			cont.encoder.EncodeNTT(pl_scale, val_scale, cont.logN-1)
			cont.evaluator.Mul(ct_boot, pl_scale, ct_boot)
			cont.evaluator.Rescale(ct_boot, cont.params.Scale(), ct_boot)

			ct_iboot := cont.pack_evaluator.ConjugateNew(ct_boot)
			ct_res[0] = cont.evaluator.AddNew(ct_boot, ct_iboot)
			ct_res[1] = cont.evaluator.DivByiNew(cont.evaluator.SubNew(ct_boot, ct_iboot))
			fmt.Printf("Imaginary packing and unpacking done in %s \n", img_eval_part+time.Since(img_eval))

			start = time.Now()
			for pos := 0; pos < 2; pos++ {
				// fmt.Println("before relu: LV = ", ct_res[pos].Level(), " Scale = ", math.Log2(ct_res[pos].Scale))
				// fmt.Println("after Rescale: LV = ", ct_boot.Level(), " Scale = 2^", math.Log2(ct_boot.Scale))
				ct_res[pos] = evalReLU(cont.params, cont.evaluator, ct_res[pos], alpha)
				cont.evaluator.MulByPow2(ct_res[pos], int(pow), ct_res[pos])
				cont.evaluator.SetScale(ct_res[pos], cont.params.Scale())
				// printDebug(cont.params, ct_res[pos], vals_relu, cont.decryptor, cont.encoder)
			}
			fmt.Printf("Relu Done in %s \n", time.Since(start))
		}

		// fmt.Println()
		// fmt.Println("===============  DECRYPTION  ===============")
		// fmt.Println()
		start = time.Now()
		vals_tmp1 := cont.encoder.Decode(cont.decryptor.DecryptNew(ct_res[0]), log_slots)
		vals_tmp2 := cont.encoder.Decode(cont.decryptor.DecryptNew(ct_res[1]), log_slots)
		fmt.Printf("Decryption Done in %s \n", time.Since(start))

		test_out := post_trim_BL(vals_tmp1, raw_in_wid, in_wid)
		test_out = append(test_out, post_trim_BL(vals_tmp2, raw_in_wid, in_wid)...)
		test_out = post_process_BL(test_out, raw_in_wid)

		printDebugCfsPlain(test_out, real_out)
	}
}
