package main

import (
	"fmt"
	"strconv"
	"time"

	"github.com/dwkim606/test_lattigo/ckks"
)

// Fast Conv without boot, Assume full batch with Po2 in_wid & N
// Normal Conv without output modification (e.g., trimming or expanding)
// Assume that the input is 0 padded according to kernel size: only in_wid - (ker_wid-1)/2 elements in row and columns are nonzero
// Also support non-full batching case
func testConv_in(in_batch, in_wid, ker_wid, total_test_num int, boot bool) {
	kind := "Conv"
	printResult := false
	raw_in_batch := in_batch         // same as python
	raw_in_wid := in_wid - ker_wid/2 // same as python
	norm := in_batch / raw_in_batch
	test_dir := "test_conv_data/"
	pow := 4.0

	// set basic variables for above input variables
	kp_wid, out_batch, logN, trans := set_Variables(in_batch, raw_in_wid, in_wid, ker_wid, kind)
	raw_out_batch := out_batch / norm

	// generate Context: params, Keys, rotations, general plaintexts
	cont := newContext(logN, ker_wid, []int{in_wid}, []int{kp_wid}, boot, kind)
	fmt.Println("vec size: log2 = ", cont.logN)
	fmt.Println("raw input width: ", raw_in_wid)
	fmt.Println("kernel width: ", ker_wid)
	fmt.Println("num raw batches in & out: ", raw_in_batch, ", ", raw_out_batch)

	for test_iter := 0; test_iter < total_test_num; test_iter++ {
		fmt.Println(test_iter+1, "-th iter...start")
		raw_input := readTxt(test_dir+"test_conv"+strconv.Itoa(ker_wid)+"_batch_"+strconv.Itoa(in_batch)+"_in_"+strconv.Itoa(test_iter)+".csv", raw_in_wid*raw_in_wid*raw_in_batch)
		ker_in := readTxt(test_dir+"test_conv"+strconv.Itoa(ker_wid)+"_batch_"+strconv.Itoa(in_batch)+"_ker_"+strconv.Itoa(test_iter)+".csv", raw_in_batch*raw_out_batch*ker_wid*ker_wid)
		bn_a := readTxt(test_dir+"test_conv"+strconv.Itoa(ker_wid)+"_batch_"+strconv.Itoa(in_batch)+"_bna_"+strconv.Itoa(test_iter)+".csv", raw_out_batch)
		bn_b := readTxt(test_dir+"test_conv"+strconv.Itoa(ker_wid)+"_batch_"+strconv.Itoa(in_batch)+"_bnb_"+strconv.Itoa(test_iter)+".csv", raw_out_batch)

		// input encryption
		input := prep_Input(raw_input, raw_in_wid, in_wid, cont.N, norm, trans, printResult)
		start := time.Now()
		plain_tmp := ckks.NewPlaintext(cont.params, cont.ECD_LV, cont.params.Scale()) // contain plaintext values
		cont.encoder.EncodeCoeffs(input, plain_tmp)
		ctxt_input := cont.encryptor.EncryptNew(plain_tmp)
		fmt.Printf("Encryption done in %s \n", time.Since(start))

		// Kernel Prep & Conv (+BN) Evaluation
		var ct_result *ckks.Ciphertext
		if boot {
			ct_result = evalConv_BNRelu_new(cont, ctxt_input, ker_in, bn_a, bn_b, 0.0, pow, in_wid, kp_wid, ker_wid, raw_in_batch, raw_out_batch, norm, 0, 0, 2, 0, kind, false, false)
		} else {
			ct_result = evalConv_BN(cont, ctxt_input, ker_in, bn_a, bn_b, in_wid, ker_wid, raw_in_batch, raw_out_batch, norm, float64(1<<30), trans)
		}

		start = time.Now()
		cont.decryptor.Decrypt(ct_result, plain_tmp)
		cfs_tmp := cont.encoder.DecodeCoeffs(plain_tmp)
		fmt.Printf("Decryption Done in %s \n", time.Since(start))

		test_out := post_process(cfs_tmp, raw_in_wid, in_wid)
		var real_out []float64
		if boot {
			real_out = readTxt(test_dir+"test_conv"+strconv.Itoa(ker_wid)+"_batch_"+strconv.Itoa(in_batch)+"_reluout_"+strconv.Itoa(test_iter)+".csv", raw_in_wid*raw_in_wid*raw_in_batch)
		} else {
			real_out = readTxt(test_dir+"test_conv"+strconv.Itoa(ker_wid)+"_batch_"+strconv.Itoa(in_batch)+"_out_"+strconv.Itoa(test_iter)+".csv", raw_in_wid*raw_in_wid*raw_in_batch)
		}

		printDebugCfsPlain(test_out, real_out)
	}

}

func testResNet_crop_sparse(st, end, ker_wid, depth int, debug, cf100 bool) {
	// init_batch fixed to 16
	ker_name := "ker" + strconv.Itoa(ker_wid)
	weight_dir := "Resnet_weights/weights_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid1/" // !! NEED to remove "_test"
	out_dir := "Resnet_enc_results/results_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid1/"
	fc_out := 10    // 100 for cifar100
	init_pow := 6.0 // covers [-2^pow, 2^pow] values at ReLU evaluation
	mid_pow := 6.0
	final_pow := 6.0
	if cf100 {
		weight_dir = "Resnet_weights/weights_cf100_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid1/"
		out_dir = "Resnet_enc_results/results_cf100_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid1/"
		fc_out = 100 // 100 for cifar100
		if ker_wid == 3 {
			final_pow = 7.0
		} else if ker_wid == 5 {
			final_pow = 6.0
		} else {
			final_pow = 5.0
		}
		init_pow = 5.0
		mid_pow = 5.0
	}

	var num_blcs [3]int
	switch depth {
	case 20:
		num_blcs[0], num_blcs[1], num_blcs[2] = 7, 5, 5
	case 14:
		num_blcs[0], num_blcs[1], num_blcs[2] = 5, 3, 3
	case 8:
		num_blcs[0], num_blcs[1], num_blcs[2] = 3, 1, 1
	default:
		panic("wrong depth (not in 8, 14, 20)!")
	}
	real_batch := []int{16, 32, 64} // same as python (small for faster eval test) !! NEEDS to be changed for real test input {16, 32, 64}
	norm := []int{4, 8, 16}         // only use 1/norm batches among full batches (i.e., sparse packing)
	step := []int{1, 1, 1}          // non-one only when it is for inside

	logN := 16 // !! NEEDS to be modified to 16
	alpha := 0.0
	in_wids := []int{32, 16, 8}                                         // before cropping
	raw_in_wids := []int{32 - ker_wid/2, 16 - ker_wid/2, 8 - ker_wid/2} // same as python
	fast_pack := true
	ker_size := ker_wid * ker_wid
	max_batch := make([]int, len(real_batch)) // the max batch
	for i := range max_batch {
		max_batch[i] = (1 << logN) / (in_wids[i] * in_wids[i])
	}

	cont := newContext(logN, ker_wid, in_wids, raw_in_wids, true, "Resnet_crop_sparse")

	for iter := st; iter < end; iter++ {
		fmt.Println("Running ", iter, "-th iter... ker size: ", ker_wid)
		image := readTxt("Resnet_plain_data/crop_ker"+strconv.Itoa(ker_wid)+"_d"+strconv.Itoa(depth)+"_wid1/test_image_"+strconv.Itoa(iter)+".csv", in_wids[0]*in_wids[0]*3)
		// image := make([]float64, in_wids[0]*in_wids[0]*3)
		// for i := range image {
		// 	image[i] = 1.0 - 1.0*float64(i)/float64(len(image))
		// }
		if cf100 {
			image = readTxt("Resnet_plain_data/cf100_crop_ker"+strconv.Itoa(ker_wid)+"_d"+strconv.Itoa(depth)+"_wid1/test_image_"+strconv.Itoa(iter)+".csv", in_wids[0]*in_wids[0]*3)
		}
		input := make([]float64, cont.N)
		k := 0
		for i := 0; i < in_wids[0]; i++ {
			for j := 0; j < in_wids[0]; j++ {
				for b := 0; b < 3; b++ {
					if (i < raw_in_wids[0]) && (j < raw_in_wids[0]) {
						input[i*in_wids[0]*max_batch[0]+j*max_batch[0]+b*norm[0]] = image[k] // sparse pack the input
					}
					k++
				}
			}
		}
		fmt.Println("Input: ")
		prt_mat_norm(input, max_batch[0], norm[0], 3, false)
		fmt.Println("vec size: ", cont.N)
		fmt.Println("input width: ", raw_in_wids)
		fmt.Println("kernel width: ", ker_wid)
		fmt.Println("num batches: ", real_batch)

		enc_start := time.Now()
		pl_input := ckks.NewPlaintext(cont.params, cont.ECD_LV, cont.params.Scale()) // contain plaintext values
		cont.encoder.EncodeCoeffs(input, pl_input)
		ct_input := cont.encryptor.EncryptNew(pl_input)
		fmt.Printf("Encryption done in %s \n", time.Since(enc_start))

		timings := make([]float64, 6)
		begin_start := time.Now()
		start := time.Now()

		// ResNet Block 1
		pow := init_pow
		ct_layer := ct_input
		for i := 1; i <= num_blcs[0]; i++ {
			bn_a := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-a.csv", real_batch[0])
			bn_b := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-b.csv", real_batch[0])
			// bn_a := make([]float64, real_batch[0])
			// bn_b := make([]float64, real_batch[0])
			// for i := range bn_a {
			// 	bn_a[i] = 0.2
			// 	bn_b[i] = 0.0
			// }
			ker_in_batch := 3
			if i != 1 {
				ker_in_batch = real_batch[0]
			}
			ker_in := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-conv.csv", ker_in_batch*real_batch[0]*ker_size)
			// ker_in := make([]float64, ker_in_batch*real_batch[0]*ker_size)
			// for i := range ker_in {
			// 	ker_in[i] = 0.05 * float64(i) / float64(len(ker_in))
			// }
			ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in, bn_a, bn_b, alpha, pow, in_wids[0], raw_in_wids[0], ker_wid, ker_in_batch, real_batch[0], norm[0], 0, step[0], 2, 2, "Conv_sparse", fast_pack, debug)
			pow = mid_pow
			fmt.Println("Block1, Layer ", i, "done!")
		}
		fmt.Println("Block1 done.") // !!!! HERE is DONE
		timings[0] = time.Since(start).Seconds()
		start = time.Now()

		ker_in12 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0])+"-conv.csv", real_batch[0]*real_batch[1]*ker_size)
		bn_a := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0])+"-a.csv", real_batch[1])
		bn_b := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0])+"-b.csv", real_batch[1])
		// ker_in12 := make([]float64, real_batch[0]*real_batch[1]*ker_size)
		// for i := range ker_in12 {
		// 	ker_in12[i] = 0.05 * float64(i) / float64(len(ker_in12))
		// }
		// bn_a := make([]float64, real_batch[1])
		// bn_b := make([]float64, real_batch[1])
		// for i := range bn_a {
		// 	bn_a[i] = 0.1
		// 	bn_b[i] = 0.0
		// }
		ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in12, bn_a, bn_b, alpha, pow, in_wids[0], raw_in_wids[1], ker_wid, real_batch[0], real_batch[1], norm[1], 0, step[1], 2, 1, "StrConv_sparse", fast_pack, debug)
		fmt.Println("Block1 to 2 done!")
		timings[1] = time.Since(start).Seconds()
		start = time.Now()

		// ResNet Block 2
		for i := 1; i <= num_blcs[1]; i++ {
			bn_a2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+i)+"-a.csv", real_batch[1])
			bn_b2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+i)+"-b.csv", real_batch[1])
			ker_in2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+i)+"-conv.csv", real_batch[1]*real_batch[1]*ker_size)
			// bn_a2 := make([]float64, real_batch[1])
			// bn_b2 := make([]float64, real_batch[1])
			// ker_in2 := make([]float64, real_batch[1]*real_batch[1]*ker_size)
			// for i := range bn_a2 {
			// 	bn_a2[i] = 0.1
			// 	bn_b2[i] = 0.0
			// }
			// for i := range ker_in2 {
			// 	ker_in2[i] = 0.05 * float64(i) / float64(len(ker_in2))
			// }

			ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in2, bn_a2, bn_b2, alpha, pow, in_wids[1], raw_in_wids[1], ker_wid, real_batch[1], real_batch[1], norm[1], 0, step[1], 2, 3, "Conv_sparse", fast_pack, debug)
			fmt.Println("Block2, Layer ", i, "done!")
		}
		fmt.Println("Block2 done.")
		timings[2] = time.Since(start).Seconds()
		start = time.Now()

		ker_in23 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+1)+"-conv.csv", real_batch[1]*real_batch[2]*ker_size)
		bn_a3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+1)+"-a.csv", real_batch[2])
		bn_b3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+1)+"-b.csv", real_batch[2])
		// bn_a3 := make([]float64, real_batch[2])
		// bn_b3 := make([]float64, real_batch[2])
		// ker_in23 := make([]float64, real_batch[1]*real_batch[2]*ker_size)
		// for i := range bn_a3 {
		// 	bn_a3[i] = 0.1
		// 	bn_b3[i] = 0.0
		// }
		// for i := range ker_in23 {
		// 	ker_in23[i] = 0.05 * float64(i) / float64(len(ker_in23))
		// }
		ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in23, bn_a3, bn_b3, alpha, pow, in_wids[1], raw_in_wids[2], ker_wid, real_batch[1], real_batch[2], norm[2], 0, step[2], 2, 2, "StrConv_sparse", fast_pack, debug)
		fmt.Println("Block2 to 3 done!")
		timings[3] = time.Since(start).Seconds()
		start = time.Now()

		// ResNet Block 3
		for i := 1; i <= num_blcs[2]; i++ {
			bn_a3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+i+1)+"-a.csv", real_batch[2])
			bn_b3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+i+1)+"-b.csv", real_batch[2])
			ker_in3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+i+1)+"-conv.csv", real_batch[2]*real_batch[2]*ker_size)
			// bn_a3 := make([]float64, real_batch[2])
			// bn_b3 := make([]float64, real_batch[2])
			// ker_in3 := make([]float64, real_batch[2]*real_batch[2]*ker_size)
			// for i := range bn_a3 {
			// 	bn_a3[i] = 0.1
			// 	bn_b3[i] = 0.0
			// }
			// for i := range ker_in3 {
			// 	ker_in3[i] = 0.1 * float64(i) / float64(len(ker_in3))
			// }

			if i == num_blcs[2] {
				pow = final_pow
			}
			ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in3, bn_a3, bn_b3, alpha, pow, in_wids[2], raw_in_wids[2], ker_wid, real_batch[2], real_batch[2], norm[2], 0, step[2], 2, 4, "Conv_sparse", fast_pack, debug)
			fmt.Println("Block3, Layer ", i, "done!")
		}
		fmt.Println("Block3 done.")
		timings[4] = time.Since(start).Seconds()
		start = time.Now()

		ker_inf_wid := raw_in_wids[2]
		if ker_inf_wid%2 == 0 {
			ker_inf_wid++
		}
		ker_inf := readTxt(weight_dir+"final-fckernel.csv", real_batch[2]*fc_out)
		// ker_inf := make([]float64, real_batch[2]*fc_out)
		// for i := range ker_inf {
		// 	ker_inf[i] = 0.1 * float64(i)
		// }
		var ct_result, ct_result2 *ckks.Ciphertext
		if cf100 {
			ker_inf_1 := make([]float64, ker_inf_wid*ker_inf_wid*real_batch[2]*fc_out/2)
			ker_inf_2 := make([]float64, ker_inf_wid*ker_inf_wid*real_batch[2]*fc_out/2)
			for i := 0; i < fc_out/2; i++ {
				for j := 0; j < real_batch[2]; j++ {
					for b := 0; b < ker_inf_wid*ker_inf_wid; b++ {
						ker_inf_1[j*fc_out/2+i+b*real_batch[2]*fc_out/2] = ker_inf[j*fc_out+i]
						ker_inf_2[j*fc_out/2+i+b*real_batch[2]*fc_out/2] = ker_inf[j*fc_out+i+fc_out/2]
					}
				}
			}
			bn_af := make([]float64, fc_out/2)
			for i := range bn_af {
				bn_af[i] = 1.0 / float64(raw_in_wids[2]*raw_in_wids[2]) // for reduce mean on raw_in_wids[2]**2 elements
			}
			bn_bf := readTxt(weight_dir+"final-fcbias.csv", fc_out)
			bn_bf_1 := make([]float64, fc_out/2)
			bn_bf_2 := make([]float64, fc_out/2)
			for i := range bn_bf_1 {
				bn_bf_1[i] = bn_bf[i]
				bn_bf_2[i] = bn_bf[i+fc_out/2]
			}
			ct_result = evalConv_BN(cont, ct_layer, ker_inf_1, bn_af, bn_bf_1, in_wids[2], ker_inf_wid, real_batch[2], fc_out/2, norm[2], float64(1<<30), false)
			ct_result2 = evalConv_BN(cont, ct_layer, ker_inf_2, bn_af, bn_bf_2, in_wids[2], ker_inf_wid, real_batch[2], fc_out/2, norm[2], float64(1<<30), false)
			fmt.Println("Final FC done.")
			timings[5] = time.Since(start).Seconds()
			start = time.Now()
		} else {
			ker_inf_ := make([]float64, ker_inf_wid*ker_inf_wid*real_batch[2]*fc_out)
			for i := range ker_inf {
				for b := 0; b < ker_inf_wid*ker_inf_wid; b++ {
					ker_inf_[i+b*real_batch[2]*fc_out] = ker_inf[i]
				}
			}
			bn_af := make([]float64, fc_out)
			for i := range bn_af {
				bn_af[i] = 1.0 / float64(raw_in_wids[2]*raw_in_wids[2]) // for reduce mean on raw_in_wids[2]**2 elements
			}
			bn_bf := readTxt(weight_dir+"final-fcbias.csv", fc_out)
			// bn_bf := make([]float64, fc_out)
			// for i := range bn_bf {
			// 	bn_bf[i] = 1 * float64(i)
			// }
			ct_result = evalConv_BN(cont, ct_layer, ker_inf_, bn_af, bn_bf, in_wids[2], ker_inf_wid, real_batch[2], fc_out, norm[2], float64(1<<30), false)
			fmt.Println("Final FC done.")
			timings[5] = time.Since(start).Seconds()
			start = time.Now()
		}

		fmt.Println()
		fmt.Println("===============  DECRYPTION  ===============")
		fmt.Println()
		if cf100 {
			cont.decryptor.Decrypt(ct_result, pl_input)
			res_tmp1 := cont.encoder.DecodeCoeffs(pl_input)
			cont.decryptor.Decrypt(ct_result2, pl_input)
			res_tmp2 := cont.encoder.DecodeCoeffs(pl_input)
			fmt.Printf("Decryption Done in %s \n", time.Since(start))
			res_out := append(prt_mat_one_norm(res_tmp1, max_batch[2], norm[2], ker_inf_wid/2+1, ker_inf_wid/2+1)[:fc_out/2], prt_mat_one_norm(res_tmp2, max_batch[2], norm[2], ker_inf_wid/2+1, ker_inf_wid/2+1)[:fc_out/2]...)
			fmt.Println("\n result: ", res_out)
			writeTxt(out_dir+"class_result_"+ker_name+"_"+strconv.Itoa(iter)+".csv", res_out)
		} else {
			cont.decryptor.Decrypt(ct_result, pl_input)
			res_tmp := cont.encoder.DecodeCoeffs(pl_input)
			fmt.Printf("Decryption Done in %s \n", time.Since(start))
			res_out := prt_mat_one_norm(res_tmp, max_batch[2], norm[2], ker_inf_wid/2+1, ker_inf_wid/2+1)
			fmt.Println("\n result: ", res_out[:fc_out])
			writeTxt(out_dir+"class_result_"+ker_name+"_"+strconv.Itoa(iter)+".csv", res_out[:fc_out])
		}

		fmt.Println("Blc1: ", timings[0], " sec")
		fmt.Println("Blc1->2: ", timings[1], " sec")
		fmt.Println("Blc2: ", timings[2], " sec")
		fmt.Println("Blc2->3: ", timings[3], " sec")
		fmt.Println("Blc3: ", timings[4], " sec")
		fmt.Println("Final (reduce_mean & FC): ", timings[5], " sec")
		fmt.Printf("Total done in %s \n", time.Since(begin_start))
	}

}

func testResNet_crop_fast_in(st, end, ker_wid, depth int, debug, cf100 bool) {
	// init_batch fixed to 16
	ker_name := "ker" + strconv.Itoa(ker_wid)
	weight_dir := "Resnet_weights/weights_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid1/"
	out_dir := "Resnet_enc_results/results_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid1/"
	fc_out := 10    // 100 for cifar100
	init_pow := 6.0 // covers [-2^pow, 2^pow] values at ReLU evaluation
	mid_pow := 6.0
	final_pow := 6.0
	if cf100 {
		weight_dir = "Resnet_weights/weights_cf100_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid1/"
		out_dir = "Resnet_enc_results/results_cf100_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid1/"
		fc_out = 100 // 100 for cifar100
		if ker_wid == 3 {
			final_pow = 7.0
		} else if ker_wid == 5 {
			final_pow = 6.0
		} else {
			final_pow = 5.0
		}
		init_pow = 5.0
		mid_pow = 5.0
	}

	var num_blcs [3]int
	switch depth {
	case 20:
		num_blcs[0], num_blcs[1], num_blcs[2] = 7, 5, 5
	case 14:
		num_blcs[0], num_blcs[1], num_blcs[2] = 5, 3, 3
	case 8:
		num_blcs[0], num_blcs[1], num_blcs[2] = 3, 1, 1
	default:
		panic("wrong depth (not in 8, 14, 20)!")
	}
	real_batch := []int{16, 32, 64} // same as python
	norm := []int{4, 2, 1}          // only use 1/norm batches
	step := []int{1, 2, 4}
	prt_start := []int{1, 1, 1}
	if ker_wid == 5 {
		prt_start[0] = 1
		prt_start[1] = 2
		prt_start[2] = 4
	}

	logN := 16
	alpha := 0.0
	in_wids := []int{32, 16, 8}                                         // before cropping
	raw_in_wids := []int{32 - ker_wid/2, 16 - ker_wid/2, 8 - ker_wid/2} // same as python
	fast_pack := true
	ker_size := ker_wid * ker_wid
	max_batch := make([]int, len(real_batch)) // the max batch
	for i := range max_batch {
		max_batch[i] = (1 << logN) / (in_wids[i] * in_wids[i])
	}

	cont := newContext(logN, ker_wid, in_wids, raw_in_wids, true, "Resnet_crop_fast")

	for iter := st; iter < end; iter++ {
		fmt.Println("Running ", iter, "-th iter... ker size: ", ker_wid)
		image := readTxt("Resnet_plain_data/crop_ker"+strconv.Itoa(ker_wid)+"_d"+strconv.Itoa(depth)+"_wid1/test_image_"+strconv.Itoa(iter)+".csv", in_wids[0]*in_wids[0]*3)
		if cf100 {
			image = readTxt("Resnet_plain_data/cf100_crop_ker"+strconv.Itoa(ker_wid)+"_d"+strconv.Itoa(depth)+"_wid1/test_image_"+strconv.Itoa(iter)+".csv", in_wids[0]*in_wids[0]*3)
		}
		input := make([]float64, cont.N)
		k := 0
		for i := 0; i < in_wids[0]; i++ {
			for j := 0; j < in_wids[0]; j++ {
				for b := 0; b < 3; b++ {
					if (i < raw_in_wids[0]) && (j < raw_in_wids[0]) {
						input[i*in_wids[0]*max_batch[0]+j*max_batch[0]+b*norm[0]] = image[k]
					}
					k++
				}
			}
		}
		fmt.Println("Input: ")
		prt_mat_norm(input, max_batch[0], norm[0], 1, false)
		fmt.Println("vec size: ", cont.N)
		fmt.Println("input width: ", raw_in_wids)
		fmt.Println("kernel width: ", ker_wid)
		fmt.Println("num batches: ", real_batch)

		enc_start := time.Now()
		pl_input := ckks.NewPlaintext(cont.params, cont.ECD_LV, cont.params.Scale()) // contain plaintext values
		cont.encoder.EncodeCoeffs(input, pl_input)
		ct_input := cont.encryptor.EncryptNew(pl_input)
		fmt.Printf("Encryption done in %s \n", time.Since(enc_start))

		timings := make([]float64, 6)
		begin_start := time.Now()
		start := time.Now()

		// ResNet Block 1
		pow := init_pow
		ct_layer := ct_input
		for i := 1; i <= num_blcs[0]; i++ {
			bn_a := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-a.csv", real_batch[0])
			bn_b := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-b.csv", real_batch[0])
			ker_in_batch := 3
			if i != 1 {
				ker_in_batch = real_batch[0]
			}
			ker_in := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-conv.csv", ker_in_batch*real_batch[0]*ker_size)
			ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in, bn_a, bn_b, alpha, pow, in_wids[0], raw_in_wids[0], ker_wid, ker_in_batch, real_batch[0], norm[0], 0, step[0], 2, 0, "Conv_inside", fast_pack, debug)
			pow = mid_pow
			fmt.Println("Block1, Layer ", i, "done!")
		}
		fmt.Println("Block1 done.")
		timings[0] = time.Since(start).Seconds()
		start = time.Now()

		ker_in12 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0])+"-conv.csv", real_batch[0]*real_batch[1]*ker_size)
		ker_in12_new := make([]float64, 2*real_batch[0]*real_batch[1]*ker_size)
		for k := 0; k < ker_size; k++ {
			for i := 0; i < real_batch[0]; i++ {
				for j := 0; j < real_batch[1]; j++ {
					ker_in12_new[k*2*real_batch[0]*real_batch[1]+2*i*real_batch[1]+j] = ker_in12[k*real_batch[0]*real_batch[1]+i*real_batch[1]+j]
				}
			}
		}
		bn_a := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0])+"-a.csv", real_batch[1])
		bn_b := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0])+"-b.csv", real_batch[1])
		ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in12_new, bn_a, bn_b, alpha, pow, in_wids[0], raw_in_wids[1], ker_wid, real_batch[1], real_batch[1], norm[1], 0, step[1], 2, 0, "StrConv_inside", fast_pack, debug)
		fmt.Println("Block1 to 2 done!")
		if debug {
			max_bat := cont.N / (in_wids[0] * in_wids[0])
			res_ttmp := cont.encoder.DecodeCoeffs(cont.decryptor.DecryptNew(ct_layer))
			prt_mat_norm_step(res_ttmp, max_bat, norm[1], step[1], prt_start[1], 3, false)
		}
		timings[1] = time.Since(start).Seconds()
		start = time.Now()

		// ResNet Block 2
		for i := 1; i <= num_blcs[1]; i++ {
			bn_a2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+i)+"-a.csv", real_batch[1])
			bn_b2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+i)+"-b.csv", real_batch[1])
			ker_in2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+i)+"-conv.csv", real_batch[1]*real_batch[1]*ker_size)

			ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in2, bn_a2, bn_b2, alpha, pow, in_wids[0], raw_in_wids[1], ker_wid, real_batch[1], real_batch[1], norm[1], 0, step[1], 2, 0, "Conv_inside", fast_pack, debug)
			fmt.Println("Block2, Layer ", i, "done!")
		}
		fmt.Println("Block2 done.")
		timings[2] = time.Since(start).Seconds()
		start = time.Now()

		ker_in23 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+1)+"-conv.csv", real_batch[1]*real_batch[2]*ker_size)
		bn_a3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+1)+"-a.csv", real_batch[2])
		bn_b3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+1)+"-b.csv", real_batch[2])
		ker_in23_new := make([]float64, 2*real_batch[1]*real_batch[2]*ker_size)
		for k := 0; k < ker_size; k++ {
			for i := 0; i < real_batch[1]; i++ {
				for j := 0; j < real_batch[2]; j++ {
					ker_in23_new[k*2*real_batch[1]*real_batch[2]+2*i*real_batch[2]+j] = ker_in23[k*real_batch[1]*real_batch[2]+i*real_batch[2]+j]
				}
			}
		}
		ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in23_new, bn_a3, bn_b3, alpha, pow, in_wids[0], raw_in_wids[2], ker_wid, real_batch[2], real_batch[2], norm[2], 0, step[2], 2, 0, "StrConv_inside", fast_pack, debug)
		fmt.Println("Block2 to 3 done!")
		if debug {
			max_bat := cont.N / (in_wids[0] * in_wids[0])
			res_ttmp := cont.encoder.DecodeCoeffs(cont.decryptor.DecryptNew(ct_layer))
			prt_mat_norm_step(res_ttmp, max_bat, norm[2], step[2], prt_start[2], 3, false)
		}
		timings[3] = time.Since(start).Seconds()
		start = time.Now()

		// ResNet Block 3
		for i := 1; i <= num_blcs[2]; i++ {
			bn_a3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+i+1)+"-a.csv", real_batch[2])
			bn_b3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+i+1)+"-b.csv", real_batch[2])
			ker_in3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+i+1)+"-conv.csv", real_batch[2]*real_batch[2]*ker_size)

			if i == num_blcs[2] {
				pow = final_pow
			}
			ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in3, bn_a3, bn_b3, alpha, pow, in_wids[0], raw_in_wids[2], ker_wid, real_batch[2], real_batch[2], norm[2], 0, step[2], 2, 0, "Conv_inside", fast_pack, debug)
			fmt.Println("Block3, Layer ", i, "done!")
		}
		fmt.Println("Block3 done.")
		timings[4] = time.Since(start).Seconds()
		start = time.Now()

		ker_inf_wid := raw_in_wids[0]
		if ker_inf_wid%2 == 0 {
			ker_inf_wid++
		}
		ker_inf := readTxt(weight_dir+"final-fckernel.csv", real_batch[2]*fc_out)
		var ct_result, ct_result2 *ckks.Ciphertext
		if cf100 {
			ker_inf_1 := make([]float64, ker_inf_wid*ker_inf_wid*real_batch[2]*fc_out/2)
			ker_inf_2 := make([]float64, ker_inf_wid*ker_inf_wid*real_batch[2]*fc_out/2)
			for i := 0; i < fc_out/2; i++ {
				for j := 0; j < real_batch[2]; j++ {
					for b := 0; b < ker_inf_wid*ker_inf_wid; b++ {
						ker_inf_1[j*fc_out/2+i+b*real_batch[2]*fc_out/2] = ker_inf[j*fc_out+i]
						ker_inf_2[j*fc_out/2+i+b*real_batch[2]*fc_out/2] = ker_inf[j*fc_out+i+fc_out/2]
					}
				}
			}
			bn_af := make([]float64, fc_out/2)
			for i := range bn_af {
				bn_af[i] = 1.0 / float64(raw_in_wids[2]*raw_in_wids[2]) // for reduce mean on raw_in_wids[2]**2 elements
			}
			bn_bf := readTxt(weight_dir+"final-fcbias.csv", fc_out)
			bn_bf_1 := make([]float64, fc_out/2)
			bn_bf_2 := make([]float64, fc_out/2)
			for i := range bn_bf_1 {
				bn_bf_1[i] = bn_bf[i]
				bn_bf_2[i] = bn_bf[i+fc_out/2]
			}
			ct_result = evalConv_BN(cont, ct_layer, ker_inf_1, bn_af, bn_bf_1, in_wids[0], ker_inf_wid, real_batch[2], fc_out/2, norm[2], float64(1<<30), false)
			ct_result2 = evalConv_BN(cont, ct_layer, ker_inf_2, bn_af, bn_bf_2, in_wids[0], ker_inf_wid, real_batch[2], fc_out/2, norm[2], float64(1<<30), false)
			fmt.Println("Final FC done.")
			timings[5] = time.Since(start).Seconds()
			start = time.Now()
		} else {
			ker_inf_ := make([]float64, ker_inf_wid*ker_inf_wid*real_batch[2]*fc_out)
			for i := range ker_inf {
				for b := 0; b < ker_inf_wid*ker_inf_wid; b++ {
					ker_inf_[i+b*real_batch[2]*fc_out] = ker_inf[i]
				}
			}
			bn_af := make([]float64, fc_out)
			for i := range bn_af {
				bn_af[i] = 1.0 / float64(raw_in_wids[2]*raw_in_wids[2]) // for reduce mean on raw_in_wids[2]**2 elements
			}
			bn_bf := readTxt(weight_dir+"final-fcbias.csv", fc_out)
			ct_result = evalConv_BN(cont, ct_layer, ker_inf_, bn_af, bn_bf, in_wids[0], ker_inf_wid, real_batch[2], fc_out, norm[2], float64(1<<30), false)
			fmt.Println("Final FC done.")
			timings[5] = time.Since(start).Seconds()
			start = time.Now()
		}

		fmt.Println()
		fmt.Println("===============  DECRYPTION  ===============")
		fmt.Println()
		if cf100 {
			cont.decryptor.Decrypt(ct_result, pl_input)
			res_tmp1 := cont.encoder.DecodeCoeffs(pl_input)
			cont.decryptor.Decrypt(ct_result2, pl_input)
			res_tmp2 := cont.encoder.DecodeCoeffs(pl_input)
			fmt.Printf("Decryption Done in %s \n", time.Since(start))
			res_out := append(prt_mat_one_norm(res_tmp1, max_batch[0], norm[2], ker_inf_wid/2+1, ker_inf_wid/2+1)[:fc_out/2], prt_mat_one_norm(res_tmp2, max_batch[0], norm[2], ker_inf_wid/2+1, ker_inf_wid/2+1)[:fc_out/2]...)
			fmt.Println("\n result: ", res_out)
			writeTxt(out_dir+"class_result_"+ker_name+"_"+strconv.Itoa(iter)+".csv", res_out)
		} else {
			cont.decryptor.Decrypt(ct_result, pl_input)
			res_tmp := cont.encoder.DecodeCoeffs(pl_input)
			fmt.Printf("Decryption Done in %s \n", time.Since(start))
			res_out := prt_mat_one_norm(res_tmp, max_batch[0], norm[2], ker_inf_wid/2+1, ker_inf_wid/2+1)
			// fmt.Print(res_out)
			fmt.Println("\n result: ", res_out[:fc_out])
			writeTxt(out_dir+"class_result_"+ker_name+"_"+strconv.Itoa(iter)+".csv", res_out[:fc_out])
		}

		fmt.Println("Blc1: ", timings[0], " sec")
		fmt.Println("Blc1->2: ", timings[1], " sec")
		fmt.Println("Blc2: ", timings[2], " sec")
		fmt.Println("Blc2->3: ", timings[3], " sec")
		fmt.Println("Blc3: ", timings[4], " sec")
		fmt.Println("Final (reduce_mean & FC): ", timings[5], " sec")
		fmt.Printf("Total done in %s \n", time.Since(begin_start))
	}
}

func testResNet_crop_sparse_wide(st, end, ker_wid, depth, wide_case int, debug, cf100 bool) {
	// init_batch fixed to 16
	ker_name := "ker" + strconv.Itoa(ker_wid)
	weight_dir := "Resnet_weights/weights_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid" + strconv.Itoa(wide_case) + "/"
	out_dir := "Resnet_enc_results/results_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid" + strconv.Itoa(wide_case) + "/"
	fc_out := 10

	init_pow := 5.0
	mid_pow := 5.0 // needs to be 5.0 in k3 d20 w3 for best performance
	final_pow := 5.0
	if ker_wid == 5 {
		init_pow = 6.0
		mid_pow = 6.0
		final_pow = 6.0
	}

	if cf100 {
		weight_dir = "Resnet_weights/weights_cf100_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid" + strconv.Itoa(wide_case) + "/"
		out_dir = "Resnet_enc_results/results_cf100_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid" + strconv.Itoa(wide_case) + "/"
		fc_out = 100
		final_pow = 7.0
		init_pow = 5.0
		mid_pow = 5.0
		if (ker_wid == 5) && (depth == 8) {
			init_pow = 6.0
			final_pow = 6.0
		}
	}

	init_batch := 16

	var num_blcs [3]int
	switch depth {
	case 20:
		num_blcs[0], num_blcs[1], num_blcs[2] = 7, 5, 5
	case 14:
		num_blcs[0], num_blcs[1], num_blcs[2] = 5, 3, 3
	case 8:
		num_blcs[0], num_blcs[1], num_blcs[2] = 3, 1, 1
	default:
		panic("wrong depth case (not in 8,14,20)!")
	}
	real_batch := []int{32, 64, 128} // same as python
	norm := []int{2, 4, 8}           // only use 1/norm batches
	log_sparse := []int{1, 2, 3}
	step := []int{1, 1, 1}
	kind := "Resnet_crop_sparse_wide2"

	if wide_case == 3 {
		real_batch = []int{48, 96, 192}
		norm = []int{1, 2, 4}
		log_sparse = []int{0, 1, 2}
		kind = "Resnet_crop_sparse_wide3"
	} else if wide_case != 2 {
		panic("wrong wide_case (2 nor 3)!")
	}

	logN := 16
	alpha := 0.0
	in_wids := []int{32, 16, 8}                                         // before cropping
	raw_in_wids := []int{32 - ker_wid/2, 16 - ker_wid/2, 8 - ker_wid/2} // same as python
	fast_pack := true
	ker_size := ker_wid * ker_wid
	max_batch := make([]int, len(real_batch)) // the max batch
	for i := range max_batch {
		max_batch[i] = (1 << logN) / (in_wids[i] * in_wids[i])
	}

	cont := newContext(logN, ker_wid, in_wids, raw_in_wids, true, kind)

	for iter := st; iter < end; iter++ {
		fmt.Println("Running ", iter, "-th iter... ker size: ", ker_wid)
		image := readTxt("Resnet_plain_data/crop_ker"+strconv.Itoa(ker_wid)+"_d"+strconv.Itoa(depth)+"_wid"+strconv.Itoa(wide_case)+"/test_image_"+strconv.Itoa(iter)+".csv", in_wids[0]*in_wids[0]*3)

		if cf100 {
			image = readTxt("Resnet_plain_data/cf100_crop_ker"+strconv.Itoa(ker_wid)+"_d"+strconv.Itoa(depth)+"_wid"+strconv.Itoa(wide_case)+"/test_image_"+strconv.Itoa(iter)+".csv", in_wids[0]*in_wids[0]*3)
		}
		input := make([]float64, cont.N)
		k := 0
		for i := 0; i < in_wids[0]; i++ {
			for j := 0; j < in_wids[0]; j++ {
				for b := 0; b < 3; b++ {
					if (i < raw_in_wids[0]) && (j < raw_in_wids[0]) {
						input[i*in_wids[0]*max_batch[0]+j*max_batch[0]+b*norm[0]] = image[k]
					}
					k++
				}
			}
		}
		fmt.Println("Input: ")
		prt_mat_norm(input, max_batch[0], norm[0], 3, false)
		fmt.Println("vec size: ", cont.N)
		fmt.Println("input width: ", raw_in_wids)
		fmt.Println("kernel width: ", ker_wid)
		fmt.Println("num batches: ", real_batch)

		enc_start := time.Now()
		pl_input := ckks.NewPlaintext(cont.params, cont.ECD_LV, cont.params.Scale()) // contain plaintext values
		cont.encoder.EncodeCoeffs(input, pl_input)
		ct_input := cont.encryptor.EncryptNew(pl_input)
		fmt.Printf("Encryption done in %s \n", time.Since(enc_start))
		enc_start = time.Now()

		timings := make([]float64, 6)
		begin_start := time.Now()
		start := time.Now()

		// ResNet Block 1
		pow := init_pow
		ct_layer := ct_input
		for i := 1; i <= num_blcs[0]; i++ {
			if i == 5 {
				pow = mid_pow
			}
			var bn_batch int
			if i == 1 {
				bn_batch = init_batch
			} else {
				bn_batch = real_batch[0]
			}
			bn_a := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-a.csv", bn_batch)
			bn_b := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-b.csv", bn_batch)

			if i == 1 {
				ker_in := readTxt(weight_dir+"w0-conv.csv", 3*init_batch*ker_size)
				ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in, bn_a, bn_b, alpha, pow, in_wids[0], raw_in_wids[0], ker_wid, 3, init_batch, norm[0], 0, step[0], 2, log_sparse[0], "Conv_sparse", fast_pack, debug)
				// pow = mid_pow
			} else if i == 2 {
				ker_in := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-conv.csv", init_batch*real_batch[0]*ker_size)
				ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in, bn_a, bn_b, alpha, pow, in_wids[0], raw_in_wids[0], ker_wid, init_batch, real_batch[0], norm[0], 0, step[0], 2, log_sparse[0], "Conv_sparse", fast_pack, debug)
			} else {
				ker_in := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-conv.csv", real_batch[0]*real_batch[0]*ker_size)
				ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in, bn_a, bn_b, alpha, pow, in_wids[0], raw_in_wids[0], ker_wid, real_batch[0], real_batch[0], norm[0], 0, step[0], 2, log_sparse[0], "Conv_sparse", fast_pack, debug)
			}
			fmt.Println("Block1, Layer ", i, "done!")
		}
		fmt.Println("Block1 done.")
		timings[0] = time.Since(start).Seconds()
		start = time.Now()

		ker_in12 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0])+"-conv.csv", real_batch[0]*real_batch[1]*ker_size)
		ker_in12_0 := make([]float64, len(ker_in12)/2)
		ker_in12_1 := make([]float64, len(ker_in12)/2)
		if wide_case == 3 {
			for k := 0; k < ker_size; k++ {
				for i := 0; i < real_batch[0]; i++ {
					for j := 0; j < real_batch[1]/2; j++ {
						ker_in12_0[k*real_batch[0]*real_batch[1]/2+(i*real_batch[1]/2+j)] = ker_in12[k*real_batch[0]*real_batch[1]+(i*real_batch[1]+2*j)]   // [i][2*j]
						ker_in12_1[k*real_batch[0]*real_batch[1]/2+(i*real_batch[1]/2+j)] = ker_in12[k*real_batch[0]*real_batch[1]+(i*real_batch[1]+2*j+1)] // [i][2*j+1]
					}
				}
			}
		}

		bn_a := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0])+"-a.csv", real_batch[1])
		bn_b := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0])+"-b.csv", real_batch[1])

		if wide_case == 2 {
			ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in12, bn_a, bn_b, alpha, pow, in_wids[0], raw_in_wids[1], ker_wid, real_batch[0], real_batch[1], norm[1], 0, step[1], 2, log_sparse[0]-1, "StrConv_sparse", fast_pack, debug)
		} else if wide_case == 3 {
			bn_a_0 := make([]float64, real_batch[1]/2)
			bn_a_1 := make([]float64, real_batch[1]/2)
			bn_b_0 := make([]float64, real_batch[1]/2)
			bn_b_1 := make([]float64, real_batch[1]/2)
			for i := range bn_b_0 {
				bn_a_0[i] = bn_a[2*i]
				bn_a_1[i] = bn_a[2*i+1]
				bn_b_0[i] = bn_b[2*i]
				bn_b_1[i] = bn_b[2*i+1]
			}
			ct_result1 := evalConv_BNRelu_new(cont, ct_layer, ker_in12_0, bn_a_0, bn_b_0, alpha, pow, in_wids[0], raw_in_wids[1], ker_wid, real_batch[0], real_batch[1]/2, norm[0], 0, step[1], 2, 0, "StrConv_sparse_full", fast_pack, debug)
			ct_result2 := evalConv_BNRelu_new(cont, ct_layer, ker_in12_1, bn_a_1, bn_b_1, alpha, pow, in_wids[0], raw_in_wids[1], ker_wid, real_batch[0], real_batch[1]/2, norm[0], 0, step[1], 2, 0, "StrConv_sparse_full", fast_pack, debug)

			xi := make([]float64, cont.N)
			xi[2] = 1.0
			xi_plain := ckks.NewPlaintext(cont.params, ct_result2.Level(), 1.0)
			cont.encoder.EncodeCoeffs(xi, xi_plain)
			cont.encoder.ToNTT(xi_plain)
			ct_result2 = cont.evaluator.MulNew(ct_result2, xi_plain)
			ct_layer = cont.evaluator.AddNew(ct_result1, ct_result2)
		}
		fmt.Println("Block1 to 2 done!")
		timings[1] = time.Since(start).Seconds()
		start = time.Now()

		// ResNet Block 2
		for i := 1; i <= num_blcs[1]; i++ {
			if i == 5 {
				pow = init_pow
			}
			bn_a2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+i)+"-a.csv", real_batch[1])
			bn_b2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+i)+"-b.csv", real_batch[1])
			ker_in2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+i)+"-conv.csv", real_batch[1]*real_batch[1]*ker_size)

			ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in2, bn_a2, bn_b2, alpha, pow, in_wids[1], raw_in_wids[1], ker_wid, real_batch[1], real_batch[1], norm[1], 0, step[1], 2, log_sparse[1], "Conv_sparse", fast_pack, debug)
			fmt.Println("Block2, Layer ", i, "done!")
		}
		fmt.Println("Block2 done.")
		timings[2] = time.Since(start).Seconds()
		start = time.Now()

		pow = mid_pow
		ker_in23 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+1)+"-conv.csv", real_batch[1]*real_batch[2]*ker_size)
		bn_a3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+1)+"-a.csv", real_batch[2])
		bn_b3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+1)+"-b.csv", real_batch[2])

		ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in23, bn_a3, bn_b3, alpha, pow, in_wids[1], raw_in_wids[2], ker_wid, real_batch[1], real_batch[2], norm[2], 0, step[2], 2, log_sparse[1]-1, "StrConv_sparse", fast_pack, debug)
		fmt.Println("Block2 to 3 done!")
		timings[3] = time.Since(start).Seconds()
		start = time.Now()

		// ResNet Block 3
		for i := 1; i <= num_blcs[2]; i++ {
			if i == 3 {
				pow = init_pow
			}
			if i == 5 {
				pow = mid_pow
			}
			bn_a3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+i+1)+"-a.csv", real_batch[2])
			bn_b3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+i+1)+"-b.csv", real_batch[2])
			ker_in3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+i+1)+"-conv.csv", real_batch[2]*real_batch[2]*ker_size)

			if i == num_blcs[2] {
				pow = final_pow
			}
			ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in3, bn_a3, bn_b3, alpha, pow, in_wids[2], raw_in_wids[2], ker_wid, real_batch[2], real_batch[2], norm[2], 0, step[2], 2, log_sparse[2], "Conv_sparse", fast_pack, debug)
			fmt.Println("Block3, Layer ", i, "done!")
		}
		fmt.Println("Block3 done.")
		timings[4] = time.Since(start).Seconds()
		start = time.Now()

		ker_inf_wid := raw_in_wids[2]
		if ker_inf_wid%2 == 0 {
			ker_inf_wid++
		}
		ker_inf := readTxt(weight_dir+"final-fckernel.csv", real_batch[2]*fc_out)

		ker_inf_ := make([]float64, ker_inf_wid*ker_inf_wid*real_batch[2]*fc_out)
		for i := range ker_inf {
			for b := 0; b < ker_inf_wid*ker_inf_wid; b++ {
				ker_inf_[i+b*real_batch[2]*fc_out] = ker_inf[i]
			}
		}
		bn_af := make([]float64, fc_out)
		for i := range bn_af {
			bn_af[i] = 1.0 / float64(raw_in_wids[2]*raw_in_wids[2]) // for reduce mean on raw_in_wids[2]**2 elements
		}
		bn_bf := readTxt(weight_dir+"final-fcbias.csv", fc_out)

		ct_result := evalConv_BN(cont, ct_layer, ker_inf_, bn_af, bn_bf, in_wids[2], ker_inf_wid, real_batch[2], fc_out, norm[2], float64(1<<30), false)
		fmt.Println("Final FC done.")
		timings[5] = time.Since(start).Seconds()
		start = time.Now()

		fmt.Println()
		fmt.Println("===============  DECRYPTION  ===============")
		fmt.Println()
		cont.decryptor.Decrypt(ct_result, pl_input)
		res_tmp := cont.encoder.DecodeCoeffs(pl_input)
		fmt.Printf("Decryption Done in %s \n", time.Since(start))
		res_out := prt_mat_one_norm(res_tmp, max_batch[2], norm[2], ker_inf_wid/2+1, ker_inf_wid/2+1)
		fmt.Println("\n result: ", res_out[:fc_out])
		writeTxt(out_dir+"class_result_"+ker_name+"_"+strconv.Itoa(iter)+".csv", res_out[:fc_out])

		fmt.Println("Blc1: ", timings[0], " sec")
		fmt.Println("Blc1->2: ", timings[1], " sec")
		fmt.Println("Blc2: ", timings[2], " sec")
		fmt.Println("Blc2->3: ", timings[3], " sec")
		fmt.Println("Blc3: ", timings[4], " sec")
		fmt.Println("Final (reduce_mean & FC): ", timings[5], " sec")
		fmt.Printf("Total done in %s \n", time.Since(begin_start))
	}
}

func testResNet_crop_fast_wide_in(st, end, ker_wid, depth, wide_case int, debug, cf100 bool) {
	// init_batch fixed to 16
	ker_name := "ker" + strconv.Itoa(ker_wid)
	weight_dir := "Resnet_weights/weights_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid" + strconv.Itoa(wide_case) + "/"
	out_dir := "Resnet_enc_results/results_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid" + strconv.Itoa(wide_case) + "/"
	fc_out := 10 // 100 for cifar100

	init_pow := 5.0
	mid_pow := 5.0 // needs to be 5.0 in k3 d20 w3 for best performance
	final_pow := 5.0
	if ker_wid == 5 {
		init_pow = 6.0
		mid_pow = 6.0
		final_pow = 6.0
	}

	if cf100 {
		weight_dir = "Resnet_weights/weights_cf100_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid" + strconv.Itoa(wide_case) + "/"
		out_dir = "Resnet_enc_results/results_cf100_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid" + strconv.Itoa(wide_case) + "/"
		fc_out = 100 // 100 for cifar100
		final_pow = 7.0
		init_pow = 5.0
		mid_pow = 5.0
		if (ker_wid == 5) && (depth == 8) {
			init_pow = 6.0
			final_pow = 6.0
		}
	}

	init_batch := 16 // needs to be modified to 16

	var num_blcs [3]int
	switch depth {
	case 20:
		num_blcs[0], num_blcs[1], num_blcs[2] = 7, 5, 5
	case 14:
		num_blcs[0], num_blcs[1], num_blcs[2] = 5, 3, 3
	case 8:
		num_blcs[0], num_blcs[1], num_blcs[2] = 3, 1, 1
	default:
		panic("wrong depth case (not in 8,14,20)!")
	}
	real_batch := []int{32, 64, 128} // same as python
	norm := []int{2, 4, 2}           // only use 1/norm batches
	step := []int{1, 1, 2}
	prt_start := []int{1, 1, 1}
	kind := "Resnet_crop_fast_wide2"
	if ker_wid == 5 {
		prt_start[0] = 1
		prt_start[1] = 1
		prt_start[2] = 2
	}
	if wide_case == 3 {
		real_batch = []int{48, 96, 192}
		norm = []int{1, 2, 1}
		kind = "Resnet_crop_fast_wide3"
	} else if wide_case != 2 {
		panic("wrong wide_case (2 nor 3)!")
	}

	logN := 16
	alpha := 0.0
	in_wids := []int{32, 16, 8}                                         // before cropping
	raw_in_wids := []int{32 - ker_wid/2, 16 - ker_wid/2, 8 - ker_wid/2} // same as python
	fast_pack := true
	ker_size := ker_wid * ker_wid
	max_batch := make([]int, len(real_batch)) // the max batch
	for i := range max_batch {
		max_batch[i] = (1 << logN) / (in_wids[i] * in_wids[i])
	}

	cont := newContext(logN, ker_wid, in_wids, raw_in_wids, true, kind)

	for iter := st; iter < end; iter++ {
		fmt.Println("Running ", iter, "-th iter... ker size: ", ker_wid)
		image := readTxt("Resnet_plain_data/crop_ker"+strconv.Itoa(ker_wid)+"_d"+strconv.Itoa(depth)+"_wid"+strconv.Itoa(wide_case)+"/test_image_"+strconv.Itoa(iter)+".csv", in_wids[0]*in_wids[0]*3)
		if cf100 {
			image = readTxt("Resnet_plain_data/cf100_crop_ker"+strconv.Itoa(ker_wid)+"_d"+strconv.Itoa(depth)+"_wid"+strconv.Itoa(wide_case)+"/test_image_"+strconv.Itoa(iter)+".csv", in_wids[0]*in_wids[0]*3)
		}
		input := make([]float64, cont.N)
		k := 0
		for i := 0; i < in_wids[0]; i++ {
			for j := 0; j < in_wids[0]; j++ {
				for b := 0; b < 3; b++ {
					if (i < raw_in_wids[0]) && (j < raw_in_wids[0]) {
						input[i*in_wids[0]*max_batch[0]+j*max_batch[0]+b*norm[0]] = image[k]
					}
					k++
				}
			}
		}
		fmt.Println("Input: ")
		prt_mat_norm(input, max_batch[0], norm[0], 1, false)
		fmt.Println("vec size: ", cont.N)
		fmt.Println("input width: ", raw_in_wids)
		fmt.Println("kernel width: ", ker_wid)
		fmt.Println("num batches: ", real_batch)

		enc_start := time.Now()
		pl_input := ckks.NewPlaintext(cont.params, cont.ECD_LV, cont.params.Scale()) // contain plaintext values
		cont.encoder.EncodeCoeffs(input, pl_input)
		ct_input := cont.encryptor.EncryptNew(pl_input)
		fmt.Printf("Encryption done in %s \n", time.Since(enc_start))
		enc_start = time.Now()

		timings := make([]float64, 6)
		begin_start := time.Now()
		start := time.Now()

		// ResNet Block 1
		pow := init_pow
		ct_layer := ct_input
		for i := 1; i <= num_blcs[0]; i++ {
			if i == 5 {
				pow = mid_pow
			}
			var bn_batch int
			if i == 1 {
				bn_batch = init_batch
			} else {
				bn_batch = real_batch[0]
			}
			bn_a := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-a.csv", bn_batch)
			bn_b := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-b.csv", bn_batch)
			if i == 1 {
				ker_in := readTxt(weight_dir+"w0-conv.csv", 3*init_batch*ker_size)
				ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in, bn_a, bn_b, alpha, pow, in_wids[0], raw_in_wids[0], ker_wid, 3, init_batch, norm[0], 0, step[0], 2, 0, "Conv", fast_pack, debug)
				// pow = mid_pow
			} else if i == 2 {
				ker_in := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-conv.csv", init_batch*real_batch[0]*ker_size)
				ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in, bn_a, bn_b, alpha, pow, in_wids[0], raw_in_wids[0], ker_wid, init_batch, real_batch[0], norm[0], 0, step[0], 2, 0, "Conv", fast_pack, debug)
			} else {
				ker_in := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-conv.csv", real_batch[0]*real_batch[0]*ker_size)
				ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in, bn_a, bn_b, alpha, pow, in_wids[0], raw_in_wids[0], ker_wid, real_batch[0], real_batch[0], norm[0], 0, step[0], 2, 0, "Conv", fast_pack, debug)
			}
			fmt.Println("Block1, Layer ", i, "done!")
		}
		fmt.Println("Block1 done.")
		timings[0] = time.Since(start).Seconds()
		start = time.Now()

		ker_in12 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0])+"-conv.csv", real_batch[0]*real_batch[1]*ker_size)
		ker_in12_new := make([]float64, 2*real_batch[0]*real_batch[1]*ker_size)
		ker_in12_0 := make([]float64, len(ker_in12)/2)
		ker_in12_1 := make([]float64, len(ker_in12)/2)
		if wide_case == 2 {
			for k := 0; k < ker_size; k++ {
				for i := 0; i < real_batch[0]; i++ {
					for j := 0; j < real_batch[1]; j++ {
						ker_in12_new[k*2*real_batch[0]*real_batch[1]+2*i*real_batch[1]+j] = ker_in12[k*real_batch[0]*real_batch[1]+i*real_batch[1]+j]
					}
				}
			}
		} else if wide_case == 3 {
			for k := 0; k < ker_size; k++ {
				for i := 0; i < real_batch[0]; i++ {
					for j := 0; j < real_batch[1]/2; j++ {
						ker_in12_0[k*real_batch[0]*real_batch[1]/2+(i*real_batch[1]/2+j)] = ker_in12[k*real_batch[0]*real_batch[1]+(i*real_batch[1]+2*j)]   // [i][2*j]
						ker_in12_1[k*real_batch[0]*real_batch[1]/2+(i*real_batch[1]/2+j)] = ker_in12[k*real_batch[0]*real_batch[1]+(i*real_batch[1]+2*j+1)] // [i][2*j+1]
					}
				}
			}
		}

		bn_a := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0])+"-a.csv", real_batch[1])
		bn_b := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0])+"-b.csv", real_batch[1])

		if wide_case == 2 {
			ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in12_new, bn_a, bn_b, alpha, pow, in_wids[0], 2*raw_in_wids[1], ker_wid, real_batch[1], real_batch[1], norm[0]/2, 0, step[1], 2, 0, "StrConv_odd", fast_pack, debug)
		} else if wide_case == 3 {
			bn_a_0 := make([]float64, real_batch[1]/2)
			bn_a_1 := make([]float64, real_batch[1]/2)
			bn_b_0 := make([]float64, real_batch[1]/2)
			bn_b_1 := make([]float64, real_batch[1]/2)
			for i := range bn_b_0 {
				bn_a_0[i] = bn_a[2*i]
				bn_a_1[i] = bn_a[2*i+1]
				bn_b_0[i] = bn_b[2*i]
				bn_b_1[i] = bn_b[2*i+1]
			}
			ct_result1 := evalConv_BNRelu_new(cont, ct_layer, ker_in12_0, bn_a_0, bn_b_0, alpha, pow, in_wids[0], 2*raw_in_wids[1], ker_wid, real_batch[0], real_batch[0], norm[0], 0, step[1], 2, 0, "StrConv_odd", fast_pack, debug)
			ct_result2 := evalConv_BNRelu_new(cont, ct_layer, ker_in12_1, bn_a_1, bn_b_1, alpha, pow, in_wids[0], 2*raw_in_wids[1], ker_wid, real_batch[0], real_batch[0], norm[0], 2, step[1], 2, 0, "StrConv_odd", fast_pack, debug)
			ct_layer = cont.evaluator.AddNew(ct_result1, ct_result2)
		}
		fmt.Println("Block1 to 2 done!")
		if debug {
			max_bat := cont.N / (in_wids[1] * in_wids[1])
			res_ttmp := cont.encoder.DecodeCoeffs(cont.decryptor.DecryptNew(ct_layer))
			prt_mat_norm_step(res_ttmp, max_bat, norm[1], step[1], prt_start[1], 3, false)
		}
		timings[1] = time.Since(start).Seconds()
		start = time.Now()

		// ResNet Block 2
		for i := 1; i <= num_blcs[1]; i++ {
			if i == 5 {
				pow = init_pow
			}
			bn_a2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+i)+"-a.csv", real_batch[1])
			bn_b2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+i)+"-b.csv", real_batch[1])
			ker_in2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+i)+"-conv.csv", real_batch[1]*real_batch[1]*ker_size)

			ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in2, bn_a2, bn_b2, alpha, pow, in_wids[1], raw_in_wids[1], ker_wid, real_batch[1], real_batch[1], norm[1], 0, step[1], 2, 0, "Conv_inside", fast_pack, debug)
			fmt.Println("Block2, Layer ", i, "done!")
		}
		fmt.Println("Block2 done.")
		timings[2] = time.Since(start).Seconds()
		start = time.Now()

		pow = mid_pow
		ker_in23 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+1)+"-conv.csv", real_batch[1]*real_batch[2]*ker_size)
		bn_a3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+1)+"-a.csv", real_batch[2])
		bn_b3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+1)+"-b.csv", real_batch[2])
		ker_in23_new := make([]float64, 2*real_batch[1]*real_batch[2]*ker_size)
		for k := 0; k < ker_size; k++ {
			for i := 0; i < real_batch[1]; i++ {
				for j := 0; j < real_batch[2]; j++ {
					ker_in23_new[k*2*real_batch[1]*real_batch[2]+2*i*real_batch[2]+j] = ker_in23[k*real_batch[1]*real_batch[2]+i*real_batch[2]+j]
				}
			}
		}
		ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in23_new, bn_a3, bn_b3, alpha, pow, in_wids[1], raw_in_wids[2], ker_wid, real_batch[2], real_batch[2], norm[2], 0, step[2], 2, 0, "StrConv_inside", fast_pack, debug)
		fmt.Println("Block2 to 3 done!")
		if debug {
			max_bat := cont.N / (in_wids[1] * in_wids[1])
			res_ttmp := cont.encoder.DecodeCoeffs(cont.decryptor.DecryptNew(ct_layer))
			prt_mat_norm_step(res_ttmp, max_bat, norm[2], step[2], prt_start[2], 3, false)
		}
		timings[3] = time.Since(start).Seconds()
		start = time.Now()

		// ResNet Block 3
		for i := 1; i <= num_blcs[2]; i++ {
			if i == 3 {
				pow = init_pow
			}
			if i == 5 {
				pow = mid_pow
			}
			bn_a3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+i+1)+"-a.csv", real_batch[2])
			bn_b3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+i+1)+"-b.csv", real_batch[2])
			ker_in3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+i+1)+"-conv.csv", real_batch[2]*real_batch[2]*ker_size)

			if i == num_blcs[2] {
				pow = final_pow
			}
			ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in3, bn_a3, bn_b3, alpha, pow, in_wids[1], raw_in_wids[2], ker_wid, real_batch[2], real_batch[2], norm[2], 0, step[2], 2, 0, "Conv_inside", fast_pack, debug)
			fmt.Println("Block3, Layer ", i, "done!")
		}
		fmt.Println("Block3 done.")
		timings[4] = time.Since(start).Seconds()
		start = time.Now()

		ker_inf_wid := raw_in_wids[1]
		if ker_inf_wid%2 == 0 {
			ker_inf_wid++
		}
		ker_inf := readTxt(weight_dir+"final-fckernel.csv", real_batch[2]*fc_out)
		ker_inf_ := make([]float64, ker_inf_wid*ker_inf_wid*real_batch[2]*fc_out)
		for i := range ker_inf {
			for b := 0; b < ker_inf_wid*ker_inf_wid; b++ {
				ker_inf_[i+b*real_batch[2]*fc_out] = ker_inf[i]
			}
		}
		bn_af := make([]float64, fc_out)
		for i := range bn_af {
			bn_af[i] = 1.0 / float64(raw_in_wids[2]*raw_in_wids[2]) // for reduce mean on raw_in_wids[2]**2 elements
		}
		bn_bf := readTxt(weight_dir+"final-fcbias.csv", fc_out)
		ct_result := evalConv_BN(cont, ct_layer, ker_inf_, bn_af, bn_bf, in_wids[1], ker_inf_wid, real_batch[2], fc_out, norm[2], float64(1<<30), false)
		fmt.Println("Final FC done.")
		timings[5] = time.Since(start).Seconds()
		start = time.Now()

		fmt.Println()
		fmt.Println("===============  DECRYPTION  ===============")
		fmt.Println()
		cont.decryptor.Decrypt(ct_result, pl_input)
		res_tmp := cont.encoder.DecodeCoeffs(pl_input)
		fmt.Printf("Decryption Done in %s \n", time.Since(start))
		res_out := prt_mat_one_norm(res_tmp, max_batch[1], norm[2], ker_inf_wid/2+1, ker_inf_wid/2+1)
		// fmt.Print(res_out)
		fmt.Println("\n result: ", res_out[:fc_out])
		writeTxt(out_dir+"class_result_"+ker_name+"_"+strconv.Itoa(iter)+".csv", res_out[:fc_out])

		fmt.Println("Blc1: ", timings[0], " sec")
		fmt.Println("Blc1->2: ", timings[1], " sec")
		fmt.Println("Blc2: ", timings[2], " sec")
		fmt.Println("Blc2->3: ", timings[3], " sec")
		fmt.Println("Blc3: ", timings[4], " sec")
		fmt.Println("Final (reduce_mean & FC): ", timings[5], " sec")
		fmt.Printf("Total done in %s \n", time.Since(begin_start))
	}
}

func testImagenet_final_fast_in(st, end, ker_wid int) {
	// We use full packing: i.e., in_wid**2 element is contained in po2_in_wid**2 sized block <-> half padding of Resnet
	// So ReLU, keep or rot, StoC done on both the 1st & 2nd part of the CtoS ciphertexts
	ker_name := "ker" + strconv.Itoa(ker_wid) // "ker5"
	weight_dir := "weight_imgnet_" + ker_name + "_h5/"
	logN := 16
	raw_in_wids := []int{14, 7}   // same as python
	real_batch := []int{256, 512} // same as python
	iter := 2
	in_wids := make([]int, len(raw_in_wids))
	kp_wids := make([]int, len(raw_in_wids))
	var num_blc1, num_blc2 int
	if ker_name == "ker3" {
		in_wids[0] = 16
		in_wids[1] = 8
		kp_wids[0] = 14
		kp_wids[1] = 7
		num_blc1 = 3
		num_blc2 = 3
	} else if ker_name == "ker5" {
		in_wids[0] = 16
		in_wids[1] = 8
		kp_wids[0] = 14
		kp_wids[1] = 6
		num_blc1 = 3
		num_blc2 = 3
	} else {
		panic("strange ker name!")
	}
	cont := newContext(logN, ker_wid, in_wids, kp_wids, true, "Imagenet_final_fast")

	ker_size := ker_wid * ker_wid
	max_batch := make([]int, len(real_batch)) // the max batch
	for i := range max_batch {
		max_batch[i] = cont.N / (in_wids[i] * in_wids[i])
	}
	alpha := 0.0 // 0.3 => leakyrelu
	init_pow := 6.0
	mid_pow := 5.0
	final_pow := 6.0

	// ker5_iter := []int{804, 886, 901, 956}
	// {3, 29, 87, 254, 357,
	// 399, 435, 455, 475, 476,
	// 518, 540, 545, 571, 631,
	// 657, 699, 711, 748, 790,
	// 804, 886, 901, 956}

	// for _, name_iter := range ker5_iter {
	for name_iter := st; name_iter < end; name_iter++ {
		weight_num := 10
		norm := 1
		fmt.Println("Start ", name_iter, "-th iter..")

		raw_input := readTxt(ker_name+"_data/test_image_"+strconv.Itoa(name_iter)+".csv", raw_in_wids[0]*raw_in_wids[0]*real_batch[0])
		input := make([]float64, in_wids[0]*in_wids[0]*real_batch[0])
		for i := 0; i < raw_in_wids[0]; i++ {
			for j := 0; j < raw_in_wids[0]; j++ {
				for b := 0; b < real_batch[0]; b++ {
					input[i*in_wids[0]*real_batch[0]+j*real_batch[0]+b] = raw_input[i*raw_in_wids[0]*real_batch[0]+j*real_batch[0]+b]
				}
			}
		}
		fmt.Println("Input: ")
		prt_mat(input, max_batch[0], 1)
		fmt.Println("vec size: ", cont.N)
		fmt.Println("input width: ", raw_in_wids)
		fmt.Println("kernel width: ", ker_wid)
		fmt.Println("num batches: ", real_batch)

		start := time.Now()
		pl_input := ckks.NewPlaintext(cont.params, cont.ECD_LV, cont.params.Scale()) // contain plaintext values
		cont.encoder.EncodeCoeffs(input, pl_input)
		ct_input := cont.encryptor.EncryptNew(pl_input)
		fmt.Printf("Encryption done in %s \n", time.Since(start))

		timings := make([]float64, 4)
		begin_start := time.Now()
		new_start := time.Now()

		// Block 1
		pow := init_pow
		ct_layer := ct_input
		for i := 1; i <= num_blc1; i++ {
			ker_in1 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-conv.csv", real_batch[0]*real_batch[0]*ker_size)
			weight_num++
			bn_a1 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-a.csv", real_batch[0])
			bn_b1 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-b.csv", real_batch[0])
			if i == num_blc1 {
				pow = mid_pow
			}
			ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in1, bn_a1, bn_b1, alpha, pow, in_wids[0], kp_wids[0], ker_wid, real_batch[0], real_batch[0], norm, 0, 0, iter, 0, "Conv", false, false)
			fmt.Println("Block1, Layer ", i, "done!")
		}
		fmt.Println("Block1 done!")
		timings[0] = time.Since(new_start).Seconds()
		new_start = time.Now()

		ker_in12 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-conv.csv", real_batch[0]*real_batch[1]*ker_size)
		weight_num++
		bn_a12 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-a.csv", real_batch[1])
		bn_b12 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-b.csv", real_batch[1])

		ker_in12_0 := make([]float64, len(ker_in12)/2)
		ker_in12_1 := make([]float64, len(ker_in12)/2)
		for k := 0; k < ker_size; k++ {
			for i := 0; i < real_batch[0]; i++ {
				for j := 0; j < real_batch[1]/2; j++ {
					ker_in12_0[k*real_batch[0]*real_batch[1]/2+(i*real_batch[1]/2+j)] = ker_in12[k*real_batch[0]*real_batch[1]+(i*real_batch[1]+j)]                 // [i][j]
					ker_in12_1[k*real_batch[0]*real_batch[1]/2+(i*real_batch[1]/2+j)] = ker_in12[k*real_batch[0]*real_batch[1]+(i*real_batch[1]+real_batch[1]/2+j)] // [i][j+B/2]
				}
			}
		}
		bn_a12_0 := make([]float64, real_batch[1]/2)
		bn_a12_1 := make([]float64, real_batch[1]/2)
		bn_b12_0 := make([]float64, real_batch[1]/2)
		bn_b12_1 := make([]float64, real_batch[1]/2)
		for i := range bn_b12_0 {
			bn_a12_0[i] = bn_a12[i]
			bn_a12_1[i] = bn_a12[i+real_batch[1]/2]
			bn_b12_0[i] = bn_b12[i]
			bn_b12_1[i] = bn_b12[i+real_batch[1]/2]
		}

		// block1 to block 2
		ct_result1 := evalConv_BNRelu_new(cont, ct_layer, ker_in12_0, bn_a12_0, bn_b12_0, alpha, pow, in_wids[0], 2*kp_wids[1], ker_wid, real_batch[0], real_batch[0], norm, 0, 0, iter, 0, "StrConv", false, false)
		ct_result2 := evalConv_BNRelu_new(cont, ct_layer, ker_in12_1, bn_a12_1, bn_b12_1, alpha, pow, in_wids[0], 2*kp_wids[1], ker_wid, real_batch[0], real_batch[0], norm, 1, 0, iter, 0, "StrConv", false, false)
		ct_layer = cont.evaluator.AddNew(ct_result1, ct_result2)
		fmt.Println("Block1 to 2 done!")
		// res_tmp := cont.encoder.DecodeCoeffs(cont.decryptor.DecryptNew(ct_result))
		// prt_mat_norm(res_tmp, max_batch[1], 1, 4, false)
		timings[1] = time.Since(new_start).Seconds()
		new_start = time.Now()

		// Block 2
		for i := 1; i <= num_blc2; i++ {
			ker_in2 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-conv.csv", real_batch[1]*real_batch[1]*ker_size)
			weight_num++
			bn_a2 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-a.csv", real_batch[1])
			bn_b2 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-b.csv", real_batch[1])
			if i == num_blc2 {
				pow = final_pow
			}
			ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in2, bn_a2, bn_b2, alpha, pow, in_wids[1], kp_wids[1], ker_wid, real_batch[1], real_batch[1], norm, 0, 0, iter, 0, "Conv", false, false)
			fmt.Println("Block2, Layer ", i, "done!")
		}
		fmt.Println("Block2 done!")
		timings[2] = time.Since(new_start).Seconds()
		new_start = time.Now()

		// RMFC
		fin_out_batch := 1000
		ker_inf := readTxt(weight_dir+"fc.csv", real_batch[1]*fin_out_batch)
		bn_af := make([]float64, real_batch[1]*2)
		if ker_wid == 3 {
			for i := range bn_af {
				bn_af[i] = 1.0 / (7 * 7) // for reduce mean on 8*8 elements
			}
		} else {
			for i := range bn_af {
				bn_af[i] = 1.0 / (6 * 6) // for reduce mean on 8*8 elements
			}
		}
		bn_bf := make([]float64, real_batch[1]*2)
		for i := range bn_bf {
			bn_bf[i] = 0.0 //10.0 * float64(i)
		}
		ker_inf_ := make([]float64, 7*7*real_batch[1]*fin_out_batch)
		for b := 0; b < 7*7; b++ {
			for i := 0; i < real_batch[1]; i++ {
				for j := 0; j < fin_out_batch; j++ {
					ker_inf_[b*real_batch[1]*fin_out_batch+i*fin_out_batch+j] = ker_inf[i*fin_out_batch+j]
				}
			}
		}
		ct_result := evalConv_BN(cont, ct_layer, ker_inf_, bn_af, bn_bf, in_wids[1], 7, real_batch[1], 1000, 1, float64(1<<30), false)
		timings[3] = time.Since(new_start).Seconds()
		new_start = time.Now()

		cont.decryptor.Decrypt(ct_result, pl_input)
		res_tmp := cont.encoder.DecodeCoeffs(pl_input)
		fmt.Printf("Decryption done in %s \n", time.Since(new_start))
		final_result := prt_mat_one_norm(res_tmp, max_batch[1], 1, 4, 4)
		writeTxt(ker_name+"_enc_result/enc_result_"+strconv.Itoa(name_iter)+".csv", final_result[:1000])

		fmt.Println("Blc1: ", timings[0], " sec")
		fmt.Println("Blc1->2: ", timings[1], " sec")
		fmt.Println("Blc2: ", timings[2], " sec")
		fmt.Println("Final (reduce_mean & FC): ", timings[3], " sec")
		fmt.Printf("Total done in %s \n", time.Since(begin_start))
	}
}

func testImagenet_sparse(st, end, ker_wid int) {
	// We use full packing: i.e., in_wid**2 element is contained in po2_in_wid**2 sized block <-> half padding of Resnet
	// So ReLU, keep or rot, StoC done on both the 1st & 2nd part of the CtoS ciphertexts
	debug := false
	ker_name := "ker" + strconv.Itoa(ker_wid) // "ker5"
	weight_dir := "weight_imgnet_" + ker_name + "_h5/"
	logN := 16
	raw_in_wids := []int{14, 7}   // same as python
	real_batch := []int{256, 512} // same as python
	log_sparse := []int{0, 1}
	norm := []int{1, 2}
	iter := 2
	in_wids := make([]int, len(raw_in_wids))
	kp_wids := make([]int, len(raw_in_wids))
	var num_blc1, num_blc2 int
	if ker_name == "ker3" {
		in_wids[0] = 16
		in_wids[1] = 8
		kp_wids[0] = 14
		kp_wids[1] = 7
		num_blc1 = 3
		num_blc2 = 3
	} else if ker_name == "ker5" {
		in_wids[0] = 16
		in_wids[1] = 8
		kp_wids[0] = 14
		kp_wids[1] = 6
		num_blc1 = 3
		num_blc2 = 3
	} else {
		panic("strange ker name!")
	}
	cont := newContext(logN, ker_wid, in_wids, kp_wids, true, "Imagenet_sparse")

	ker_size := ker_wid * ker_wid
	max_batch := make([]int, len(real_batch)) // the max batch
	for i := range max_batch {
		max_batch[i] = cont.N / (in_wids[i] * in_wids[i])
	}
	alpha := 0.0 // 0.3 => leakyrelu
	init_pow := 6.0
	mid_pow := 5.0
	final_pow := 6.0

	for name_iter := st; name_iter < end; name_iter++ {
		weight_num := 10
		fmt.Println("Start ", name_iter, "-th iter..")

		raw_input := readTxt(ker_name+"_data/test_image_"+strconv.Itoa(name_iter)+".csv", raw_in_wids[0]*raw_in_wids[0]*real_batch[0])
		input := make([]float64, in_wids[0]*in_wids[0]*real_batch[0])
		for i := 0; i < raw_in_wids[0]; i++ {
			for j := 0; j < raw_in_wids[0]; j++ {
				for b := 0; b < real_batch[0]; b++ {
					input[i*in_wids[0]*real_batch[0]+j*real_batch[0]+b] = raw_input[i*raw_in_wids[0]*real_batch[0]+j*real_batch[0]+b]
				}
			}
		}
		fmt.Println("Input: ")
		prt_mat(input, max_batch[0], 1)
		fmt.Println("vec size: ", cont.N)
		fmt.Println("input width: ", raw_in_wids)
		fmt.Println("kernel width: ", ker_wid)
		fmt.Println("num batches: ", real_batch)

		start := time.Now()
		pl_input := ckks.NewPlaintext(cont.params, cont.ECD_LV, cont.params.Scale()) // contain plaintext values
		cont.encoder.EncodeCoeffs(input, pl_input)
		ct_input := cont.encryptor.EncryptNew(pl_input)
		fmt.Printf("Encryption done in %s \n", time.Since(start))

		timings := make([]float64, 4)
		begin_start := time.Now()
		new_start := time.Now()

		// Block 1
		pow := init_pow
		ct_layer := ct_input
		for i := 1; i <= num_blc1; i++ {
			ker_in1 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-conv.csv", real_batch[0]*real_batch[0]*ker_size)
			weight_num++
			bn_a1 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-a.csv", real_batch[0])
			bn_b1 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-b.csv", real_batch[0])
			if i == num_blc1 {
				pow = mid_pow
			}
			ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in1, bn_a1, bn_b1, alpha, pow, in_wids[0], kp_wids[0], ker_wid, real_batch[0], real_batch[0], norm[0], 0, 1, iter, log_sparse[0], "Conv_sparse", true, debug)
			fmt.Println("Block1, Layer ", i, "done!")
		}
		fmt.Println("Block1 done!")
		timings[0] = time.Since(new_start).Seconds()
		new_start = time.Now()

		ker_in12 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-conv.csv", real_batch[0]*real_batch[1]*ker_size)
		weight_num++
		bn_a12 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-a.csv", real_batch[1])
		bn_b12 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-b.csv", real_batch[1])

		ker_in12_0 := make([]float64, len(ker_in12)/2)
		ker_in12_1 := make([]float64, len(ker_in12)/2)
		for k := 0; k < ker_size; k++ {
			for i := 0; i < real_batch[0]; i++ {
				for j := 0; j < real_batch[1]/2; j++ {
					// ker_in12_0[k*real_batch[0]*real_batch[1]/2+(i*real_batch[1]/2+j)] = ker_in12[k*real_batch[0]*real_batch[1]+(i*real_batch[1]+j)]                 // [i][j]
					// ker_in12_1[k*real_batch[0]*real_batch[1]/2+(i*real_batch[1]/2+j)] = ker_in12[k*real_batch[0]*real_batch[1]+(i*real_batch[1]+real_batch[1]/2+j)] // [i][j+B/2]
					ker_in12_0[k*real_batch[0]*real_batch[1]/2+(i*real_batch[1]/2+j)] = ker_in12[k*real_batch[0]*real_batch[1]+(i*real_batch[1]+2*j)]   // [i][2*j]
					ker_in12_1[k*real_batch[0]*real_batch[1]/2+(i*real_batch[1]/2+j)] = ker_in12[k*real_batch[0]*real_batch[1]+(i*real_batch[1]+2*j+1)] // [i][2*j+1]

				}
			}
		}
		bn_a12_0 := make([]float64, real_batch[1]/2)
		bn_a12_1 := make([]float64, real_batch[1]/2)
		bn_b12_0 := make([]float64, real_batch[1]/2)
		bn_b12_1 := make([]float64, real_batch[1]/2)
		for i := range bn_b12_0 {
			// bn_a12_0[i] = bn_a12[i]
			// bn_a12_1[i] = bn_a12[i+real_batch[1]/2]
			// bn_b12_0[i] = bn_b12[i]
			// bn_b12_1[i] = bn_b12[i+real_batch[1]/2]
			bn_a12_0[i] = bn_a12[2*i]
			bn_a12_1[i] = bn_a12[2*i+1]
			bn_b12_0[i] = bn_b12[2*i]
			bn_b12_1[i] = bn_b12[2*i+1]
		}

		// block1 to block 2
		// ct_result1 := evalConv_BNRelu_new(cont, ct_layer, ker_in12_0, bn_a12_0, bn_b12_0, alpha, pow, in_wids[0], 2*kp_wids[1], ker_wid, real_batch[0], real_batch[0], norm, 0, 0, iter, 0, "StrConv", false, false)
		// ct_result2 := evalConv_BNRelu_new(cont, ct_layer, ker_in12_1, bn_a12_1, bn_b12_1, alpha, pow, in_wids[0], 2*kp_wids[1], ker_wid, real_batch[0], real_batch[0], norm, 1, 0, iter, 0, "StrConv", false, false)
		// ct_layer = cont.evaluator.AddNew(ct_result1, ct_result2)

		ct_result1 := evalConv_BNRelu_new(cont, ct_layer, ker_in12_0, bn_a12_0, bn_b12_0, alpha, pow, in_wids[0], kp_wids[1], ker_wid, real_batch[0], real_batch[1]/2, norm[0], 0, 1, 2, 0, "StrConv_sparse_full", true, debug)
		ct_result2 := evalConv_BNRelu_new(cont, ct_layer, ker_in12_1, bn_a12_1, bn_b12_1, alpha, pow, in_wids[0], kp_wids[1], ker_wid, real_batch[0], real_batch[1]/2, norm[0], 0, 1, 2, 0, "StrConv_sparse_full", true, debug)

		xi := make([]float64, cont.N)
		xi[2] = 1.0
		xi_plain := ckks.NewPlaintext(cont.params, ct_result2.Level(), 1.0)
		cont.encoder.EncodeCoeffs(xi, xi_plain)
		cont.encoder.ToNTT(xi_plain)
		ct_result2 = cont.evaluator.MulNew(ct_result2, xi_plain)
		ct_layer = cont.evaluator.AddNew(ct_result1, ct_result2)

		fmt.Println("Block1 to 2 done!")
		// res_tmp := cont.encoder.DecodeCoeffs(cont.decryptor.DecryptNew(ct_result))
		// prt_mat_norm(res_tmp, max_batch[1], 1, 4, false)
		timings[1] = time.Since(new_start).Seconds()
		new_start = time.Now()

		// Block 2
		for i := 1; i <= num_blc2; i++ {
			ker_in2 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-conv.csv", real_batch[1]*real_batch[1]*ker_size)
			weight_num++
			bn_a2 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-a.csv", real_batch[1])
			bn_b2 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-b.csv", real_batch[1])
			if i == num_blc2 {
				pow = final_pow
			}
			ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in2, bn_a2, bn_b2, alpha, pow, in_wids[1], kp_wids[1], ker_wid, real_batch[1], real_batch[1], norm[1], 0, 1, iter, log_sparse[1], "Conv_sparse", true, debug)
			fmt.Println("Block2, Layer ", i, "done!")
		}
		fmt.Println("Block2 done!")
		timings[2] = time.Since(new_start).Seconds()
		new_start = time.Now()

		// RMFC
		fin_out_batch := 1000
		ker_inf := readTxt(weight_dir+"fc.csv", real_batch[1]*fin_out_batch)
		bn_af := make([]float64, real_batch[1]*2)
		if ker_wid == 3 {
			for i := range bn_af {
				bn_af[i] = 1.0 / (7 * 7) // for reduce mean on 8*8 elements
			}
		} else {
			for i := range bn_af {
				bn_af[i] = 1.0 / (6 * 6) // for reduce mean on 8*8 elements
			}
		}
		bn_bf := make([]float64, real_batch[1]*2)
		for i := range bn_bf {
			bn_bf[i] = 0.0 //10.0 * float64(i)
		}
		ker_inf_ := make([]float64, 7*7*real_batch[1]*fin_out_batch)
		for b := 0; b < 7*7; b++ {
			for i := 0; i < real_batch[1]; i++ {
				for j := 0; j < fin_out_batch; j++ {
					ker_inf_[b*real_batch[1]*fin_out_batch+i*fin_out_batch+j] = ker_inf[i*fin_out_batch+j]
				}
			}
		}
		ct_result := evalConv_BN(cont, ct_layer, ker_inf_, bn_af, bn_bf, in_wids[1], 7, real_batch[1], fin_out_batch, 1, float64(1<<30), false)
		fmt.Println("Final FC done.")
		timings[3] = time.Since(new_start).Seconds()
		new_start = time.Now()

		cont.decryptor.Decrypt(ct_result, pl_input)
		res_tmp := cont.encoder.DecodeCoeffs(pl_input)
		fmt.Printf("Decryption done in %s \n", time.Since(new_start))
		final_result := prt_mat_one_norm(res_tmp, max_batch[1], 1, 4, 4)
		writeTxt(ker_name+"_enc_result/enc_result_"+strconv.Itoa(name_iter)+".csv", final_result[:1000])

		fmt.Println("Blc1: ", timings[0], " sec")
		fmt.Println("Blc1->2: ", timings[1], " sec")
		fmt.Println("Blc2: ", timings[2], " sec")
		fmt.Println("Final (reduce_mean & FC): ", timings[3], " sec")
		fmt.Printf("Total done in %s \n", time.Since(begin_start))
	}
}
