package main

import (
	"fmt"
	"math"
	"time"

	"github.com/dwkim606/test_lattigo/ckks"
)

// take raw_in_wid then outputs appropriate kp_wid and out_batch
// only for our convs Test (not for BL)
func set_Variables(batch, raw_in_wid, in_wid, ker_wid int, kind string) (kp_wid, out_batch, logN int, trans bool) {
	N := batch * in_wid * in_wid
	logN = 0
	for ; (1 << logN) < N; logN++ {
	}
	max_kp_wid := in_wid - ((ker_wid - 1) / 2) // max possible size of raw_in_wid

	switch kind {
	case "Conv":
		trans = false
		kp_wid = raw_in_wid
		out_batch = batch
		if kp_wid > max_kp_wid {
			fmt.Println("max raw_in_wid: ", max_kp_wid)
			panic("too large raw_in_wid.")
		}
	case "StrConv", "StrConv_fast", "StrConv_odd":
		trans = false
		kp_wid = 2 * (in_wid/2 - ker_wid/2)
		out_batch = batch
		if kp_wid > max_kp_wid {
			fmt.Println("max raw_in_wid: ", max_kp_wid)
			panic("too large raw_in_wid.")
		}
	case "StrConv_inside":
		trans = false
		kp_wid = (in_wid/2 - ker_wid/2)
		out_batch = batch
	case "TransConv":
		trans = true
		kp_wid = 2 * raw_in_wid
		out_batch = batch / 4
		if kp_wid > max_kp_wid {
			fmt.Println("max raw_in_wid: ", max_kp_wid/2)
			panic("too large raw_in_wid.")
		}
	default:
		panic("Wrong kinds!")
	}

	return
}

// apply rotation for strided conv (compress) or transposed conv (extend)
// the same rotation for all batches; use BSGS to reduce rotations
// assume that input batches are well-ordered. compress: (0,4) (1,5) (2,6) (3,7) to (0,1,2,...6,7) extend: (0,2,4,6,1,3,5,7) to (0,1) (2,3) (4,5) (6,7)
// rotation for each batch position (0 to 3) is applied after or before compress or extend, resp.
// total rotation = 2*in_wid*4 + (4-1); depth = 2
func evalRot_BL(cont *context, ct_input *ckks.Ciphertext, in_wid, pos int, trans bool) (ct_res *ckks.Ciphertext) {
	if trans {
		in_size := in_wid * in_wid
		cont.evaluator.Rotate(ct_input, pos*in_size, ct_input)
		ct_res = bsgs_ctxt(cont.evaluator, cont.encoder, ct_input, cont.m_idx[in_wid][0], cont.r_idx[in_wid][0], cont.params)
	} else {
		out_size := in_wid * in_wid / 4
		ct_res = bsgs_ctxt(cont.evaluator, cont.encoder, ct_input, cont.m_idx[in_wid][0], cont.r_idx[in_wid][0], cont.params)
		cont.evaluator.Rotate(ct_res, -pos*out_size, ct_res)
	}
	return
}

// Eval Conv only, always assume max batch
// in_wid must be Po2 (also include padding), includes kernel preparation
// norm == 1 : normal case, norm == 4 : in & out batch is (1,0,0,0,2,0,0,0,3,0,0,0,4,0,0,0)
// for test, use pack_evaluator optimizer
func evalConv_BN_BL_test(cont *context, ct_input *ckks.Ciphertext, ker_in, bn_a, bn_b []float64, in_wid, ker_wid, real_ib, real_ob, pos, norm, pad int, trans, printResult bool) (ct_res *ckks.Ciphertext) {
	in_size := in_wid * in_wid
	out_size := in_size
	max_batch := cont.N / (2 * in_size)

	// fmt.Println()
	// fmt.Println("===============  (KER) PREPARATION  ===============")
	// fmt.Println()
	start := time.Now()
	max_ker_rs := reshape_ker_BL(ker_in, bn_a, ker_wid, real_ib, real_ob, max_batch, pos, norm, trans)
	scale_exp := cont.params.Scale() * cont.params.Scale()
	if trans {
		scale_exp = cont.params.Scale() * cont.params.Scale() * cont.params.Scale()
	}
	bn_b_slots := make([]complex128, cont.N/2)
	for i, elt := range bn_b {
		for j := 0; j < in_wid-pad; j++ {
			for k := 0; k < in_wid-pad; k++ {
				bn_b_slots[j+k*in_wid+norm*out_size*i] = complex(elt, 0)
			}
		}
	}

	pl_bn_b := ckks.NewPlaintext(cont.params, cont.ECD_LV, scale_exp)
	cont.encoder.EncodeNTT(pl_bn_b, bn_b_slots, cont.logN-1)
	fmt.Printf("Plaintext (kernel) preparation, Done in %s \n", time.Since(start))

	// fmt.Println()
	// fmt.Println("===============  EVALUATION  ===============")
	// fmt.Println()
	start = time.Now()
	ct_inputs_rots := preConv_BL(cont.pack_evaluator, ct_input, in_wid, ker_wid)
	fmt.Printf("preConv done in %s \n", time.Since(start))

	var rot_iters int
	if norm*real_ob == max_batch {
		rot_iters = real_ob
	} else {
		rot_iters = max_batch
	}
	for i := 0; i < rot_iters; i++ {
		ct_tmp := postConv_BL(cont.params, cont.encoder, cont.pack_evaluator, ct_inputs_rots, in_wid, ker_wid, norm*i, pad, max_ker_rs)
		if i == 0 {
			ct_res = ct_tmp
		} else {
			cont.evaluator.Add(ct_res, cont.pack_evaluator.RotateNew(ct_tmp, norm*i*out_size), ct_res)
		}
	}

	if ct_res.Scale != scale_exp {
		panic("Different scale between pl_bn_b and ctxt")
	}
	cont.evaluator.Add(ct_res, pl_bn_b, ct_res)
	fmt.Printf("Conv (with BN) Done in %s \n", time.Since(start))

	return ct_res
}

// reduce mean and final FC layer (in_batch -> 16)
// assume that ct_input has batch (1,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,3,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0)
// ker_fc is of size in_batch*10 and 1-dim from [64,10] shape
func evalRMFC_BL(cont *context, ct_input *ckks.Ciphertext, ker_fc, bias []float64, printResult bool) (ct_res *ckks.Ciphertext) {
	rs_ker := make([][]float64, 64)
	for i := 0; i < 64; i++ {
		rs_ker[i] = make([]float64, 16)
		for j := 0; j < 10; j++ {
			rs_ker[i][j] = ker_fc[j+i*10] / 64.0 // we will add 64 elts instead of averaging them
		}
	}

	// sum 64 elements instead of averaging them
	ct_avg := ct_input
	for i := 1; i < 64; i *= 2 {
		ct_avg = cont.evaluator.AddNew(ct_avg, cont.evaluator.RotateNew(ct_avg, i))
	}

	for i := 0; i < 16; i++ {
		tmp := make([]complex128, cont.N/2)
		for j := 0; j < 64; j++ {
			tmp[j*64*8] = complex(rs_ker[j][(j%16+16-i)%16], 0)
		}
		pl_ker := cont.encoder.EncodeNTTAtLvlNew(ct_avg.Level(), tmp, cont.logN-1)

		if i == 0 {
			ct_res = cont.evaluator.MulNew(ct_avg, pl_ker)
		} else {
			ct_tmp := cont.evaluator.MulNew(ct_avg, pl_ker)
			cont.evaluator.Add(ct_res, cont.evaluator.RotateNew(ct_tmp, i*64*8), ct_res)
		}
	}

	// final rotations to add up (4 = 64/16)
	for i := 1; i < 4; i *= 2 {
		ct_res = cont.evaluator.AddNew(ct_res, cont.evaluator.RotateNew(ct_res, i*16*64*8))
	}

	tmp := make([]complex128, cont.N/2)
	for j := 0; j < 10; j++ {
		tmp[j*64*8] = complex(bias[j], 0)
	}
	pl_bias := cont.encoder.EncodeNTTAtLvlNew(ct_res.Level(), tmp, cont.logN-1)
	cont.evaluator.Add(ct_res, pl_bias, ct_res)

	return
}

// reduce mean and final FC layer
// assume that ct_input has full batch (1,2,3,4,...)
// ker_fc is of size in_batch*out and 1-dim from [in_batch,out] shape
func evalRMFC_BL_img(cont *context, ct_input *ckks.Ciphertext, ker_fc []float64, in_batch, out_num, raw_in_wid int, printResult bool) (ct_res *ckks.Ciphertext) {
	rs_ker := make([][]float64, in_batch)
	for i := 0; i < in_batch; i++ {
		rs_ker[i] = make([]float64, out_num)
		for j := 0; j < out_num; j++ {
			rs_ker[i][j] = ker_fc[j+i*out_num] / float64(raw_in_wid*raw_in_wid) // we will add 64 elts instead of averaging them
		}
	}

	// sum 64 elements instead of averaging them (but only 49 elts are non-zero)
	ct_avg := ct_input
	for i := 1; i < 64; i *= 2 {
		ct_avg = cont.evaluator.AddNew(ct_avg, cont.evaluator.RotateNew(ct_avg, i))
	}

	for i := 0; i < in_batch; i++ {
		tmp := make([]complex128, cont.N/2)
		for j := 0; j < out_num; j++ {
			tmp[(i+j)%in_batch*64] = complex(rs_ker[(i+j)%in_batch][j], 0)
		}
		pl_ker := cont.encoder.EncodeNTTAtLvlNew(ct_avg.Level(), tmp, cont.logN-1)

		if i == 0 {
			ct_res = cont.evaluator.MulNew(ct_avg, pl_ker)
		} else {
			ct_tmp := cont.evaluator.MulNew(ct_avg, pl_ker)
			cont.evaluator.Add(ct_res, cont.evaluator.RotateNew(ct_tmp, i*64), ct_res)
		}
	}

	return
}

// Eval Conv only, always assume max batch
// in_wid must be Po2 (also include padding),
// include kernel preparation
// norm = 2 : in&out batches are (1,0,2,0,3,0,...)
func evalConv_BN(cont *context, ct_input *ckks.Ciphertext, ker_in, bn_a, bn_b []float64, in_wid, ker_wid, real_ib, real_ob, norm int, out_scale float64, trans bool) (ct_res *ckks.Ciphertext) {
	max_batch := cont.N / (in_wid * in_wid)

	// fmt.Println()
	// fmt.Println("===============  (KER) PREPARATION  ===============")
	// fmt.Println()
	start := time.Now()
	pl_ker := prep_Ker(cont.params, cont.encoder, ker_in, bn_a, in_wid, ker_wid, real_ib, real_ob, norm, cont.ECD_LV, 0, trans)
	// fmt.Printf("for prep_ker %s \n", time.Since(start))
	b_coeffs := make([]float64, cont.N)
	for i := range bn_b {
		for j := 0; j < in_wid*in_wid; j++ {
			b_coeffs[norm*i+j*max_batch] = bn_b[i]
		}
	}
	// scale_exp := ct_input.Scale * cont.params.Scale() * float64(max_batch/norm)
	pl_bn_b := ckks.NewPlaintext(cont.params, 0, out_scale)
	// pl_bn_b := ckks.NewPlaintext(cont.params, cont.ECD_LV, scale_exp) // contain plaintext values
	cont.encoder.EncodeCoeffs(b_coeffs, pl_bn_b)
	cont.encoder.ToNTT(pl_bn_b)
	fmt.Printf("Plaintext (kernel) preparation, Done in %s \n", time.Since(start))

	// fmt.Println()
	// fmt.Println("===============  EVALUATION  ===============")
	// fmt.Println()

	start = time.Now()
	ct_res = conv_then_pack(cont.params, cont.pack_evaluator, ct_input, pl_ker, cont.pl_idx, max_batch, norm, cont.ECD_LV, out_scale)
	if (pl_bn_b.Scale != ct_res.Scale) || (ct_res.Level() != 0) {
		fmt.Println("plain scale: ", pl_bn_b.Scale)
		fmt.Println("ctxt scale: ", ct_res.Scale)
		fmt.Println("ctxt lv: ", ct_res.Level())
		panic("LV or scale after conv then pack, inconsistent")
	}
	cont.evaluator.Add(ct_res, pl_bn_b, ct_res) // for Batch Normalization (BN)

	fmt.Printf("Conv (with BN) Done in %s \n", time.Since(start))

	return ct_res
}

// Eval Conv, BN, relu with Boot
// in_wid must be Po2 (also include padding)
// stride = true: apply [1,2,2,1] stride; false: [1,1,1,1]
// pack_pos: position to pack (0,1,2,3): only for strided case
// real_ib, real_ob: real number of batches (less or equal than max_batch)
// step: step of the output (used for conv_inside only)
// log_sparse: 0 if no full slot, 1 if half slot, etc (maybe the same as norm[])
func evalConv_BNRelu_new(cont *context, ct_input *ckks.Ciphertext, ker_in, bn_a, bn_b []float64, alpha, pow float64, in_wid, kp_wid, ker_wid, real_ib, real_ob, norm, pack_pos, step, iter, log_sparse int, kind string, fast_pack, debug bool) (ct_res *ckks.Ciphertext) {
	// iter := 2 // for full packing (contrary to half packing)
	var trans, stride, odd, inside bool
	odd = false
	trans = false
	stride = false
	inside = false
	sparse := false
	in_step := step
	modify_ker := false
	full := false
	switch kind {
	case "Conv_sparse": // sparse pack, normal conv
		sparse = true
	case "StrConv_sparse": // sparse pack, strided conv, 2 convs -> add them -> 1 boot
		modify_ker = true
		sparse = true
		stride = true
	case "StrConv_sparse_full": // sparse pack but full pack
		sparse = true
		modify_ker = true
		stride = true
		full = true
	case "Conv_inside":
		inside = true
	case "StrConv_inside":
		in_step = step / 2
		if step%2 != 0 {
			panic("step can not be divided by 2 (for strided conv)")
		}
		inside = true
	case "StrConv", "StrConv_fast":
		stride = true
	case "StrConv_odd":
		stride = true
		odd = true
	case "TransConv":
		trans = true
	case "Conv":
	default:
		panic("No kind!")
	}

	if odd { // multiply x^{offset} before conv to move input to appropriate position.
		odd_time := time.Now()
		var offset int // offset depends on the real_wid = in_wid-ker_wid/2 is even or not
		if (in_wid-ker_wid/2)%2 == 0 {
			offset = 0
		} else {
			offset = cont.N / (in_wid * in_wid) * (in_wid + 1)
			// offset = real_ib * norm * (in_wid + 1)
		}
		// fmt.Println("offset: ", offset)
		xi := make([]float64, cont.N)
		xi[offset] = 1.0
		xi_plain := ckks.NewPlaintext(cont.params, cont.ECD_LV, 1.0)
		cont.encoder.EncodeCoeffs(xi, xi_plain)
		cont.encoder.ToNTT(xi_plain)
		ct_input = cont.evaluator.MulNew(ct_input, xi_plain)
		fmt.Printf("for odd stride, offset time %s \n", time.Since(odd_time))
	}

	var ct_conv *ckks.Ciphertext
	if modify_ker {
		if !full {
			bn_a_0 := make([]float64, real_ib)
			bn_a_1 := make([]float64, real_ib)
			bn_b_0 := make([]float64, real_ib)
			bn_b_1 := make([]float64, real_ib)
			for i := range bn_b_0 {
				bn_a_0[i] = bn_a[2*i]
				bn_a_1[i] = bn_a[2*i+1]
				bn_b_0[i] = bn_b[2*i]
				bn_b_1[i] = bn_b[2*i+1]
			}
			ker_in_0 := make([]float64, len(ker_in)/2)
			ker_in_1 := make([]float64, len(ker_in)/2)
			for k := 0; k < ker_wid*ker_wid; k++ {
				for i := 0; i < real_ib; i++ {
					for j := 0; j < real_ob/2; j++ {
						ker_in_0[k*real_ib*real_ob/2+(i*real_ob/2+j)] = ker_in[k*real_ib*real_ob+(i*real_ob+2*j)]   // [i][2*j]
						ker_in_1[k*real_ib*real_ob/2+(i*real_ob/2+j)] = ker_in[k*real_ib*real_ob+(i*real_ob+2*j+1)] // [i][2*j+1]
					}
				}
			}
			ct_result1 := evalConv_BN(cont, ct_input, ker_in_0, bn_a_0, bn_b_0, in_wid, ker_wid, real_ib, real_ob/2, norm/2, math.Exp2(math.Round(math.Log2(float64(cont.params.Q()[0]))-(pow+8))), trans)
			ct_result2 := evalConv_BN(cont, ct_input, ker_in_1, bn_a_1, bn_b_1, in_wid, ker_wid, real_ib, real_ob/2, norm/2, math.Exp2(math.Round(math.Log2(float64(cont.params.Q()[0]))-(pow+8))), trans)

			xi := make([]float64, cont.N)
			offset := norm / 4 // cont.N / norm
			xi[offset] = 1.0
			xi_plain := ckks.NewPlaintext(cont.params, ct_result2.Level(), 1.0)
			cont.encoder.EncodeCoeffs(xi, xi_plain)
			cont.encoder.ToNTT(xi_plain)
			ct_result2 = cont.evaluator.MulNew(ct_result2, xi_plain)

			ct_conv = cont.evaluator.AddNew(ct_result1, ct_result2)

			// res_tmp := cont.encoder.DecodeCoeffs(cont.decryptor.DecryptNew(ct_conv))
			max_batch := cont.N / (in_wid * in_wid)
			// prt_mat_norm_step(res_tmp, max_batch, norm/4, step, 1, 4, false)

			for i := range xi {
				xi[i] = 0.0
			}
			if (in_wid-ker_wid/2)%2 != 0 {
				xi[0] = 1.0
			} else {
				// fmt.Println("offset nonzero!")
				offset = cont.N - (max_batch)*(in_wid+1)
				xi[offset] = -1.0
			}
			xi_plain = ckks.NewPlaintext(cont.params, ct_conv.Level(), 1.0)
			cont.encoder.EncodeCoeffs(xi, xi_plain)
			cont.encoder.ToNTT(xi_plain)
			ct_conv = cont.evaluator.MulNew(ct_conv, xi_plain)
			// res_tmp = cont.encoder.DecodeCoeffs(cont.decryptor.DecryptNew(ct_conv))
			// fmt.Println("After offset: ")
			// prt_mat_norm_step(res_tmp, max_batch, norm/4, step, 1, 4, false)
		} else { // need to cover the case with full packing
			ct_conv = evalConv_BN(cont, ct_input, ker_in, bn_a, bn_b, in_wid, ker_wid, real_ib, real_ob, norm, math.Exp2(math.Round(math.Log2(float64(cont.params.Q()[0]))-(pow+8))), trans)

			// res_tmp := cont.encoder.DecodeCoeffs(cont.decryptor.DecryptNew(ct_conv))
			max_batch := cont.N / (in_wid * in_wid)
			// prt_mat_norm_step(res_tmp, max_batch, norm, step, 1, 4, false)
			xi := make([]float64, cont.N)
			for i := range xi {
				xi[i] = 0.0
			}
			var offset int
			if (in_wid-ker_wid/2)%2 != 0 {
				xi[0] = 1.0
			} else {
				// fmt.Println("offset nonzero!")
				offset = cont.N - (max_batch)*(in_wid+1)
				xi[offset] = -1.0
			}
			xi_plain := ckks.NewPlaintext(cont.params, ct_conv.Level(), 1.0)
			cont.encoder.EncodeCoeffs(xi, xi_plain)
			cont.encoder.ToNTT(xi_plain)
			ct_conv = cont.evaluator.MulNew(ct_conv, xi_plain)
			// res_tmp = cont.encoder.DecodeCoeffs(cont.decryptor.DecryptNew(ct_conv))
			// fmt.Println("After offset: ")
			// prt_mat_norm_step(res_tmp, max_batch, norm, step, 1, 4, false)
		}
	} else {
		if inside {
			new_ker_wid := ker_wid*in_step - in_step + 1
			new_ker_in := make([]float64, len(ker_in)*new_ker_wid*new_ker_wid/(ker_wid*ker_wid))

			for i := 0; i < ker_wid; i++ {
				for j := 0; j < ker_wid; j++ {
					for ib := 0; ib < real_ib; ib++ {
						for ob := 0; ob < real_ob; ob++ {
							new_ker_in[in_step*i*new_ker_wid*real_ib*real_ob+(in_step*j)*real_ib*real_ob+ib*real_ob+ob] = ker_in[i*ker_wid*real_ib*real_ob+j*real_ib*real_ob+ib*real_ob+ob]
						}
					}
				}
			}
			ct_conv = evalConv_BN(cont, ct_input, new_ker_in, bn_a, bn_b, in_wid, new_ker_wid, real_ib, real_ob, norm, math.Exp2(math.Round(math.Log2(float64(cont.params.Q()[0]))-(pow+8))), trans)
		} else {
			ct_conv = evalConv_BN(cont, ct_input, ker_in, bn_a, bn_b, in_wid, ker_wid, real_ib, real_ob, norm, math.Exp2(math.Round(math.Log2(float64(cont.params.Q()[0]))-(pow+8))), trans)
		}
	}

	ct_conv.Scale = ct_conv.Scale * math.Pow(2, pow)

	// Only for checking the correctness (for CtoS)
	var slot1, slot2 []complex128
	var cfs_preB []float64
	if debug {
		cfs_preB = cont.encoder.DecodeCoeffs(cont.decryptor.DecryptNew(ct_conv))
	}
	fmt.Println("Bootstrapping... Ours (until CtoS):")
	start := time.Now()
	ct_boots := make([]*ckks.Ciphertext, 2)
	switch log_sparse {
	case 0:
		ct_boots[0], ct_boots[1], _ = cont.btp.BootstrappConv_CtoS(ct_conv)
	case 1:
		ct_boots[0], ct_boots[1], _ = cont.btp2.BootstrappConv_CtoS(ct_conv)
	case 2:
		ct_boots[0], ct_boots[1], _ = cont.btp3.BootstrappConv_CtoS(ct_conv)
	case 3:
		ct_boots[0], ct_boots[1], _ = cont.btp4.BootstrappConv_CtoS(ct_conv)
	case 4:
		ct_boots[0], ct_boots[1], _ = cont.btp5.BootstrappConv_CtoS(ct_conv)
	default:
		panic("No cases for log_sparse")
	}

	fmt.Printf("Done in %s \n", time.Since(start))
	// fmt.Println("after Boot (CtoS): LV = ", ct_boots[0].Level(), " Scale = ", math.Log2(ct_boots[0].Scale))

	if debug {
		slot1, slot2 = debugCtoS(cont, cfs_preB, log_sparse)
		slot1 = printDebug(log_sparse, cont.params, ct_boots[0], slot1, cont.decryptor, cont.encoder) // Compare before & after CtoS
		slot2 = printDebug(log_sparse, cont.params, ct_boots[1], slot2, cont.decryptor, cont.encoder) // Compare before & after CtoS
	}

	start = time.Now()
	for ul := 0; ul < iter; ul++ { // up & low parts
		if ct_boots[ul] != nil {
			ct_boots[ul] = evalReLU(cont.params, cont.evaluator, ct_boots[ul], alpha)
			cont.evaluator.MulByPow2(ct_boots[ul], int(pow), ct_boots[ul])
		}
	}
	fmt.Printf("ReLU Done in %s \n", time.Since(start))
	start = time.Now()

	// Only for checking the correctness (for ReLU)
	var cfs_postB []float64
	if debug {
		fmt.Println("after Relu: ", math.Log2(ct_boots[0].Scale), "lv: ", ct_boots[0].Level())
		relu1, relu2 := debugReLU(cont, slot1, slot2, alpha, pow)
		relu1 = printDebug(log_sparse, cont.params, ct_boots[0], relu1, cont.decryptor, cont.encoder)
		relu2 = printDebug(log_sparse, cont.params, ct_boots[1], relu2, cont.decryptor, cont.encoder)
		cfs_postB = debugStoC(cont, relu1, relu2, in_wid, kp_wid, pack_pos, step, log_sparse, kind, fast_pack)
	}

	ct_keep := make([]*ckks.Ciphertext, iter) // for extend (rotation) of ctxt_in
	for ul := 0; ul < iter; ul++ {
		if trans {
			ct_keep[ul] = ext_ctxt(cont.evaluator, cont.encoder, ct_boots[ul], cont.r_idx[in_wid][ul], cont.params)
		} else if stride {
			if sparse { // we will use ext_double to reduce rotations; hence similar to fast_pack case
				if ct_boots[ul] != nil {
					if ul == 0 {
						ct_keep[ul] = ext_double_ctxt(cont.evaluator, cont.encoder, ct_boots[ul], cont.m_idx[in_wid][pack_pos], cont.r_idx[in_wid][pack_pos], cont.params)
					} else {
						ct_keep[ul] = ext_double_ctxt(cont.evaluator, cont.encoder, ct_boots[ul], cont.m_idx_l[in_wid][pack_pos], cont.r_idx_l[in_wid][pack_pos], cont.params)
					}
				} else {
					ct_keep[ul] = nil
				}
			} else {
				if fast_pack {
					if ul == 0 {
						ct_keep[ul] = ext_double_ctxt(cont.evaluator, cont.encoder, ct_boots[ul], cont.m_idx[in_wid][pack_pos], cont.r_idx[in_wid][pack_pos], cont.params)
					} else {
						ct_keep[ul] = ext_double_ctxt(cont.evaluator, cont.encoder, ct_boots[ul], cont.m_idx_l[in_wid][pack_pos], cont.r_idx_l[in_wid][pack_pos], cont.params)
					}
				} else {
					if ul == 0 {
						ct_keep[ul] = ext_ctxt(cont.evaluator, cont.encoder, ct_boots[ul], cont.r_idx[in_wid][pack_pos], cont.params)
					} else {
						ct_keep[ul] = ext_ctxt(cont.evaluator, cont.encoder, ct_boots[ul], cont.r_idx_l[in_wid][pack_pos], cont.params)
					}
				}
			}
		} else if inside {
			if ct_boots[ul] != nil {
				if sparse {
					ct_keep[ul] = keep_ctxt(cont.params, cont.evaluator, cont.encoder, ct_boots[ul], cont.ext_idx[in_wid][ul])
				} else {
					ct_keep[ul] = keep_ctxt(cont.params, cont.evaluator, cont.encoder, ct_boots[ul], cont.ext_idx[step][ul])
				}
			} else {
				ct_keep[ul] = nil
			}
		} else {
			if ct_boots[ul] != nil {
				ct_keep[ul] = keep_ctxt(cont.params, cont.evaluator, cont.encoder, ct_boots[ul], cont.ext_idx[in_wid][ul])
			} else {
				ct_keep[ul] = nil
			}
		}
	}

	if iter == 1 {
		ct_boots[1] = nil
		ct_res = cont.btp.BootstrappConv_StoC(ct_keep[0], ct_boots[1])
		if log_sparse != 0 {
			panic("we didn't implement this case")
		}
	} else {
		switch log_sparse {
		case 0:
			ct_res = cont.btp.BootstrappConv_StoC(ct_keep[0], ct_keep[1])
		case 1:
			ct_res = cont.btp2.BootstrappConv_StoC(ct_keep[0], ct_keep[1])
		case 2:
			ct_res = cont.btp3.BootstrappConv_StoC(ct_keep[0], ct_keep[1])
		case 3:
			ct_res = cont.btp4.BootstrappConv_StoC(ct_keep[0], ct_keep[1])
		case 4:
			ct_res = cont.btp5.BootstrappConv_StoC(ct_keep[0], ct_keep[1])
		default:
			panic("No cases for log_sparse")
		}
	}

	cont.evaluator.Rescale(ct_res, cont.params.Scale(), ct_res)
	fmt.Printf("Boot (StoC) Done in %s \n", time.Since(start))

	// Only for checking the correctness (for StoC)
	if debug {
		fmt.Println("Boot out: ")
		switch log_sparse {
		case 0:
			printDebugCfs(cont.params, ct_res, cfs_postB, cont.decryptor, cont.encoder)
		case 1:
			printDebugCfs(cont.params2, ct_res, cfs_postB, cont.decryptor, cont.encoder)
		case 2:
			printDebugCfs(cont.params3, ct_res, cfs_postB, cont.decryptor, cont.encoder)
		case 3:
			printDebugCfs(cont.params4, ct_res, cfs_postB, cont.decryptor, cont.encoder)
		case 4:
			printDebugCfs(cont.params5, ct_res, cfs_postB, cont.decryptor, cont.encoder)
		default:
			panic("No cases for log_sparse")
		}
		max_batch := cont.N / (in_wid * in_wid)
		res_tmp := cont.encoder.DecodeCoeffs(cont.decryptor.DecryptNew(ct_res))
		if inside {
			start := 1
			if ker_wid == 5 {
				start = step
			}
			prt_mat_norm_step(res_tmp, max_batch, norm, step, start, 3, false)
		} else {
			if stride {
				max_batch = 4 * cont.N / (in_wid * in_wid)
				start := 1
				if ker_wid == 5 {
					start = step
				}
				prt_mat_norm_step(res_tmp, max_batch, norm, step, start, 3, false)
			} else {
				prt_mat_norm(res_tmp, max_batch, norm, 3, false)
			}
		}
	}

	return ct_res
}

// log_spars = 0 -> full slot, 1 -> full/2 , ...
func debugCtoS(cont *context, cfs_preB []float64, log_sparse int) (slot1, slot2 []complex128) {
	preB_cfs1 := make([]float64, cont.params.Slots())
	preB_cfs2 := make([]float64, cont.params.Slots())
	slot1 = make([]complex128, cont.params.Slots()/(1<<log_sparse)) // first part of ceffs
	slot2 = make([]complex128, cont.params.Slots()/(1<<log_sparse)) // second part of ceffs
	for i := range preB_cfs1 {
		preB_cfs1[i] = cfs_preB[reverseBits(uint32(i), cont.params.LogSlots())] // first part of coeffs
		preB_cfs2[i] = cfs_preB[reverseBits(uint32(i), cont.params.LogSlots())+uint32(cont.params.Slots())]
		if i < len(slot1) {
			slot1[i] = complex(preB_cfs1[i], 0)
			slot2[i] = complex(preB_cfs2[i], 0)
		}
		// slot1[i] = complex(preB_cfs1[i]/math.Pow(2, float64(pow)), 0)
		// slot2[i] = complex(preB_cfs2[i]/math.Pow(2, float64(pow)), 0)
	}
	if log_sparse != 0 { // not full slot, we pack two ciphertexts slots into one ciphertext
		slot1 = append(slot1, slot2...)
		slot2 = nil
	}

	return
}

func debugReLU(cont *context, slot1, slot2 []complex128, alpha, pow float64) (relu1, relu2 []complex128) {
	relu1 = make([]complex128, len(slot1))
	if slot2 != nil {
		relu2 = make([]complex128, len(slot1))
	} else {
		relu2 = nil
	}

	for i := range relu1 {
		relu1[i] = complex((math.Max(0, real(slot1[i]))+math.Min(0, real(slot1[i])*alpha))*math.Pow(2, float64(pow)), 0)
	}
	for i := range relu2 {
		relu2[i] = complex((math.Max(0, real(slot2[i]))+math.Min(0, real(slot2[i])*alpha))*math.Pow(2, float64(pow)), 0)
	}

	return
}

func debugStoC(cont *context, slot1, slot2 []complex128, in_wid, kp_wid, pos, step, log_sparse int, kind string, fast_pack bool) (cfs_postB []float64) {
	slot1_fl := make([]float64, cont.params.Slots())
	slot2_fl := make([]float64, cont.params.Slots())
	if slot2 == nil {
		slot2_fl = nil
	}
	for i := range slot1 {
		slot1_fl[i] = real(slot1[i])
	}
	for i := range slot2 {
		slot2_fl[i] = real(slot2[i])
	}

	raw_in_wid_odd := true
	if kp_wid%2 == 0 { // valid only for resnet case (precise result: raw_in_wid is odd or not)
		raw_in_wid_odd = false
	}

	var tmp1, tmp2 []float64
	switch kind {
	case "Conv":
		tmp1 = keep_vec(slot1_fl, in_wid, kp_wid, 0)
		tmp2 = keep_vec(slot2_fl, in_wid, kp_wid, 1)
	case "StrConv_inside", "Conv_inside", "Conv_sparse":
		if slot2_fl != nil {
			tmp1 = keep_vec_stride(slot1_fl, in_wid, kp_wid, step, 0, raw_in_wid_odd)
			tmp2 = keep_vec_stride(slot2_fl, in_wid, kp_wid, step, 1, raw_in_wid_odd)
		} else {
			tmp := keep_vec_sparse(slot1_fl, in_wid, kp_wid, log_sparse)
			tmp1 = make([]float64, cont.params.Slots())
			tmp2 = make([]float64, cont.params.Slots())
			for i := 0; i < len(slot1)/2; i++ {
				tmp1[i] = tmp[i]
				tmp2[i] = tmp[i+len(slot1)/2]
			}
		}
	case "StrConv_sparse", "StrConv_sparse_full":
		if slot2_fl != nil {
			tmp1 = comprs_vec_sparse(slot1_fl, in_wid, kp_wid, log_sparse, 0, pos)
			tmp2 = comprs_vec_sparse(slot2_fl, in_wid, kp_wid, log_sparse, 0, pos)
		} else {
			tmp := comprs_vec_sparse(slot1_fl, in_wid, kp_wid, log_sparse, 0, pos)
			tmp1 = make([]float64, cont.params.Slots())
			tmp2 = make([]float64, cont.params.Slots())
			for i := 0; i < len(slot1)/2; i++ {
				tmp1[i] = tmp[i]
				tmp2[i] = tmp[i+len(slot1)/2]
			}
		}
	case "StrConv", "StrConv_fast", "StrConv_odd":
		if fast_pack {
			tmp1 = comprs_full_fast(slot1_fl, in_wid, kp_wid, pos, 0)
			tmp2 = comprs_full_fast(slot2_fl, in_wid, kp_wid, pos, 1)
		} else {
			tmp1 = comprs_full(slot1_fl, in_wid, kp_wid, pos, 0)
			tmp2 = comprs_full(slot2_fl, in_wid, kp_wid, pos, 1)
		}
	default:
		panic("No kind!")
	}

	cfs_postB1 := make([]float64, cont.params.N()/2)
	cfs_postB2 := make([]float64, cont.params.N()/2)
	for i := range cfs_postB1 {
		cfs_postB1[i] = tmp1[reverseBits(uint32(i), cont.params.LogSlots())]
	}
	for i := range cfs_postB2 {
		cfs_postB2[i] = tmp2[reverseBits(uint32(i), cont.params.LogSlots())]
	}
	cfs_postB = append(cfs_postB1, cfs_postB2...) // After rot(ext) and boot
	return
}
