package main

import (
	"fmt"
	"math"
	"time"

	"github.com/dwkim606/test_lattigo/ckks"
	"github.com/dwkim606/test_lattigo/rlwe"
)

// output bitreversed input with bitwid
func reverseBits(num uint32, bitwid int) uint32 {
	num = num << (32 - bitwid)

	var ret = uint32(0)
	var power = uint32(31)
	for num != 0 {
		ret += (num & 1) << power
		num = num >> 1
		power -= 1
	}
	return ret
}

// output the slice with bitreversed order from input
func reverseOrder(input []float64, bitwid int) []float64 {
	out := make([]float64, len(input))
	for i := range out {
		out[i] = input[reverseBits(uint32(i), bitwid)]
	}

	return out
}

// Extract upper left elements of size prt_wid * prt_wid from the input arrg vec with batch batches
// prt_wid, batch all are those from the output of our conv algorithm (consider padding)
func reshape_conv_out(result []float64, prt_wid, out_num int) []float64 {
	prt_out := make([]float64, prt_wid*prt_wid*out_num)
	in_wid := 2 * prt_wid
	batch := len(result) / (in_wid * in_wid)

	for i := 0; i < out_num; i++ {
		for j := 0; j < prt_wid; j++ {
			for k := 0; k < prt_wid; k++ {
				prt_out[i+out_num*(j*prt_wid+k)] = result[i+batch*(j*in_wid+k)] //[batch*(in_wid+1)*(ker_wid-1)+i+batch*(j*in_wid+k)]
			}
		}
	}

	return prt_out
}

// Reshape 1-D input (from python) into H,W,Batch format
// i.e., 0, 1, 2, 3, -> 1st input of 0,1,2,3-th batch
// only for BL test
func reshape_input_BL(input []float64, in_wid int) (out []complex128) {
	out = make([]complex128, len(input))
	batch := len(input) / (in_wid * in_wid)
	l := 0

	for i := 0; i < in_wid; i++ {
		for j := 0; j < in_wid; j++ {
			for k := 0; k < batch; k++ {
				out[i*in_wid+j+k*in_wid*in_wid] = complex(input[l], 0)
				l++
			}
		}
	}

	return out
}

// Reshape 1-D kernel (from python) into (H,W,in,out) format, and applies BN, then into max ker
// ker[i][j][ib][ob]: (i,j)-th elt of kernel for ib-th input to ob-th output
// only for BL test // for transposed conv, we should add rearragement!!
// norm == 1 : normal case, norm == 4 : in & out batch is (1,0,0,0,2,0,0,0,3,0,0,0,4,0,0,0)
func reshape_ker_BL(input, BN_a []float64, ker_wid, inB, outB, max_bat, pos, norm int, trans bool) (max_ker_rs [][][][]float64) {
	ker_rs := make([][][][]float64, ker_wid)
	for i := 0; i < ker_wid; i++ {
		ker_rs[i] = make([][][]float64, ker_wid)
		for j := 0; j < ker_wid; j++ {
			ker_rs[i][j] = make([][]float64, inB)
			for ib := 0; ib < inB; ib++ {
				ker_rs[i][j][ib] = make([]float64, outB)
				for ob := 0; ob < outB; ob++ {
					if trans {
						if ib < (inB / 4) {
							ker_rs[i][j][ib][ob] = input[(4*ib+pos)+ob*inB+(ker_wid-j-1)*outB*inB+(ker_wid-i-1)*outB*inB*ker_wid] * BN_a[ob] // Apply BN
						}
					} else {
						ker_rs[i][j][ib][ob] = input[ob+ib*outB+j*outB*inB+i*outB*inB*ker_wid] * BN_a[ob] // Apply BN
					}
				}
			}
		}
	}
	// overload to max batch case
	max_ker_rs = make([][][][]float64, ker_wid)
	for i := 0; i < ker_wid; i++ {
		max_ker_rs[i] = make([][][]float64, ker_wid)
		for j := 0; j < ker_wid; j++ {
			max_ker_rs[i][j] = make([][]float64, max_bat)
			for ib := 0; ib < max_bat; ib++ {
				max_ker_rs[i][j][ib] = make([]float64, max_bat)
			}
			for ib := 0; ib < inB; ib++ {
				for ob := 0; ob < outB; ob++ {
					max_ker_rs[i][j][norm*ib][norm*ob] = ker_rs[i][j][ib][ob]
				}
			}
		}
	}

	return max_ker_rs
}

// rotate input ciphertext and outputs rotated ciphertexts
// always assume Odd ker_wid
func preConv_BL(evaluator ckks.Evaluator, ct_in *ckks.Ciphertext, in_wid, ker_wid int) (ct_in_rots []*ckks.Ciphertext) {
	ker_size := ker_wid * ker_wid
	ct_in_rots = make([]*ckks.Ciphertext, ker_size)

	st := -(ker_wid / 2)
	end := ker_wid / 2

	var rotations []int
	for i := st; i <= end; i++ {
		for j := st; j <= end; j++ {
			rotations = append(rotations, i*in_wid+j)
		}
	}
	ct_rots_test := evaluator.RotateHoisted(ct_in, rotations)
	k := 0
	for i := st; i <= end; i++ {
		for j := st; j <= end; j++ {
			ct_in_rots[k] = ct_rots_test[i*in_wid+j]
			k++
		}
	}

	return ct_in_rots
}

// eval Convolution for the part of output: need to sum this up with rotations
func postConv_BL(param ckks.Parameters, encoder ckks.Encoder, evaluator ckks.Evaluator, ct_in_rots []*ckks.Ciphertext, in_wid, ker_wid, rot, pad int, max_ker_rs [][][][]float64) (ct_out *ckks.Ciphertext) {

	max_batch := param.Slots() / (in_wid * in_wid)
	postKer := make([]complex128, param.Slots())
	pl_tmp := ckks.NewPlaintext(param, ct_in_rots[0].Level(), param.Scale())

	iter := 0
	for i := 0; i < ker_wid; i++ {
		for j := 0; j < ker_wid; j++ {
			for k := 0; k < max_batch; k++ {
				for ki := 0; ki < (in_wid - pad); ki++ { // position of input
					for kj := 0; kj < (in_wid - pad); kj++ {
						postKer[k*in_wid*in_wid+ki*in_wid+kj] = complex(max_ker_rs[i][j][k][(k-rot+max_batch)%max_batch], 0)
						if (((ki + i - (ker_wid / 2)) < 0) || ((ki + i - (ker_wid / 2)) >= (in_wid - pad))) || (((kj + j - (ker_wid / 2)) < 0) || ((kj + j - (ker_wid / 2)) >= (in_wid - pad))) {
							postKer[k*in_wid*in_wid+ki*in_wid+kj] = complex(0, 0)
						}
					}
				}
			}
			encoder.Encode(pl_tmp, postKer, param.LogSlots())
			encoder.ToNTT(pl_tmp)
			if (i == 0) && (j == 0) {
				ct_out = evaluator.MulNew(ct_in_rots[iter], pl_tmp)
			} else {
				ct_tmp := evaluator.MulNew(ct_in_rots[iter], pl_tmp)
				evaluator.Add(ct_out, ct_tmp, ct_out)
			}
			iter++
		}
	}

	return ct_out
}

// Reshape 1-D ker_in (from python) into batch number of ker_outs: ker_out[i][j] = j-th kernel (elements then batch order) for i-th output
// i.e., ker_out is of the shape (out_batch, (in_batch * ker_size))
// ker_out[i] = [k1 for 1st input, ..., ,kk for 1st input, k1 for 2nd input, ...]
// trans = true for transposed convolution (in trans convolution of python,  we should rearrage ker_out carefully)
func reshape_ker(ker_in []float64, k_sz, out_batch int, trans bool) (ker_out [][]float64) {
	ker_out = make([][]float64, out_batch)
	in_batch := len(ker_in) / (k_sz * out_batch)

	for i := 0; i < out_batch; i++ {
		ker_out[i] = make([]float64, k_sz*in_batch)
		for j := 0; j < in_batch; j++ {
			for k := 0; k < k_sz; k++ {
				if trans {
					ker_out[i][j*k_sz+(k_sz-k-1)] = ker_in[j+i*in_batch+k*out_batch*in_batch]
				} else {
					ker_out[i][j*k_sz+k] = ker_in[i+j*out_batch+k*out_batch*in_batch]
					// ker_out[i][j*k_sz+k] = ker_in[j+i*in_batch+k*out_batch*in_batch]
				}
			}
		}
	}
	return
}

// Encode ker_outs from reshape_ker into the i-th ker vector output
// in_wid, in_batch is those for input (to be convolved) includng padding
func encode_ker_final(ker_in [][]float64, pos, i, in_wid, in_batch, ker_wid int) []float64 {
	vec_size := in_wid * in_wid * in_batch
	output := make([]float64, vec_size)
	bias := pos * ker_wid * ker_wid * in_batch
	k_sz := ker_wid * ker_wid

	// allocate each kernel so that 0-th batch and B-1th batch adds together at B-1th position (B = in_batch)
	for j := 0; j < in_batch; j++ { // j-th input (batch)
		for k := 0; k < k_sz; k++ {
			// fmt.Println("ecd: ", j, k)
			output[(in_wid*(k/ker_wid)+k%ker_wid)*in_batch+j] = ker_in[i][(in_batch-1-j)*k_sz+(k_sz-1-k)+bias] // * scale;
		}
	}

	// move the kernel to left adj times, so that the result of "transposed" convolution appears at 0-th position
	// adj := (in_wid+1)*(ker_wid-3)/2 + (in_batch - 1)

	adj := (in_batch - 1) + (in_batch)*(in_wid+1)*(ker_wid-1)/2
	tmp := make([]float64, adj)
	for i := 0; i < adj; i++ {
		tmp[i] = output[vec_size-adj+i]
		output[vec_size-adj+i] = -output[i]
	}
	for i := 0; i < vec_size-2*adj; i++ {
		output[i] = output[i+adj]
	}
	for i := 0; i < adj; i++ {
		output[i+vec_size-2*adj] = tmp[i]
	}

	return output
}

// Generate the logN # of plaintexts idx[i] = X^(2^i) and GaloisKeys for each
// Required for Packing
func gen_idxNlogs(ECD_LV int, keygen rlwe.KeyGenerator, sk *rlwe.SecretKey, encoder ckks.Encoder, params ckks.Parameters) (idx []*ckks.Plaintext, pack_eval ckks.Evaluator) {
	logN := params.LogN()
	N := params.N()
	gals := []uint64{}
	idx = make([]*ckks.Plaintext, logN)
	coeffs := make([]float64, N)

	for i := 0; i < logN; i++ {
		coeffs[1<<i] = 1.0
		idx[i] = ckks.NewPlaintext(params, ECD_LV, 1.0)
		encoder.EncodeCoeffs(coeffs, idx[i])
		encoder.ToNTT(idx[i])
		coeffs[1<<i] = 0.0

		gals = append(gals, (1<<(i+1) + 1)) // Add galois autos required
	}

	pack_eval = ckks.NewEvaluator(params, rlwe.EvaluationKey{Rtks: keygen.GenRotationKeys(gals, sk)})

	return idx, pack_eval
}

// Pack cnum (alwyas Po2) number of ctxts(in_a[i]) into one
// Each in_a[i] must be arrvec (i.e., sparse with steps) with full size
// Also get the plaintexts idx[i] = X^i from gen_idx
func pack_ctxts(pack_eval ckks.Evaluator, ctxts_in []*ckks.Ciphertext, max_cnum, real_cnum int, idx []*ckks.Plaintext, params ckks.Parameters) (ctxt *ckks.Ciphertext) {
	step := max_cnum / 2
	// step := real_cnum / 2
	norm := max_cnum / real_cnum
	ctxts := make([]*ckks.Ciphertext, max_cnum)
	for i := 0; i < max_cnum; i++ {
		if i%norm == 0 {
			ctxts[i] = ctxts_in[i].CopyNew()
			ctxts[i].SetScalingFactor(ctxts[i].Scale * float64(real_cnum))
		}
	}

	var tmp1, tmp2 *ckks.Ciphertext

	logStep := 0
	for i := step; i > 1; i /= 2 {
		logStep++
	}
	j := params.LogN() - logStep

	for step >= norm {
		for i := 0; i < step; i += norm {
			tmp1 = pack_eval.MulNew(ctxts[i+step], idx[logStep])
			tmp2 = pack_eval.SubNew(ctxts[i], tmp1)
			pack_eval.Add(ctxts[i], tmp1, tmp1)
			pack_eval.RotateGal(tmp2, (1<<j)+1, tmp2)
			pack_eval.Add(tmp1, tmp2, ctxts[i])
		}
		step /= 2
		logStep--
		j++
	}

	return ctxts[0]
}

// rotate and add input with m_idx then r_idx so that it outputs a result ctxt.
func bsgs_ctxt(eval ckks.Evaluator, encoder ckks.Encoder, input *ckks.Ciphertext, m_idx, r_idx map[int][]int, params ckks.Parameters) (result *ckks.Ciphertext) {
	st := true
	var mid_result *ckks.Ciphertext
	for rot, elt := range m_idx { // rot should be in increasing order
		tmp := make([]complex128, params.Slots())
		for i := range elt {
			tmp[i] = complex(float64(elt[i]), 0)
		}
		plain_tmp := ckks.NewPlaintext(params, input.Level(), 1048576.0) // set scale of rot idx to be 2^(scale/2)
		encoder.EncodeNTT(plain_tmp, tmp, params.LogSlots())

		if st {
			mid_result = eval.RotateNew(eval.MulNew(input, plain_tmp), rot)
			st = false
		} else {
			ctxt_tmp := eval.RotateNew(eval.MulNew(input, plain_tmp), rot)
			eval.Add(mid_result, ctxt_tmp, mid_result)
		}
	}

	st = true
	for rot, elt := range r_idx {
		tmp := make([]complex128, params.Slots())
		for i := range elt {
			tmp[i] = complex(float64(elt[i]), 0)
		}
		plain_tmp := ckks.NewPlaintext(params, input.Level(), 1048576.0) // set scale of rot idx to be 2^(scale/2)
		encoder.EncodeNTT(plain_tmp, tmp, params.LogSlots())

		if st {
			result = eval.RotateNew(eval.MulNew(mid_result, plain_tmp), rot)
			st = false
		} else {
			ctxt_tmp := eval.RotateNew(eval.MulNew(mid_result, plain_tmp), rot)
			eval.Add(result, ctxt_tmp, result)
		}
	}

	// eval.Rescale(result, params.Scale(), result)

	return result
}

// extend ctxt using given rotations so that it outputs a ctxt to be convolved with filter
func ext_ctxt(eval ckks.Evaluator, encoder ckks.Encoder, input *ckks.Ciphertext, r_idx map[int][]int, params ckks.Parameters) (result *ckks.Ciphertext) {
	st := true

	for rot, elt := range r_idx {
		tmp := make([]complex128, params.Slots())
		for i := range elt {
			tmp[i] = complex(float64(elt[i]), 0)
		}
		// plain_tmp := encoder.EncodeNTTAtLvlNew(input.Level(), tmp, params.LogSlots())
		plain_tmp := ckks.NewPlaintext(params, input.Level(), float64(params.Q()[input.Level()]))
		encoder.EncodeNTT(plain_tmp, tmp, params.LogSlots())

		if st {
			result = eval.RotateNew(eval.MulNew(input, plain_tmp), rot)
			st = false
		} else {
			ctxt_tmp := eval.RotateNew(eval.MulNew(input, plain_tmp), rot)
			eval.Add(result, ctxt_tmp, result)
		}
	}

	eval.Rescale(result, params.Scale(), result)

	return result
}

// extend ctxt using given rotations so that it outputs a ctxt to be convolved with filter
func ext_double_ctxt(eval ckks.Evaluator, encoder ckks.Encoder, input *ckks.Ciphertext, m_idx, r_idx map[int][]int, params ckks.Parameters) (result *ckks.Ciphertext) {
	st := true
	sqr_sc := math.Sqrt(float64(params.Q()[input.Level()]))
	var mid_result *ckks.Ciphertext
	for rot, elt := range m_idx {
		tmp := make([]complex128, params.Slots())
		for i := range elt {
			tmp[i] = complex(float64(elt[i]), 0)
		}
		plain_tmp := ckks.NewPlaintext(params, input.Level(), sqr_sc)
		encoder.EncodeNTT(plain_tmp, tmp, params.LogSlots())
		if st {
			mid_result = eval.RotateNew(eval.MulNew(input, plain_tmp), rot)
			st = false
		} else {
			ctxt_tmp := eval.RotateNew(eval.MulNew(input, plain_tmp), rot)
			eval.Add(mid_result, ctxt_tmp, mid_result)
		}
	}

	st = true
	for rot, elt := range r_idx {
		tmp := make([]complex128, params.Slots())
		for i := range elt {
			tmp[i] = complex(float64(elt[i]), 0)
		}
		plain_tmp := ckks.NewPlaintext(params, input.Level(), sqr_sc)
		encoder.EncodeNTT(plain_tmp, tmp, params.LogSlots())
		if st {
			result = eval.RotateNew(eval.MulNew(mid_result, plain_tmp), rot)
			st = false
		} else {
			ctxt_tmp := eval.RotateNew(eval.MulNew(mid_result, plain_tmp), rot)
			eval.Add(result, ctxt_tmp, result)
		}
	}

	eval.Rescale(result, params.Scale(), result)

	return result
}

// keep ctxt using given idx so that it outputs a ctxt to be convolved with filter
func keep_ctxt(params ckks.Parameters, eval ckks.Evaluator, encoder ckks.Encoder, input *ckks.Ciphertext, idx []int) (result *ckks.Ciphertext) {
	tmp := make([]complex128, params.Slots())
	for i := range idx {
		tmp[i] = complex(float64(idx[i]), 0)
	}
	// plain_tmp := encoder.EncodeNTTAtLvlNew(input.Level(), tmp, params.LogSlots())
	plain_tmp := ckks.NewPlaintext(params, input.Level(), float64(params.Q()[input.Level()]))
	encoder.EncodeNTT(plain_tmp, tmp, params.LogSlots())

	result = eval.MulNew(input, plain_tmp)

	eval.Rescale(result, params.Scale(), result)

	return result
}

// Divide coeff gen later
// alpha: for leaky relu
func evalReLU(params ckks.Parameters, evaluator ckks.Evaluator, ctxt_in *ckks.Ciphertext, alpha float64) (ctxt_out *ckks.Ciphertext) {
	// time_coeffs := time.Now()
	// alpha 10 (from minimax)
	// prescale is multiplied to improve precision (previously done in StoC matmult)
	// coeffs_tmp := []complex128{0.0, 10.8541842577442 * prescale, 0.0, -62.2833925211098 * prescale * prescale * prescale,
	// 	0.0, 114.369227820443 * prescale * prescale * prescale * prescale * prescale, 0.0, -62.8023496973074 * prescale * prescale * prescale * prescale * prescale * prescale * prescale}
	coeffs_tmp := []complex128{0.0, 10.8541842577442, 0.0, -62.2833925211098, 0.0, 114.369227820443, 0.0, -62.8023496973074}
	coeffsReLU := ckks.NewPoly(coeffs_tmp)

	coeffs_tmp2 := []complex128{0.0, 4.13976170985111, 0.0, -5.84997640211679, 0.0, 2.94376255659280, 0.0, -0.454530437460152}
	coeffsReLU2 := ckks.NewPoly(coeffs_tmp2)

	// coeffs_tmp3 := []complex128{0.0, 3.29956739043733, 0.0, -7.84227260291355, 0.0, 12.8907764115564, 0.0, -12.4917112584486, 0.0, 6.94167991428074, 0.0, -2.04298067399942, 0.0, 0.246407138926031}
	// leakyRelu = x (bconst * s(x) + aconst)
	aconst := (alpha + 1) / 2.0
	bconst := (1 - alpha) / 2.0
	coeffs_tmp3 := []complex128{0.0, 3.29956739043733, 0.0, -7.84227260291355, 0.0, 12.8907764115564, 0.0, -12.4917112584486, 0.0, 6.94167991428074, 0.0, -2.04298067399942, 0.0, 0.246407138926031}
	for i := range coeffs_tmp3 {
		coeffs_tmp3[i] = coeffs_tmp3[i] * complex(bconst, 0.0)
	}
	coeffsReLU3 := ckks.NewPoly(coeffs_tmp3)

	// fmt.Println("Time for coeffs: ", time.Since(time_coeffs))

	fmt.Printf("Eval: ")
	ctxt_sign, err := evaluator.EvaluatePoly(ctxt_in, coeffsReLU, params.Scale())
	if err != nil {
		panic(err)
	}
	ctxt_sign, err = evaluator.EvaluatePoly(ctxt_sign, coeffsReLU2, params.Scale())
	if err != nil {
		panic(err)
	}
	ctxt_sign, err = evaluator.EvaluatePoly(ctxt_sign, coeffsReLU3, params.Scale())
	if err != nil {
		panic(err)
	}

	ctxt_out = evaluator.AddConstNew(ctxt_sign, aconst)

	evaluator.DropLevel(ctxt_in, ctxt_in.Level()-ctxt_out.Level())
	evaluator.Mul(ctxt_out, ctxt_in, ctxt_out)
	evaluator.Relinearize(ctxt_out, ctxt_out)

	return
}

// Encode Kernel and outputs Plain(ker) for normal conv or transposed conv (no stride)
// Prepare kernels as if the input, output is fully batched in 1 ctxt (for trans conv, output pl_ker is one (pos-th) of the 4 divided pl_ker)
// in_wid : width of input (include padding)
// BN_a: coefficient to be multiplied (for BN)
// real_ib, real_ob: real batches for kernel <=> set the same as python
func prep_Ker(params ckks.Parameters, encoder ckks.Encoder, ker_in, BN_a []float64, in_wid, ker_wid, real_ib, real_ob, norm, ECD_LV, pos int, trans bool) (pl_ker []*ckks.Plaintext) {
	max_bat := params.N() / (in_wid * in_wid)
	ker_size := ker_wid * ker_wid
	ker_rs := reshape_ker(ker_in, ker_size, real_ob, trans) // ker1[i][j] = j-th kernel for i-th output

	for i := 0; i < real_ob; i++ { // apply batch normalization
		for j := range ker_rs[i] {
			ker_rs[i][j] = ker_rs[i][j] * BN_a[i]
		}
	}

	max_ker_rs := make([][]float64, max_bat) // overloading ker_rs to the case with max_batch
	for i := 0; i < max_bat; i++ {
		max_ker_rs[i] = make([]float64, max_bat*ker_size)
	}
	for i := 0; i < real_ob; i++ {
		for j := 0; j < real_ib; j++ {
			for k := 0; k < ker_size; k++ {
				max_ker_rs[norm*i][norm*j*ker_size+k] = ker_rs[i][j*ker_size+k]
			}
		}
	}

	pl_ker = make([]*ckks.Plaintext, max_bat)
	for i := 0; i < max_bat; i++ {
		pl_ker[i] = ckks.NewPlaintext(params, ECD_LV, params.Scale())
		encoder.EncodeCoeffs(encode_ker_final(max_ker_rs, pos, i, in_wid, max_bat, ker_wid), pl_ker[i])
		encoder.ToNTT(pl_ker[i])
	}

	return pl_ker
}

// Eval Conv, then Pack
// The ciphertexts must be packed into full (without vacant position)
func conv_then_pack(params ckks.Parameters, pack_evaluator ckks.Evaluator, ctxt_in *ckks.Ciphertext, pl_ker []*ckks.Plaintext, plain_idx []*ckks.Plaintext, max_ob, norm, ECD_LV int, out_scale float64) *ckks.Ciphertext {
	start := time.Now()
	ctxt_out := make([]*ckks.Ciphertext, max_ob)
	for i := 0; i < max_ob; i++ {
		if i%norm == 0 {
			ctxt_out[i] = pack_evaluator.MulNew(ctxt_in, pl_ker[i])
			pack_evaluator.SetScale(ctxt_out[i], out_scale/(float64(max_ob/norm)))
		}
		// pack_evaluator.Rescale(ctxt_out[i], float64(1<<10), ctxt_out[i])
	}
	mt := time.Since(start)
	fmt.Println("\t mult time: ", mt)
	ctxt_result := pack_ctxts(pack_evaluator, ctxt_out, max_ob, max_ob/norm, plain_idx, params)
	fmt.Println("\t Pack time: ", time.Since(start)-mt)

	// fmt.Println("Result Scale: ", math.Log2(ctxt_result.Scale))
	// fmt.Println("Result LV: ", ctxt_result.Level())
	// fmt.Printf("Done in %s \n", time.Since(start))

	if (out_scale != ctxt_result.Scale) || (0 != ctxt_result.Level()) {
		panic("LV or scale after conv then pack, inconsistent")
	}

	return ctxt_result
}
