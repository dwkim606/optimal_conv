package main

import (
	"fmt"
	"math"
)

func toFixed(num float64, precision int) float64 {
	output := math.Pow(10, float64(precision))
	return float64(math.Round(num*output)) / output
}

// distribute input to the output starting from pos position
func arrgvec(input []int, output []int, pos int) {
	batch := len(output) / len(input)
	for i, elt := range input {
		output[pos+i*batch] = elt
	}
}

func print_vec(title string, input []float64, in_wid int, pos int) {
	row := make([]float64, in_wid)
	step := len(input) / (in_wid * in_wid)
	fmt.Println(title, ": ")
	for j := 0; j < in_wid; j++ {
		for i := range row {
			row[i] = toFixed(input[(j*in_wid+i)*step+pos], 3)
		}
		fmt.Println(row)
	}
	fmt.Println()

}

func lRot(a []float64, rotation int) []float64 {
	size := len(a)
	var newArray []float64
	for i := 0; i < rotation; i++ {
		newArray = a[1:size]
		newArray = append(newArray, a[0])
		a = newArray
	}
	return a
}

func rRot(a []float64, rotation int) []float64 {
	return lRot(a, len(a)-rotation)
}

func addSlice(a []float64, b []float64) []float64 {
	c := make([]float64, len(a))
	for i := range a {
		c[i] = a[i] + b[i]
	}
	return c
}

// (bit-reversed) input vector := (upper or lower part) of the total vector having in_wid * in_wid size elts
// Keep only the kp_wid*kp_wid values
// e.g., 1* // ** -> 10 // 00 (before bitreversed, pad = 1)
// ul: up or low
// assume N/2 sized input
func keep_vec(input []float64, in_wid, kp_wid, ul int) []float64 {
	output := make([]float64, len(input))

	tmp := gen_keep_vec(len(input), in_wid, kp_wid, ul)

	for i := range output {
		output[i] = input[i] * float64(tmp[i])
	}

	return output
}

func keep_vec_stride(input []float64, in_wid, kp_wid, step, ul int, raw_in_wid_odd bool) []float64 {
	output := make([]float64, len(input))

	tmp := gen_keep_vec_stride(len(input), in_wid, kp_wid, step, ul, raw_in_wid_odd)

	for i := range output {
		output[i] = input[i] * float64(tmp[i])
	}

	return output
}

func keep_vec_sparse(input []float64, in_wid, kp_wid, log_sparse int) []float64 {
	output := make([]float64, len(input))

	tmp := gen_keep_vec_sparse(len(input), in_wid, kp_wid, log_sparse)

	for i := range output {
		output[i] = input[i] * float64(tmp[i])
	}

	return output
}

func comprs_vec_sparse(input []float64, in_wid, kp_wid, log_sparse, ul, pos int) []float64 {

	tmp1, tmp2 := gen_comprs_sparse(len(input), in_wid, kp_wid, log_sparse, ul, pos)

	mid_output := make([]float64, len(input))
	for i := range tmp1 {
		rot_input := make([]float64, len(input))
		for j := range rot_input {
			rot_input[j] = input[j] * float64(tmp1[i][j])
		}
		rot_input = lRot(rot_input, i)
		if i < 0 {
			rot_input = rRot(rot_input, -i)
		}

		for j := range mid_output {
			mid_output[j] = rot_input[j] + mid_output[j]
		}
	}

	output := make([]float64, len(input))
	for i := range tmp2 {
		rot_input := make([]float64, len(input))
		for j := range rot_input {
			rot_input[j] = mid_output[j] * float64(tmp2[i][j])
		}
		rot_input = lRot(rot_input, i)
		if i < 0 {
			rot_input = rRot(rot_input, -i)
		}

		for j := range output {
			output[j] = rot_input[j] + output[j]
		}
	}

	return output
}

// returns the idx for keep_vec
// N: length of input (upper + lower)
// ul = 0 -> upper part, ul = 1 -> lower part
func gen_keep_vec(vec_size, in_wid, kp_wid, ul int) (idx []int) {
	logN := 0
	for ; (1 << logN) < (2 * vec_size); logN++ {
	}
	idx = make([]int, vec_size)
	batch := 2 * vec_size / (in_wid * in_wid)
	if kp_wid < in_wid/2 {
		panic("keep width too small. less than in_wid/2")
	}

	if ul == 0 {
		for i := 0; i < in_wid/2; i++ {
			for j := 0; j < kp_wid; j++ {
				for b := 0; b < batch; b++ {
					id := int(reverseBits(uint32(in_wid*batch*i+batch*j+b), logN-1))
					idx[id] = 1
				}
			}
		}
	} else if ul == 1 {
		for i := 0; i < kp_wid-in_wid/2; i++ {
			for j := 0; j < kp_wid; j++ {
				for b := 0; b < batch; b++ {
					id := int(reverseBits(uint32(in_wid*batch*i+batch*j+b), logN-1))
					idx[id] = 1
				}
			}
		}
	} else {
		panic("ul not 0 nor 1")
	}

	return idx
}

// returns the idx for keep_vec given that we have sparse pack with log_sparse (=0 if full, 1 if half sparse pack)
// Always assume log_sparse >= 1 so that both up and down part is in one ciphertext
// N: length of input (upper + lower), vec_size is always N/2!
func gen_keep_vec_sparse(vec_size, in_wid, kp_wid, log_sparse int) (idx []int) {
	logN := 0
	for ; (1 << logN) < (2 * vec_size); logN++ {
	}
	idx = make([]int, vec_size)
	batch := 2 * vec_size / (in_wid * in_wid)
	sparsity := 1 << log_sparse
	if sparsity == 1 {
		panic("We do not support full packing in gen_keep_vec_sparse")
	}
	if kp_wid < in_wid/2 {
		panic("keep width too small. less than in_wid/2")
	}

	for i := 0; i < in_wid/2; i++ {
		for j := 0; j < kp_wid; j++ {
			for b := 0; b < batch/sparsity; b++ {
				id := int(reverseBits(uint32(in_wid*batch*i+batch*j+b*sparsity), logN-1))
				idx[id] = 1
			}
		}
	}
	for i := 0; i < kp_wid-in_wid/2; i++ {
		for j := 0; j < kp_wid; j++ {
			for b := 0; b < batch/sparsity; b++ {
				id := int(reverseBits(uint32(in_wid*batch*i+batch*j+b*sparsity), logN-1)) + vec_size/sparsity
				idx[id] = 1
			}
		}
	}

	post_slot := 2 * len(idx) / sparsity
	for i := 0; i < post_slot; i++ {
		for j := 1; j < sparsity/2; j++ {
			idx[i+post_slot*j] = idx[i]
		}
	}

	return idx
}

// returns the idx for keep_vec // same as gen_keep_vec but keeps strided output only
// N: length of input (upper + lower)
// ul = 0 -> upper part, ul = 1 -> lower part
// kp_wid: number of elements to keep (in each row)
// step: distance between each element
// raw_in_wid_odd: raw_in_wid is odd or not
func gen_keep_vec_stride(vec_size, in_wid, kp_wid, step, ul int, raw_in_wid_odd bool) (idx []int) {
	logN := 0
	for ; (1 << logN) < (2 * vec_size); logN++ {
	}
	idx = make([]int, vec_size)
	batch := 2 * vec_size / (in_wid * in_wid)

	var init int
	if raw_in_wid_odd {
		init = 0
	} else {
		init = step - 1
	}

	if ul == 0 {
		for i := 0; i < kp_wid; i++ {
			if (init + i*step) < in_wid/2 {
				for j := 0; j < kp_wid; j++ {
					for b := 0; b < batch; b++ {
						id := int(reverseBits(uint32(in_wid*batch*(init+i*step)+batch*(j*step+init)+b), logN-1))
						idx[id] = 1
					}
				}
			}
		}
	} else if ul == 1 {
		for i := 0; i < kp_wid; i++ {
			if (init + i*step) >= in_wid/2 {
				for j := 0; j < kp_wid; j++ {
					for b := 0; b < batch; b++ {
						id := int(reverseBits(uint32(in_wid*batch*(init+i*step-in_wid/2)+batch*(j*step+init)+b), logN-1))
						idx[id] = 1
					}
				}
			}
		}
	} else {
		panic("ul not 0 nor 1")
	}

	return idx
}

// Assume N/2 input vector
// reverse of extend_full (after strided conv -> normal)
// in_wid = input wid including padding
// kp_wid = keep wid
// padding = true: only keeps valid elements	// (output) e.g., 12 00 // 34 00 // 00 00 // 00 00
// padding = false: keeps all elements	// (output) e.g., 12 // 34
// 0 <= pos < 4 determines to which part the output is positioned at the final output
// ul : up (0) or low (1) part
func comprs_full(input []float64, in_wid, kp_wid, pos, ul int) []float64 {
	output := make([]float64, len(input))
	batch := 2 * len(input) / (in_wid * in_wid)
	if kp_wid < in_wid/2 {
		panic("keep width too small. less than in_wid/2")
	}
	pos = int(reverseBits(uint32(pos), 2))
	padding := false
	min_wid := in_wid / 4
	if in_wid%4 != 0 {
		panic("input wid not divisible by 4")
	}
	if in_wid%2 != 0 {
		panic("input wid not divisible by 2")
	}
	log_in_wid := 0
	for ; (1 << log_in_wid) < in_wid; log_in_wid++ {
	}

	if padding {
		for j := 0; j < min_wid; j++ { // kinds of mov depends on j
			tmp := make([]float64, len(input))
			for b := 0; b < batch; b++ {
				for i := 0; i < min_wid; i++ {
					idx := 2*min_wid*in_wid*b + in_wid*j + i + min_wid*in_wid + min_wid
					tmp[idx] = input[idx]
				}
			}
			rot := -2*j*min_wid + 2*pos*min_wid*min_wid - min_wid*in_wid - min_wid
			output = addSlice(output, rRot(tmp, rot))
		}
		// // when we want to extract even positioned inputs
		// for j := 0; j < min_wid; j++ { // kinds of mov depends on j
		// 	tmp := make([]int, len(input))
		// 	for b := 0; b < batch; b++ {
		// 		for i := 0; i < min_wid; i++ {
		// 			idx := 2*min_wid*in_wid*b + in_wid*j + i
		// 			tmp[idx] = input[idx]
		// 		}
		// 	}
		// 	rot := -2*j*min_wid + 2*pos*min_wid*min_wid
		// 	output = addSlice(output, rRot(tmp, rot))
		// }
	} else {
		if ul == 0 {
			for j := 0; j < 2*min_wid; j++ { // kinds of mov depends on j
				tmp := make([]float64, len(input))
				for b := 0; b < batch; b++ {
					for i := 0; i < min_wid; i++ {
						if reverseBits(uint32(in_wid/2+j), log_in_wid) < uint32(kp_wid) {
							idx := 2*min_wid*in_wid*b + 2*min_wid*j + i + in_wid*min_wid + min_wid
							tmp[idx] = input[idx]
						}
					}
				}
				rot := -j*min_wid + 2*pos*min_wid*min_wid - min_wid - in_wid*min_wid
				output = addSlice(output, rRot(tmp, rot))
			}
		} else {
			for j := 0; j < 2*min_wid; j++ { // kinds of mov depends on j
				tmp := make([]float64, len(input))
				for b := 0; b < batch; b++ {
					for i := 0; i < min_wid; i++ {
						if (reverseBits(uint32(in_wid/2+j), log_in_wid) < uint32(kp_wid)) && (reverseBits(uint32(3*min_wid+i), log_in_wid-1) < uint32(kp_wid-in_wid/2)) {
							idx := 2*min_wid*in_wid*b + 2*min_wid*j + i + in_wid*min_wid + min_wid
							tmp[idx] = input[idx]
						}
					}
				}
				rot := -j*min_wid + 2*pos*min_wid*min_wid - min_wid - in_wid*min_wid
				output = addSlice(output, rRot(tmp, rot))
			}
		}
		// // when we want to extract even positioned inputs
		// for j := 0; j < 2*min_wid; j++ { // kinds of mov depends on j
		// 	tmp := make([]int, len(input))
		// 	for b := 0; b < batch; b++ {
		// 		for i := 0; i < min_wid; i++ {
		// 			idx := 2*min_wid*in_wid*b + 2*min_wid*j + i
		// 			tmp[idx] = input[idx]
		// 		}
		// 	}
		// 	rot := -j*min_wid + 2*pos*min_wid*min_wid
		// 	output = addSlice(output, rRot(tmp, rot))
		// }
	}

	return output
}

// Assume N/2 input vector
// reverse of extend_full (after strided conv -> normal)
// in_wid = input wid including padding
// kp_wid = keep wid
// 0 <= pos < 4 determines to which part the output is positioned at the final output
// ul : up (0) or low (1) part
func comprs_full_fast(input []float64, in_wid, kp_wid, pos, ul int) []float64 {
	mid_out := make([]float64, len(input))
	output := make([]float64, len(input))
	batch := 2 * len(input) / (in_wid * in_wid)
	if kp_wid < in_wid/2 {
		panic("keep width too small. less than in_wid/2")
	}
	pos = int(reverseBits(uint32(pos), 2))
	min_wid := in_wid / 4
	if in_wid%4 != 0 {
		panic("input wid not divisible by 4")
	}
	if in_wid%2 != 0 {
		panic("input wid not divisible by 2")
	}
	log_in_wid := 0
	for ; (1 << log_in_wid) < in_wid; log_in_wid++ {
	}

	for j := 0; j < 2*min_wid; j++ { // kinds of mov depends on j
		tmp := make([]float64, len(input))
		for b := 0; b < batch; b++ {
			for i := 0; i < min_wid; i++ {
				if (ul == 0) && (reverseBits(uint32(in_wid/2+j), log_in_wid) < uint32(kp_wid)) {
					idx := 2*min_wid*in_wid*b + 2*min_wid*j + i + in_wid*min_wid + min_wid
					tmp[idx] = input[idx]
				}
				if (ul == 1) && (reverseBits(uint32(in_wid/2+j), log_in_wid) < uint32(kp_wid)) && (reverseBits(uint32(min_wid+i), log_in_wid-1) < uint32(kp_wid-in_wid/2)) {
					idx := 2*min_wid*in_wid*b + 2*min_wid*j + i + in_wid*min_wid + min_wid
					tmp[idx] = input[idx]
				}
			}
		}
		rot := -j*min_wid + 2*min_wid*min_wid - min_wid
		mid_out = addSlice(mid_out, rRot(tmp, rot))
	}
	for b := 0; b < batch; b++ {
		tmp := make([]float64, len(input))
		for j := 0; j < 2*min_wid; j++ {
			for i := 0; i < min_wid; i++ {
				idx := 2*min_wid*in_wid*b + 3*in_wid/2*min_wid + j*min_wid + i
				tmp[idx] = mid_out[idx]
			}
		}
		rot := -3*b*min_wid*in_wid/2 + pos*min_wid*in_wid/2*batch - 3*min_wid*in_wid/2
		output = addSlice(output, rRot(tmp, rot))
	}

	return output
}

// generate vectors for comprs_full (N/2 input)
// returns the idx and rotations for each idx For comprs_full_hf
// vec_size = slots, in_wid = real in_wid including padding,
// CAUTION: rotation = -rotation (of comprs_full_hf)
func gen_comprs_full(vec_size, in_wid, kp_wid, pos, ul int) (r_idx map[int][]int) {
	r_idx = make(map[int][]int)
	batch := 2 * vec_size / (in_wid * in_wid)
	if kp_wid < in_wid/2 {
		panic("keep width too small. less than in_wid/2")
	}
	pos = int(reverseBits(uint32(pos), 2))
	padding := false
	min_wid := in_wid / 4
	if in_wid%4 != 0 {
		panic("input wid not divisible by 4")
	}
	if in_wid%2 != 0 {
		panic("input wid not divisible by 2")
	}
	log_in_wid := 0
	for ; (1 << log_in_wid) < in_wid; log_in_wid++ {
	}

	if padding {
		for j := 0; j < min_wid; j++ { // kinds of mov depends on j
			tmp := make([]int, vec_size)
			for b := 0; b < batch; b++ {
				for i := 0; i < min_wid; i++ {
					idx := 2*min_wid*in_wid*b + in_wid*j + i + min_wid*in_wid + min_wid
					tmp[idx] = 1
				}
			}
			rot := 2*j*min_wid - 2*pos*min_wid*min_wid + min_wid*in_wid + min_wid
			r_idx[rot] = tmp
		}
	} else {
		if ul == 0 {
			for j := 0; j < 2*min_wid; j++ { // kinds of mov depends on j
				tmp := make([]int, vec_size)
				for b := 0; b < batch; b++ {
					if reverseBits(uint32(in_wid/2+j), log_in_wid) < uint32(kp_wid) {
						for i := 0; i < min_wid; i++ {
							idx := 2*min_wid*in_wid*b + 2*min_wid*j + i + in_wid*min_wid + min_wid
							tmp[idx] = 1
						}
					}
				}
				rot := j*min_wid - 2*pos*min_wid*min_wid + min_wid + in_wid*min_wid
				r_idx[rot] = tmp
			}
		} else {
			for j := 0; j < 2*min_wid; j++ { // kinds of mov depends on j
				tmp := make([]int, vec_size)
				for b := 0; b < batch; b++ {
					for i := 0; i < min_wid; i++ {
						if (reverseBits(uint32(in_wid/2+j), log_in_wid) < uint32(kp_wid)) && (reverseBits(uint32(3*min_wid+i), log_in_wid-1) < uint32(kp_wid-in_wid/2)) {
							idx := 2*min_wid*in_wid*b + 2*min_wid*j + i + in_wid*min_wid + min_wid
							tmp[idx] = 1
						}
					}
				}
				rot := j*min_wid - 2*pos*min_wid*min_wid + min_wid + in_wid*min_wid
				r_idx[rot] = tmp
			}
		}
	}

	return r_idx
}

// generate vectors for comprs_full_fast (N/2 input)
// returns the idx and rotations for each idx For comprs_full_hf
// vec_size = slots, in_wid = real in_wid including padding,
// CAUTION: rotation = -rotation (of comprs_full_hf)
func gen_comprs_fast(vec_size, in_wid, kp_wid, pos, ul int) (m_idx, r_idx map[int][]int) {
	m_idx = make(map[int][]int)
	r_idx = make(map[int][]int)
	batch := 2 * vec_size / (in_wid * in_wid)

	if kp_wid < in_wid/2 {
		panic("keep width too small. less than in_wid/2")
	}
	pos = int(reverseBits(uint32(pos), 2))
	min_wid := in_wid / 4
	if in_wid%4 != 0 {
		panic("input wid not divisible by 4")
	}
	if in_wid%2 != 0 {
		panic("input wid not divisible by 2")
	}
	log_in_wid := 0
	for ; (1 << log_in_wid) < in_wid; log_in_wid++ {
	}

	for j := 0; j < 2*min_wid; j++ { // kinds of mov depends on j
		tmp := make([]int, vec_size)
		for b := 0; b < batch; b++ {
			for i := 0; i < min_wid; i++ {
				if (ul == 0) && (reverseBits(uint32(in_wid/2+j), log_in_wid) < uint32(kp_wid)) {
					idx := 2*min_wid*in_wid*b + 2*min_wid*j + i + in_wid*min_wid + min_wid
					tmp[idx] = 1
				}
				if (ul == 1) && (reverseBits(uint32(in_wid/2+j), log_in_wid) < uint32(kp_wid)) && (reverseBits(uint32(min_wid+i), log_in_wid-1) < uint32(kp_wid-in_wid/2)) {
					idx := 2*min_wid*in_wid*b + 2*min_wid*j + i + in_wid*min_wid + min_wid
					tmp[idx] = 1
				}
			}
		}
		rot := j*min_wid - 2*min_wid*min_wid + min_wid
		m_idx[rot] = tmp
	}
	for b := 0; b < batch; b++ { // kinds of mov depends on b
		tmp := make([]int, vec_size)
		for j := 0; j < 2*min_wid; j++ {
			for i := 0; i < min_wid; i++ {
				idx := 2*min_wid*in_wid*b + 3*in_wid/2*min_wid + j*min_wid + i
				tmp[idx] = 1
			}
		}
		rot := 3*b*min_wid*in_wid/2 - pos*min_wid*in_wid/2*batch + 3*min_wid*in_wid/2
		r_idx[rot] = tmp
	}

	return m_idx, r_idx
}

// generate vectors for comprs_full_fast (N/2 input)
// returns the idx and rotations for each idx For comprs_full_hf
// vec_size = full slots  = N/2, in_wid = real in_wid including padding,
// CAUTION: rotation = -rotation (of comprs_full_hf)
// log_sparse: 0 => full slots, 1 => half slots, Of the INPUT
// ul: 0(up), 1(low)
// pos: position after pack [only for full packing case]
func gen_comprs_sparse(vec_size, in_wid, kp_wid, log_sparse, ul, pos int) (m_idx, r_idx map[int][]int) {
	m_idx = make(map[int][]int)
	r_idx = make(map[int][]int)
	batch := 2 * vec_size / (in_wid * in_wid * (1 << log_sparse))

	// if kp_wid < in_wid/2 {
	// 	panic("keep width too small. less than in_wid/2")
	// }
	// pos = int(reverseBits(uint32(pos), 2))
	min_wid := in_wid / 2
	if in_wid%2 != 0 {
		panic("input wid not divisible by 2")
	}
	log_in_wid := 0
	for ; (1 << log_in_wid) < in_wid; log_in_wid++ {
	}

	if log_sparse != 0 {
		if pos != 0 {
			panic("No pos != 0 cases for log_sparse != 0")
		}
		for j := 0; j < min_wid; j++ { // kinds of mov depends on j
			tmp := make([]int, vec_size)
			for b := 0; b < batch; b++ {
				for i := 0; i < min_wid/2; i++ {
					for k := 0; k < 2; k++ {
						if (reverseBits(uint32(j), log_in_wid-1) < uint32(kp_wid)) && ((reverseBits(uint32(i), log_in_wid-2) + uint32(k)*uint32(min_wid)/2) < uint32(kp_wid)) {
							idx := k*in_wid*min_wid*batch + in_wid*in_wid*b/2 + in_wid*j/2 + i
							tmp[idx] = 1
						}
					}
				}
			}
			// repeatedly write tmp elements for log_sparse > 1 cases.
			for i := 0; i < vec_size/(1<<(log_sparse-1)); i++ {
				for k := 1; k < (1 << (log_sparse - 1)); k++ {
					tmp[i+k*vec_size/(1<<(log_sparse-1))] = tmp[i]
				}
			}
			rot := j * min_wid / 2
			m_idx[rot] = tmp
		}

		for b := 0; b < batch; b++ { // kinds of mov depends on b
			tmp := make([]int, vec_size)
			for j := 0; j < min_wid; j++ {
				for i := 0; i < min_wid/2; i++ {
					for k := 0; k < 2; k++ {
						idx := k*in_wid*min_wid*batch + b*in_wid*in_wid/2 + j*min_wid/2 + i
						tmp[idx] = 1
					}
				}
			}
			// repeatedly write tmp elements for log_sparse > 1 cases.
			for i := 0; i < vec_size/(1<<(log_sparse-1)); i++ {
				for k := 1; k < (1 << (log_sparse - 1)); k++ {
					tmp[i+k*vec_size/(1<<(log_sparse-1))] = tmp[i]
				}
			}
			rot := 3 * b * min_wid * min_wid / 2
			r_idx[rot] = tmp
		}
	} else {
		if batch > 8*min_wid {
			for j := 0; j < min_wid; j++ { // kinds of mov depends on j and b
				for bk := 0; bk < 8; bk++ {
					tmp := make([]int, vec_size)
					for b := 0; b < batch/8; b++ {
						for i := 0; i < min_wid/2; i++ {
							if (ul == 0) && (reverseBits(uint32(j), log_in_wid-1) < uint32(kp_wid)) && (reverseBits(uint32(i), log_in_wid-2) < uint32(kp_wid)) {
								idx := 8*in_wid*min_wid*b + bk*min_wid*in_wid + min_wid*j + i
								tmp[idx] = 1
							}
							if (ul == 1) && (reverseBits(uint32(j), log_in_wid-1) < uint32(kp_wid)) && (reverseBits(uint32(i), log_in_wid-2)+uint32(min_wid/2) < uint32(kp_wid)) {
								idx := 8*in_wid*min_wid*b + bk*min_wid*in_wid + min_wid*j + i
								tmp[idx] = 1
							}
						}
					}
					rot := j*min_wid/2 + 7*bk*min_wid*min_wid/2
					m_idx[rot] = tmp
				}
			}

			for b := 0; b < batch/8; b++ { // kinds of mov depends on b
				tmp := make([]int, vec_size)
				for bk := 0; bk < 8; bk++ {
					for j := 0; j < min_wid; j++ {
						for i := 0; i < min_wid/2; i++ {
							idx := 8*b*in_wid*min_wid + bk*min_wid*min_wid/2 + j*min_wid/2 + i
							tmp[idx] = 1
						}
					}
				}
				rot := 3*b*8*min_wid*min_wid/2 - int(reverseBits(uint32(pos), 2))*batch*min_wid*min_wid/2
				r_idx[rot] = tmp
			}
		} else if batch > 4*min_wid { //we may move 4*j for optimizations
			for j := 0; j < min_wid; j++ { // kinds of mov depends on j and b
				for bk := 0; bk < 4; bk++ {
					tmp := make([]int, vec_size)
					for b := 0; b < batch/4; b++ {
						for i := 0; i < min_wid/2; i++ {
							if (ul == 0) && (reverseBits(uint32(j), log_in_wid-1) < uint32(kp_wid)) && (reverseBits(uint32(i), log_in_wid-2) < uint32(kp_wid)) {
								idx := 4*in_wid*min_wid*b + bk*min_wid*in_wid + min_wid*j + i
								tmp[idx] = 1
							}
							if (ul == 1) && (reverseBits(uint32(j), log_in_wid-1) < uint32(kp_wid)) && (reverseBits(uint32(i), log_in_wid-2)+uint32(min_wid/2) < uint32(kp_wid)) {
								idx := 4*in_wid*min_wid*b + bk*min_wid*in_wid + min_wid*j + i
								tmp[idx] = 1
							}
						}
					}
					rot := j*min_wid/2 + 3*bk*min_wid*min_wid/2
					m_idx[rot] = tmp
				}
			}

			for b := 0; b < batch/4; b++ { // kinds of mov depends on b
				tmp := make([]int, vec_size)
				for bk := 0; bk < 4; bk++ {
					for j := 0; j < min_wid; j++ {
						for i := 0; i < min_wid/2; i++ {
							idx := 4*b*in_wid*min_wid + bk*min_wid*min_wid/2 + j*min_wid/2 + i
							tmp[idx] = 1
						}
					}
				}
				rot := 3*b*4*min_wid*min_wid/2 - int(reverseBits(uint32(pos), 2))*batch*min_wid*min_wid/2
				r_idx[rot] = tmp
			}
		} else {
			for j := 0; j < min_wid; j++ { // kinds of mov depends on j and b
				tmp := make([]int, vec_size)
				for b := 0; b < batch; b++ {
					for i := 0; i < min_wid/2; i++ {
						if (ul == 0) && (reverseBits(uint32(j), log_in_wid-1) < uint32(kp_wid)) && (reverseBits(uint32(i), log_in_wid-2) < uint32(kp_wid)) {
							idx := in_wid*min_wid*b + min_wid*j + i
							tmp[idx] = 1
						}
						if (ul == 1) && (reverseBits(uint32(j), log_in_wid-1) < uint32(kp_wid)) && (reverseBits(uint32(i), log_in_wid-2)+uint32(min_wid/2) < uint32(kp_wid)) {
							idx := in_wid*min_wid*b + min_wid*j + i
							tmp[idx] = 1
						}
					}
				}
				rot := j * min_wid / 2
				m_idx[rot] = tmp
			}

			for b := 0; b < batch; b++ { // kinds of mov depends on b
				tmp := make([]int, vec_size)
				for j := 0; j < min_wid; j++ {
					for i := 0; i < min_wid/2; i++ {
						idx := b*in_wid*min_wid + j*min_wid/2 + i
						tmp[idx] = 1
					}
				}
				rot := 3*b*min_wid*min_wid/2 - int(reverseBits(uint32(pos), 2))*batch*min_wid*min_wid/2
				r_idx[rot] = tmp
			}
		}
	}

	return m_idx, r_idx
}
