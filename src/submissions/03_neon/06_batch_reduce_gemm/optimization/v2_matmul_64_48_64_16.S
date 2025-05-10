    .text
    .type v2_matmul_64_48_64_16, %function
    .global v2_matmul_64_48_64_16
    /*
    * Computes C+=AB for three matrices 
    * with the dimensions M=64, N=48, and K=64.
    *
    * @param x0 pointer to column-major matrix A.
    * @param x1 pointer to column-major matrix B.
    * @param x2 pointer to column-major matrix C.
    * @param x3 leading dimension of A.
    * @param x4 leading dimension of B.
    * @param x5 leading dimension of C.
    */
v2_matmul_64_48_64_16:
// ------------------------------------------
// START PCS
// ------------------------------------------
    // save frame pointer and link register
    stp fp, lr, [sp, #-16]!
    // update frame pointer to current stack pointer
    mov fp, sp

    // save callee-saved registers
    stp x19, x20, [sp, #-16]!
    stp x21, x22, [sp, #-16]!
    stp x23, x24, [sp, #-16]!
    stp x25, x26, [sp, #-16]!
    stp x27, x28, [sp, #-16]!

    stp  d8,  d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!
// ------------------------------------------
// END PCS
// ------------------------------------------

    // multiply strides with float size
    lsl x3, x3, #2 // lda in bytes
    lsl x4, x4, #2 // ldb in bytes
    lsl x5, x5, #2 // ldc in bytes
    lsl x6, x6, #2 // br_stride_a in bytes
    lsl x7, x7, #2 // br_stride_b in bytes

    mov x8, #6
    mul x22, x4, x8 // ldb * 6 columns
    mul x23, x5, x8 // ldc * 6 columns

    // set base matrix pointers
    mov x19, x0 // A
    mov x20, x1 // B
    mov x21, x2 // C

    // Batch counter
    mov x24, #16

_n_batch:

    // N loop counter
    mov x25, #8 // 48/6 = 8 blocks

_n_loop:

    // M loop counter
    mov x11, #4 // 64/16 = 4 blocks

    // set matrix pointers
    mov  x8,  x0 // A
    mov  x9, x20 // B
    mov x10, x21 // C

_m_loop:
// ------------------------------------------
// START matmul_16_6_64
// ------------------------------------------

    // LOAD MATRIX C
    mov x12, x10
    // first column
    ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x12]
    // second column
    add x12, x12, x5
    ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x12]
    // third column
    add x12, x12, x5
    ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x12]
    // fourth column
    add x12, x12, x5
    ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x12]
    // fifth column
    add x12, x12, x5
    ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x12]
    // sixth column
    add x12, x12, x5
    ld1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x12]

    // K loop counter
    mov x14, #64
    // set start of A
    mov x15, x8
    // set start of B
    mov x16, x9
    // init row count of B
    mov x17, #0
_k_loop:
    // load column of A (16 values)
    ld1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x15]

    // B: COLUMN 0
    ldr s29, [x16]
    fmla v0.4s, v24.4s, v29.s[0]
    fmla v1.4s, v25.4s, v29.s[0]
    fmla v2.4s, v26.4s, v29.s[0]
    fmla v3.4s, v27.4s, v29.s[0]

    // B: COLUMN 1
    add x16, x16, x4
    ldr s29, [x16]
    fmla v4.4s, v24.4s, v29.s[0]
    fmla v5.4s, v25.4s, v29.s[0]
    fmla v6.4s, v26.4s, v29.s[0]
    fmla v7.4s, v27.4s, v29.s[0]

    // B: COLUMN 2
    add x16, x16, x4
    ldr s29, [x16]
    fmla  v8.4s, v24.4s, v29.s[0]
    fmla  v9.4s, v25.4s, v29.s[0]
    fmla v10.4s, v26.4s, v29.s[0]
    fmla v11.4s, v27.4s, v29.s[0]

    // B: COLUMN 3
    add x16, x16, x4
    ldr s29, [x16]
    fmla v12.4s, v24.4s, v29.s[0]
    fmla v13.4s, v25.4s, v29.s[0]
    fmla v14.4s, v26.4s, v29.s[0]
    fmla v15.4s, v27.4s, v29.s[0]

    // B: COLUMN 4
    add x16, x16, x4
    ldr s29, [x16]
    fmla v16.4s, v24.4s, v29.s[0]
    fmla v17.4s, v25.4s, v29.s[0]
    fmla v18.4s, v26.4s, v29.s[0]
    fmla v19.4s, v27.4s, v29.s[0]

    // B: COLUMN 5
    add x16, x16, x4
    ldr s29, [x16]
    fmla v20.4s, v24.4s, v29.s[0]
    fmla v21.4s, v25.4s, v29.s[0]
    fmla v22.4s, v26.4s, v29.s[0]
    fmla v23.4s, v27.4s, v29.s[0]

    // move to next column of A
    add x15, x15, x3
    // move to next row of B
    mov x16, x9
    add x17, x17, #4
    add x16, x16, x17

    // decrement loop counter
    sub x14, x14, #1
    // check if loop counter is zero
    cbnz x14, _k_loop
// END K LOOP

    // STORE MATRIX C
    mov x12, x10
    // first column
    st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x12]
    // second column
    add x12, x12, x5
    st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x12]
    // third column
    add x12, x12, x5
    st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x12]
    // fourth column
    add x12, x12, x5
    st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x12]
    // fifth column
    add x12, x12, x5
    st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x12]
    // sixth column
    add x12, x12, x5
    st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x12]

// ------------------------------------------
// END matmul_16_6_64
// ------------------------------------------

    // increase A and C pointers for next block
    // (jump 16 values)
    add x8, x8, #16*4
    add x10, x10, #16*4

    // decrement m loop counter
    sub x11, x11, #1
    // check if loop counter is zero
    cbnz x11, _m_loop
// END M LOOP

    // increase B and C pointers for next block
    // (jump 6 columns) 6*x4, 6*x5
    add x20, x20, x22
    add x21, x21, x23

    // decrement n loop counter
    sub x25, x25, #1
    // check if loop counter is zero
    cbnz x25, _n_loop
// END N LOOP

    // next A matrix
    add x0, x0, x6 // A
    mov x8, x0 // A

    // next B matrix
    add x1, x1, x7 // B
    mov x20, x1 // B

    // restore Pointer for matrix C
    mov x21, x2 // C
    mov x10, x21 // C

    sub x24, x24, #1

    cbnz x24, _n_batch
// END N BATCH

// ------------------------------------------
// START PCS
// ------------------------------------------
    // restore callee-saved registers
    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp  d8,  d9, [sp], #16

    ldp x27, x28, [sp], #16
    ldp x25, x26, [sp], #16
    ldp x23, x24, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16

    // restore frame pointer and link register
    ldp fp, lr, [sp], #16
// ------------------------------------------
// END PCS
// ------------------------------------------
    ret
