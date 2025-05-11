    .text
    .global test_asm
test_asm:
    LD1 {V0.S}[3], [X0], #4
    LD1 {V0.D}[1], [X0], #8

    ret