Neon
======

Execution Throughput and Latency
--------------------------------

For this task, we were benchmarking the execution throughput and latency for FP32 neon instructions.
Specifically, we were looking at:

1. `FMLA (vector) <https://developer.arm.com/documentation/ddi0602/2025-03/SIMD-FP-Instructions/FMLA--vector---Floating-point-fused-multiply-add-to-accumulator--vector-->`_ instruction
2. `FMADD (scalar) <https://developer.arm.com/documentation/ddi0602/2025-03/SIMD-FP-Instructions/FMADD--Floating-point-fused-multiply-add--scalar--?lang=en>`_ instruction

Throughput
^^^^^^^^^^^

As a first step we were comparing the throughput of:

1. FMLA (vector) with arrangement specifier ``4S``
2. FMLA (vector) with arrangement specifier ``2S``
3. FMADD (scalar), single-precision variant

To compare the throughput, we created three assembly programs, that were executing
several of these operations. To get proper results we were looking for any dependencies
regarding the source or destination registers of the operations.
The calculations that we were ending up with were:

.. literalinclude:: ../../../src/submissions/03_neon/01_execution_throughput_and_latency/fmla_4s_instr.s
    :language: asm
    :lines: 31-71
    :caption: FMLA (vector) with arrangement specifier ``4S``

.. literalinclude:: ../../../src/submissions/03_neon/01_execution_throughput_and_latency/fmla_2s_instr.s
    :language: asm
    :lines: 31-71
    :caption: FMLA (vector) with arrangement specifier ``2S``

.. literalinclude:: ../../../src/submissions/03_neon/01_execution_throughput_and_latency/fmadd_instr.s
    :language: asm
    :lines: 39-82
    :caption: FMADD (scalar), single-precision variant

In order to measure the throughput of these instructions we developed a C++ microbenchmark.
For each instruction we firstly performed a warm up, measured the time, counted the operations and then calculated the GFLOPs.

.. literalinclude:: ../../../src/submissions/03_neon/01_execution_throughput_and_latency/microbench.cpp
    :language: cpp
    :lines: 61-73
    :caption: Example benchmark for FMLA (vector) with arrangement specifier ``4S``

For the ``2S`` and the FMADD (scalar) instructions, we simply adjusted the calculations for the operations slightly:

.. literalinclude:: ../../../src/submissions/03_neon/01_execution_throughput_and_latency/microbench.cpp
    :language: cpp
    :lines: 85-89
    :caption: Calculations for FMLA (vector) with arrangement specifier ``2S``

.. literalinclude:: ../../../src/submissions/03_neon/01_execution_throughput_and_latency/microbench.cpp
    :language: cpp
    :lines: 101-105
    :caption: Calculations for FMLA (vector) with arrangement specifier ``2S``

For this benchmarking task we obtained the following results:

.. literalinclude:: ../../../src/submissions/03_neon/01_execution_throughput_and_latency/benchmarking_results.txt
    :language: text
    :lines: 1-23
    :caption: Throughput results for the three instructions

It can be clearly seen that the FMLA (vector) with arrangement specifier ``4S`` instruction performs approximately 
2.5 times as many floating point operations than the FMLA (vector) with arrangement specifier ``2S`` instruction. 
Further the FMLA (vector) with arrangement specifier ``2S`` instructions performs at approximately 2.5 times more
floating point operations than the FMADD (scalar) instruction. 

This clearly shows that leveraging data-level parallelism (vector-based) can yield a much higher throughput, than
using only scalar operations.

Latency
^^^^^^^^