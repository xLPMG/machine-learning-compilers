Assembly
===========

Hello Assembly
--------------

Information about the given data files.

Task 1
^^^^^^

First we would like to compile :code:`hello_assembly.c` using the GCC and Clang compiler.

The **GCC** compiler can be used with the following statement.

.. code-block:: bash

    gcc -S hello_assembly.c

The **Clang** compiler can be used with the following statement.

.. code-block:: bash

    clang -S hello_assembly.c


Task 2
^^^^^^

After compilation we would like for each generated assembly file to:

1. Find the "Hello Assembly Language!" string
2. Identify the instructions that the compilers insert to conform to the procedure call standard
3. Identify the function call to libc that prints the string

GCC
"""

The **GCC** :code:`gcc_hello_assembly.s` looks like:

.. literalinclude:: ../../../src/submissions/01_submission/01_task/gcc_hello_assembly.s 
    :language: asm
    :linenos:
    :caption: GCC compiled file

1. The string "Hello Assembly Language!" can be found in *line 7*.
2. The instructions for the procedure call standard:

.. code-block:: asm

    stp	x29, x30, [sp, -16]!
    mov	x29, sp
    ldp x29, x30, [sp], 16
    ret

3. The function call is that prints the string:

.. code-block:: asm

    bl	puts

Clang
"""""

The **Clang** :code:`clang_hello_assembly.s` looks like:

.. literalinclude:: ../../../src/submissions/01_submission/01_task/clang_hello_assembly.s 
    :language: asm
    :linenos:
    :caption: Clang compiled file

1. The string "Hello Assembly Language!" can be found in *line 31*.
2. The instructions for the procedure call standard:

.. code-block:: asm

    stp	x29, x30, [sp, -16]!
    mov	x29, sp
    ldp x29, x30, [sp], 16
    ret

3. The function call is that prints the string:

.. code-block:: asm

    bl	printf

Task 3
^^^^^^

Now we would like to test the :code:`hello_assembly` function with a C++ driver. 

.. literalinclude:: ../../../src/submissions/01_submission/01_task/driver_hello_assembly.cpp 
    :language: cpp
    :linenos:
    :caption: C++ driver

To compile and run this code, we can use:

.. code-block:: bash

    g++ driver_hello_assembly.cpp gcc_hello_assembly.s -o hello_assembly
    ./hello_assembly

The output is then:

.. code-block:: bash

    Calling hello_assembly ...
    Hello Assembly Language!
    ... returned from function call!

Assembly Function
-----------------
