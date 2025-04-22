#include <iostream>

extern "C" void hello_assembly();

int main() 
{
    std::cout << "Calling hello_assembly ...\n";
    
    // Call to hello_assembly
    hello_assembly();

    std::cout << "... returned from function call!\n";
    return 0;
}
