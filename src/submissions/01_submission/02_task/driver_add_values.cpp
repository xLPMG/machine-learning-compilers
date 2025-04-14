#include <iostream>

extern "C" void add_values(
    int32_t * a,
    int32_t * b,
    int32_t * c 
);

int main()
{
    std::cout << "Calling assembly 'add_value' function...\n";

    // Test Data
    int32_t l_value_1 = 4;
    int32_t * l_ptr_1 = &l_value_1;

    int32_t l_value_2 = 7;
    int32_t * l_ptr_2 = &l_value_2;

    int32_t l_value_o;
    int32_t * l_ptr_o = &l_value_o;

    // Call to add_values
    add_values( l_ptr_1, l_ptr_2, l_ptr_o );

    std::cout << "l_data_1 / l_value_2 / l_value_o\n"
        << l_value_1 << " / "
        << l_value_2 << " / "
        << l_value_o
        << std::endl;

    return 0;
}
