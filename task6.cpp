#include <iostream>
#include <cstdio>
#include <cstdlib>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " N" << std::endl;
        return 1;
    }

    // Convert command line argument to an integer
    int N = std::atoi(argv[1]);

    // Print integers from 0 to N (inclusive) using printf
    for (int i = 0; i <= N; ++i) {
        printf("%d", i);
        if (i != N) {
            printf(" ");
        }
    }
    printf("\n");

    // Print integers from N to 0 (inclusive) using std::cout
    for (int i = N; i >= 0; --i) {
        std::cout << i;
        if (i != 0) {
            std::cout << " ";
        }
    }
    std::cout << "\n";

    return 0;
}
