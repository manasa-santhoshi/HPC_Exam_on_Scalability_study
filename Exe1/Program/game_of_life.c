#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include "gol.h"

int main(int argc, char** argv) {
    int action = 0;          // 1 = init, 2 = run
    int x = -1, y = -1;      // matrix dimensions
    int e = 1;               // evolution mode: 0 = ordered, 1 = static
    int n = 50;              // number of steps
    int s = 50;              // snapshot frequency
    char* fname = NULL;      // filename

    // Parse command-line arguments
    int opt;
    while ((opt = getopt(argc, argv, "irx:y:e:f:n:s:")) != -1) {
        switch (opt) {
            case 'i':
                action = 1;
                break;
            case 'r':
                action = 2;
                break;
            case 'x':
                x = atoi(optarg);
                break;
            case 'y':
                y = atoi(optarg);
                break;
            case 'e':
                e = atoi(optarg);
                break;
            case 'f':
                fname = optarg;
                break;
            case 'n':
                n = atoi(optarg);
                break;
            case 's':
                s = atoi(optarg);
                break;
            default:
                fprintf(stderr, "Unknown option: -%c\n", opt);
                exit(1);
        }
    }

    // Validate inputs
    if (fname == NULL) {
        fprintf(stderr, "Error: -f <filename> is required.\n");
        exit(1);
    }

    if (action == 0) {
        fprintf(stderr, "Error: specify -i (initialise) or -r (run).\n");
        exit(1);
    }

    if (action == 1 && (x <= 0 || y <= 0)) {
        fprintf(stderr, "Error: -x and -y must be positive integers when initialising (-i).\n");
        exit(1);
    }

    // Dispatch to correct function
    if (action == 1) {
        initialise_playground(x, y, 1, fname, argc, argv);
    } else if (action == 2) {
        if (e == 1) {
            static_evolution(fname, n, s, argc, argv);
        } else if (e == 0) {
            ordered_evolution(fname, n, s, argc, argv);
        } else {
            fprintf(stderr, "Invalid -e value. Use 0 (ordered) or 1 (static).\n");
            exit(1);
        }
    } else {
        fprintf(stderr, "Invalid action. Use -i or -r.\n");
        exit(1);
    }

    return 0;
}
