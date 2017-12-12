/* ========================================================
 *   Copyright (C) 2017 All rights reserved.
 *   
 *   filename : main.c
 *   author   : ***
 *   date     : 2017-12-07
 *   info     : 
 * ======================================================== */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "deeplr.h"
#include "lr.h"

void help() {
    fprintf(stderr, "\nLR [Logistic Regression] usage:        \n");
    fprintf(stderr, "\n./lr -a <double> -g <double> -l <double> -b <int> -r <int> -k <int> -n <int> -s <int> -f <string> -t <string> -o <string>\n");
    fprintf(stderr, "     -a  learning rate                   \n");
    fprintf(stderr, "     -g  regulization paramenter         \n");
    fprintf(stderr, "     -l  Convergence tolerance           \n");
    fprintf(stderr, "     -b  1:binary or else                \n");
    fprintf(stderr, "     -r  1:L1 Norm; 2: L2 Norm           \n");
    fprintf(stderr, "     -k  length of latent factor         \n");
    fprintf(stderr, "     -n  max iteration count             \n");
    fprintf(stderr, "     -s  savestep                        \n");
    fprintf(stderr, "     -f  train input file                \n");
    fprintf(stderr, "     -t  test  input file                \n");
    fprintf(stderr, "     -o  otuput dir                      \n");
}

int main(int argc, char *argv[]) {
    srand(time(NULL));
    REGR * lr = create_deeplr_model();
    if (-1 == parse_command_line(&(lr->reg_p), argc, argv)){
        help();
        goto except;
    }
    fprintf(stderr, "command line parse done\n");
    if (-1 == init_model(lr)){
        goto except;
    }
    fprintf(stderr, "load data done\n");
    fprintf(stderr, "train: %d, lenx: %d\n", lr->train_ds->row, lr->feature_len);
    if (lr->test_ds){
        fprintf(stderr, " test: %d\n", lr->test_ds->row);
    }

    long t1 = time(NULL);
    lr->learn_fn(lr);
    long t2 = time(NULL);
    fprintf(stderr, "using seconds : %ld\n", t2 - t1);
    save_model(lr, lr->reg_p.n);
    free_model(lr);
    lr = NULL;
    return 0;
except:
    return -1;
}
