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

int parse_command_line(REGP *p, int argc, char *argv[]){
    double a = 0, g = 0, l = 1e-5;
    int b = 0, r = 1, n = 10, s = 10, k = 0;
    char * f = NULL;
    char * t = NULL;
    char * o = "./";
    int i = 0;
    char * arg = NULL;

    if ((argc & 1) == 0){
        fprintf(stderr, "command line not well formatted\n");
        return -1;
    }

    while (i < argc) {
        arg = argv[i];
        if (0 == strcmp(arg,"-a")){
            a = atof(argv[++i]);
        }
        else if (0 == strcmp(arg,"-l")){
            l = atof(argv[++i]);
        }
        else if (0 == strcmp(arg,"-g")){
            g = atof(argv[++i]);
        }
        else if (0 == strcmp(arg,"-b")){
            b = atoi(argv[++i]);
        }
        else if (0 == strcmp(arg,"-n")){
            n = atoi(argv[++i]);
        }
        else if (0 == strcmp(arg,"-s")){
            s = atoi(argv[++i]);
        }
        else if (0 == strcmp(arg,"-r")){
            r = atoi(argv[++i]);
        }
        else if (0 == strcmp(arg,"-k")){
            k = atoi(argv[++i]);
        }
        else if (0 == strcmp(arg,"-f")){
            f = argv[++i];
        }
        else if (0 == strcmp(arg,"-t")){
            t = argv[++i];
        }
        else if (0 == strcmp(arg,"-o")){
            o = argv[++i];
        }
        i += 1;
    }
    if (NULL == f) {
        fprintf(stderr,"no input data exit(-1)\n");
        return -1;
    }
    if (r != 1 && r != 2){
        fprintf(stderr, "method must be 1, or 2\n");
        return -1;
    }
    if (b != 0 && b != 1){
        fprintf(stderr, "binary must be 0, or 1\n");
        return -1;
    }
    p->alpha       = a;
    p->gamma       = g;
    p->toler       = l;
    p->b           = b;
    p->r           = r;
    p->k           = k;
    p->n           = n;
    p->s           = s;
    p->train_input = f;
    p->test_input  = t;
    p->out_dir     = o;

    return 0;
}

int main(int argc, char *argv[]) {
    REGR * lr = create_lr_model();
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
    return 0;

except:
    return -1;
}


