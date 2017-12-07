/* ========================================================
 *   Copyright (C) 2017 All rights reserved.
 *   
 *   filename : regr.c
 *   author   : liuzhiqiangruc@126.com
 *   date     : 2017-12-06
 *   info     : regression using gradient method
 * ======================================================== */
#include <stdio.h>
#include <stdlib.h>
#include "regr.h"

int init_model(REGR * regr){
    Hash * hs = hash_create(1 << 20, STRING);
    regr->train_ds = data_load(regr->reg_p.train_input, ROW, regr->reg_p.b == 1 ? BINARY : NOBINARY, NO_INITED, hs);
    if (!regr->train_ds){
        hash_free(hs);
        hs = NULL;
        return -1;
    }
    regr->test_ds = data_load(regr->reg_p.test_input, ROW, regr->reg_p.b == 1 ? BINARY : NOBINARY, INITED, hs);
    free(hs);
    hs = NULL;
    regr->feature_len = regr->train_ds->col;
    regr->x = (double*)calloc(regr->feature_len, sizeof(double));
    return 0;
}

REGR * create_model(GRAD_FN grad_fn, REPO_FN repo_fn){
    if (!repo_fn || !grad_fn){
        return NULL;
    }
    REGR *regr = (REGR*)calloc(1, sizeof(REGR));
    regr->repo_fn = repo_fn;
    regr->grad_fn = grad_fn;
    return regr;
}

int learn_model(REGR * regr){
    int i, j;
    double *g = NULL;
    double delta = 0.0, loss = 0.0, new_loss = 0.0;
    g = (double*)calloc(regr->feature_len, sizeof(double));
    loss = regr->repo_fn(regr);
    for (i = 0; i < regr->reg_p.n; i++){
        regr->grad_fn(regr, g);
        for (j = 0; j < regr->feature_len; j++){
            delta = regr->reg_p.alpha * g[j];
            if (regr->reg_p.r == 1){
                if (regr->x[j] > 0.0 && delta > regr->x[j]){
                    delta = 0.0;
                }
                if (regr->x[j] < 0.0 && delta < regr->x[j]){
                    delta = 0.0;
                }
            }
            regr->x[j] -= delta;
        }
        new_loss = regr->repo_fn(regr);
        if (loss - new_loss <= regr->reg_p.toler){
            fprintf(stderr, "conv done!!!\n");
            break;
        }
        loss = new_loss;
    }
    return 0;
}
