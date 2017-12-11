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
    if (regr->reg_p.k == 0) {   // just for simple lr
        regr->x = (double*)calloc(regr->feature_len, sizeof(double));
    }
    else{                       // with k nodes as latent layer
        regr->x = (double*)calloc((regr->feature_len + 1) * regr->reg_p.k, sizeof(double));
    }
    return 0;
}

REGR * create_model(LEARN_FN learn_fn){
    if (!learn_fn){
        return NULL;
    }
    REGR *regr = (REGR*)calloc(1, sizeof(REGR));
    regr->learn_fn = learn_fn;
    return regr;
}
