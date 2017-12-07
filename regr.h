/* ========================================================
 *   Copyright (C) 2017 All rights reserved.
 *   
 *   filename : regr.h
 *   author   : ***
 *   date     : 2017-12-06
 *   info     : 
 * ======================================================== */

#ifndef _REGR_H
#define _REGR_H

#include "regcfg.h"
#include "data.h"

typedef double (*EVAL_FN)(double *x, void * ds);
typedef void   (*GRAD_FN)(double *x, void * ds, double * g);

typedef struct _regr {
    DATA   * train_ds;        /* train data set */
    DATA   * test_ds;         /* test  data set */
    int      feature_len;     /* feature length */
    double * x;               /* feature result */
    EVAL_FN  eval_fn;         /* eval function  */
    GRAD_FN  grad_fn;         /* gradient func  */
    REGP     reg_p;
}REGR;

REGR * create_model(EVAL_FN eval_fn, GRAD_FN grad_fn);
int    init_model(REGR * regr);
int    learn_model(REGR * reg);
void   save_model(REGR * reg, int n);
void   free_model(REGR * reg);

#endif //REGR_H
