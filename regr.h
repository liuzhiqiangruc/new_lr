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

typedef struct _regr REGR;

// calculation the gradient for iteration
typedef void   (*GRAD_FN)(REGR * regr, double * g); 

// repo the process of iteration, and return the train ds loss with out regulization
typedef double (*REPO_FN)(REGR * regr);   

struct _regr {
    DATA   * train_ds;        /* train data set */
    DATA   * test_ds;         /* test  data set */
    int      feature_len;     /* feature length */
    int      K;               /* latent length  */
    double * x;               /* feature result */
    GRAD_FN  grad_fn;         /* gradient func  */
    REPO_FN  repo_fn;         /* report func    */
    REGP     reg_p;           /* init parament  */
};

REGR * create_model(GRAD_FN grad_fn, REPO_FN repo_fn);
int    init_model(REGR * regr);
int    learn_model(REGR * reg);
void   save_model(REGR * reg, int n);
void   free_model(REGR * reg);

#endif //REGR_H
