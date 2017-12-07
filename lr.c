/* ========================================================
 *   Copyright (C) 2015 All rights reserved.
 *   
 *   filename : lr.c
 *   author   : liuzhiqiangruc@126.com
 *   date     : 2015-08-27
 *   info     : LR implementation
 *              Using regression framework
 * ======================================================== */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "auc.h"
#include "lr.h"

#define sign(x) (x >0.0?1:(x<0.0?-1:0)) 

static void l1_norm(double *x, double *g, double lambda, int n){
    for (int i = 0; i < n; i++){
        if (x[i] > 0.0){
            g[i] += lambda;
        }
        else if (x[i] < 0.0){
            g[i] -= lambda;
        }
        else if (g[i] > lambda){
            g[i] -= lambda;
        }
        else if (g[i] < -lambda){
            g[i] += lambda;
        }
    }
}

void lr_grad(REGR *regr, double *g){
    DATA * ds   = regr->train_ds;
    int i = 0, j = 0;
    double yest  = 0.0, hx = 0.0;
    memset(g, 0, sizeof(double) * ds->col);
    for (i = 0; i < ds->row; i++) {
        yest = 0.0;
        for (j = 0; j < ds->len[i]; j++){
            yest += regr->x[ds->ids[ds->clen[i] + j]] * (ds->fea_type == BINARY ? 1.0 : ds->vals[ds->clen[i] + j]);
        }
        hx = yest < -30.0 ? 0.0 : (yest > 30.0 ? 1.0 : 1.0 / (1.0 + exp(-yest)));
        for (j = 0; j < ds->len[i]; j++){
            g[ds->ids[ds->clen[i] + j]] += (hx - ds->y[i]) * (ds->fea_type == BINARY ? 1.0 : ds->vals[ds->clen[i] + j]);
        }
    }
    if (regr->reg_p.r == 2){    // for l2 norm
        for (i = 0; i < ds->col; i++){
            g[i] += regr->reg_p.gamma * (regr->x[i]) * 2;
        }
    }
    else if (regr->reg_p.r == 1){ // for l1 norm
        l1_norm(regr->x, g, regr->reg_p.gamma, ds->col);
    }
}

static double loss(double * x, DATA * ds, double * hy){
    double loss = 0.0, yest = 0.0, add = 0.0;
    int i, j;
    for (i = 0; i < ds->row; i++){
        yest = 0.0;
        for (j = 0; j < ds->len[i]; j++){
            yest += x[ds->ids[ds->clen[i] + j]] * (ds->fea_type == BINARY ? 1.0 : ds->vals[ds->clen[i] + j]);
        }
        if (hy) hy[i] = yest;
        add = yest > 30.0 ? yest : (yest < -30.0 ? 0.0 : log(1.0 + exp(yest)));
        add -= (ds->y[i] > 0 ? yest : 0.0);
        loss += add;
    }
    return loss;
}

// repo the train and test loss and auc status
// and return the train data loss
double lr_repo(REGR *regr){
    double train_loss, test_loss = 0.0, train_auc, test_auc = 0.0;
    double *train_hy = (double*)calloc(regr->train_ds->row, sizeof(double));
    double *test_hy  = NULL;
    train_loss = loss(regr->x, regr->train_ds, train_hy);
    train_auc  = auc(regr->train_ds->row, train_hy, regr->train_ds->y);
    fprintf(stderr, "train_loss : %.8f, train_auc : %.8f", train_loss, train_auc); 
    if (regr->test_ds){
        test_hy = (double*)calloc(regr->test_ds->row, sizeof(double));
        test_loss = loss(regr->x, regr->test_ds, test_hy);
        test_auc  = auc(regr->test_ds->row, test_hy, regr->test_ds->y);
        fprintf(stderr, ";  test_loss : %.8f, test_auc : %.8f", test_loss, test_auc); 
    }
    fprintf(stderr, "\n");
    return train_loss;
}

REGR * create_lr_model(){
    REGR * lr = create_model(lr_grad, lr_repo);
    return lr;
}
