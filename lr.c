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
#include "lr.h"

void lr_grad(double *x, void *_regr, double *g){
    REGR * regr = (REGR*)_regr;
    DATA * ds   = regr->train_ds;
    int i = 0, j = 0;
    int row = ds->row;
    int col = ds->col;
    double yest  = 0.0, hx = 0.0;
    double *y    = ds->y;
    double *vals = ds->vals;
    unsigned int * ids = ds->ids;
    unsigned int * len = ds->len;
    unsigned int * clen = ds->clen;
    memset(g, 0, sizeof(double) * col);
    for (i = 0; i < row; i++) {
        yest = 0.0;
        if (ds->fea_type == BINARY){
            for (j = 0; j < len[i]; j++){
                yest += x[ids[clen[i] + j]];
            }
        }
        else {
            for (j = 0; j < len[i]; j++){
                yest += x[ids[clen[i] + j]] * vals[clen[i] + j];
            }
        }
        if (yest < -30) {
            hx = 0.0;
        } 
        else if (yest > 30) {
            hx = 1.0;
        } 
        else {
            hx = 1.0 / (1.0 + exp(-yest));
        }
        if (ds->fea_type == BINARY){
            for (j = 0; j < len[i]; j++){
                g[ids[clen[i] + j]] += (hx - y[i]);
            }
        }
        else{
            for (j = 0; j < len[i]; j++){
                g[ids[clen[i] + j]] += (hx - y[i]) * vals[clen[i] + j];
            }
        }
    }
    if (regr->reg_p.r == 2){    // for l2 norm
        for (i = 0; i < col; i++){
            g[i] += regr->reg_p.gamma * (x[i] + x[i]);
        }
    }
    else if (regr->reg_p.r == 1){ // for l1 norm
        for (i = 0; i < col; i++){
            if (x[i] > 0.0){
                g[i] += regr->reg_p.gamma;
            }
            else if (x[i] < 0.0){
                g[i] -= regr->reg_p.gamma;
            }
            else{
                if (g[i] > regr->reg_p.gamma){
                    g[i] -= regr->reg_p.gamma;
                }
                else if (g[i] < -regr->reg_p.gamma){
                    g[i] += regr->reg_p.gamma;
                }
                else{
                    g[i] = 0.0;
                }
            }
        }
    }
}

double lr_eval(double *x, void *_regr) {
    REGR * regr = (REGR*)_regr;
    DATA * ds   = regr->train_ds;
    int i = 0, j = 0, row = ds->row, col = ds->col;
    double loss = 0.0, yest = 0.0, add = 0.0, regloss = 0.0;
    double *y    = ds->y;
    double *vals = ds->vals;
    unsigned int * ids = ds->ids;
    unsigned int * len = ds->len;
    unsigned int * clen = ds->clen;
    for (i = 0; i < row; i++) {
        yest = 0.0;
        if (ds->fea_type == BINARY){
            for (j = 0; j < len[i]; j++){
                yest += x[ids[clen[i] + j]];
            }
        }
        else {
            for (j = 0; j < len[i]; j++){
                yest += x[ids[clen[i] + j]] * vals[clen[i] + j];
            }
        }
        if (yest > 30.0){
            add = yest;
        }
        else if (yest > -30.0){
            add = log(1 + exp(yest));
        }
        else{
            add = 0.0;
        }
        if (y[i] > 0){
            add -= yest;
        }
        loss += add;
    }
    // add loss from regularization
    regloss = 0.0;
    if (regr->reg_p.r == 2) { // for L2 Norm
        for (i = 0; i < col ; i++){
            regloss += x[i] * x[i];
        }
        loss += regloss * regr->reg_p.gamma;
    }
    else if (regr->reg_p.r == 1){  // for L1 Norm
        for (i = 0; i < col; i++){
            if (x[i] > 0.0){
                regloss += x[i];
            }
            else if (x[i] < 0.0){
                regloss -= x[i];
            }
        }
        loss += regloss * regr->reg_p.gamma;
    }
    return loss;
}

REGR * create_lr_model(){
    REGR * lr = create_model(lr_eval, lr_grad);
    return lr;
}
