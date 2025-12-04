/*
 * gradient_descent.h
 *
 * Gradient descent definitions
 */
#ifndef GRADIENT_DESCENT_H
#define GRADIENT_DESCENT_H
#include "../data/dataset.h"
#include "../nn/mlp.h"
#include "../nn/perceptron.h"
#include "optimizer.h"
#include "scheduler.h"

typedef struct {
    int max_epochs;
    float tolerance;
    int batch_size;
    int verbose;

    Optimizer *optimizer;
    Scheduler *scheduler;
    float l2_lambda;
} TrainingConfig;

typedef struct {
    float *loss_history;
    float *accuracy_history;
    int epochs_completed;
    float final_loss;
} TrainingResult;

void training_result_free(TrainingResult *result);

TrainingResult *train_perceptron(Perceptron *p, Dataset *train_data, Dataset *val_data,
                                 TrainingConfig *config);

TrainingResult *train_mlp(MLP *mlp, Dataset *train_data, Dataset *val_data, TrainingConfig *config);

TrainingResult *train_mlp_batch(MLP *mlp, Dataset *train_data, Dataset *val_data,
                                TrainingConfig *config);

#endif /* ifndef GRADIENT_DESCENT_H */
