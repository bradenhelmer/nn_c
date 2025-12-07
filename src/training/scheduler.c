/*
 * scheduler.h
 *
 * Scheduler implementations.
 */

#include "scheduler.h"
#include "../utils/utils.h"
#include <math.h>
#include <stdlib.h>

static Scheduler *scheduler_create_base(SchedulerType type, float initial_lr) {
    Scheduler *s = (Scheduler *)malloc(sizeof(Scheduler));
    s->type = type;
    s->initial_lr = initial_lr;
    s->current_lr = initial_lr;
    s->current_epoch = 0;
    s->step_size = 0;
    s->gamma = 0.0f;
    s->decay_rate = 0.0f;
    s->min_lr = 0.0f;
    s->T_max = 0;
    return s;
}

Scheduler *scheduler_create_constant(float lr) {
    return scheduler_create_base(SCHEDULER_CONSTANT, lr);
}

Scheduler *scheduler_create_step(float initial_lr, int step_size, float gamma) {
    Scheduler *s = scheduler_create_base(SCHEDULER_STEP_DECAY, initial_lr);
    s->step_size = step_size;
    s->gamma = gamma;
    return s;
}

Scheduler *scheduler_create_exponential(float initial_lr, float decay_rate) {
    Scheduler *s = scheduler_create_base(SCHEDULER_EXPONENTIAL, initial_lr);
    s->decay_rate = decay_rate;
    return s;
}

Scheduler *scheduler_create_cosine(float initial_lr, float min_lr, int T_max) {
    Scheduler *s = scheduler_create_base(SCHEDULER_COSINE_ANNEALING, initial_lr);
    s->min_lr = min_lr;
    s->T_max = T_max;
    return s;
}

void scheduler_free(Scheduler *s) {
    free(s);
}

void scheduler_step(Scheduler *s) {
    s->current_epoch++;
    switch (s->type) {
    case SCHEDULER_CONSTANT:
        // No change
        break;
    case SCHEDULER_STEP_DECAY:
        // lr = initial_lr * gamma^(epoch / step_size)
        s->current_lr = s->initial_lr * powf(s->gamma, (s->current_epoch / s->step_size));
        break;
    case SCHEDULER_EXPONENTIAL:
        // lr = initial_lr * e^(-decay_rate * epoch)
        s->current_lr = s->initial_lr * expf(-s->decay_rate * s->current_epoch);
        break;
    case SCHEDULER_COSINE_ANNEALING:
        s->current_lr = s->min_lr + 0.5f * (s->initial_lr - s->min_lr) *
                                        (1.0f + cosf(s->current_epoch * PI / s->T_max));
    }
}

float scheduler_get_lr(Scheduler *s) {
    return s->current_lr;
}
