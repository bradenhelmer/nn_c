/*
 * scheduler.h
 *
 * Scheduler declarations.
 */

#ifndef SCHEDULER_H
#define SCHEDULER_H

typedef enum {
    SCHEDULER_CONSTANT,
    SCHEDULER_STEP_DECAY,
    SCHEDULER_EXPONENTIAL,
    SCHEDULER_COSINE_ANNEALING,
} SchedulerType;

typedef struct {
    SchedulerType type;
    float initial_lr;
    float current_lr;
    int current_epoch;

    // Step decay params
    int step_size; // decay every N epochs
    float gamma;   // decay factor (e.g., 0.1)

    // Exponential decay
    float decay_rate;

    // Cosine annealing
    float min_lr;
    int T_max; // total epochs
} Scheduler;

Scheduler *scheduler_create_constant(float initial_lr);
Scheduler *scheduler_create_step(float initial_lr, int step_size, float gamma);
Scheduler *scheduler_create_exponential(float initial_lr, float decay_rate);
Scheduler *scheduler_create_cosine(float initial_lr, float min_lr, int T_max);
void scheduler_free(Scheduler *s);

void scheduler_step(Scheduler *s); // Call after each epoch
float scheduler_get_lr(Scheduler *s);

#endif /* ifndef SCHEDULER_H */
