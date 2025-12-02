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

    // Step decay params
    int step_size; // decay every N epochs
    float gamma;   // decay factor (e.g., 0.1)

    // Exponential decay
    float decay_rate;

    // Cosine annealing
    float min_lr;
    int T_max; // total epochs

    int current_epoch;

} Scheduler;

Scheduler *scheduler_create(SchedulerType type, float initial_lr);
void scheduler_free(Scheduler *s);

float scheduler_get_lr(Scheduler *s);
void scheduler_step(Scheduler *s); // Call after each epoch

// Config
void scheduler_set_step_decay(Scheduler *s, int step_size, float gamma);
void scheduler_set_exponential(Scheduler *s, float decay_rate);
void scheduler_set_cosine(Scheduler *s, float min_lr, int T_max);

#endif /* ifndef SCHEDULER_H */
