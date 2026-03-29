/* cpu_threads.c — Thread pool implementation (CPU_BACKEND.md, ADR-015).
 *
 * C11 threads preferred; POSIX pthreads fallback.  Workers use atomic
 * task claiming for natural load balancing.
 */
#include "cpu_threads.h"

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

/* ----------------------------------------------------------------
 * Threading API selection
 * ---------------------------------------------------------------- */
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L && \
    !defined(__STDC_NO_THREADS__) && !defined(__APPLE__)
    #include <threads.h>
    #define THREADS_USE_C11
#else
    #include <pthread.h>
    #define THREADS_USE_POSIX
    /* Thin compatibility wrappers mapping C11 API to pthreads */
    typedef pthread_t       thrd_t;
    typedef pthread_mutex_t mtx_t;
    typedef pthread_cond_t  cnd_t;

    static inline int thrd_create(thrd_t *thr, int (*func)(void*), void *arg) {
        return pthread_create(thr, NULL, (void*(*)(void*))func, arg) == 0 ? 0 : -1;
    }
    static inline int thrd_join(thrd_t thr, int *res) {
        void *retval;
        int rc = pthread_join(thr, &retval);
        if (res) *res = (int)(intptr_t)retval;
        return rc == 0 ? 0 : -1;
    }
    static inline int mtx_init(mtx_t *m, int type) {
        (void)type;
        return pthread_mutex_init(m, NULL) == 0 ? 0 : -1;
    }
    static inline int mtx_lock(mtx_t *m) {
        return pthread_mutex_lock(m) == 0 ? 0 : -1;
    }
    static inline int mtx_unlock(mtx_t *m) {
        return pthread_mutex_unlock(m) == 0 ? 0 : -1;
    }
    static inline void mtx_destroy(mtx_t *m) { pthread_mutex_destroy(m); }
    static inline int cnd_init(cnd_t *c) {
        return pthread_cond_init(c, NULL) == 0 ? 0 : -1;
    }
    static inline int cnd_wait(cnd_t *c, mtx_t *m) {
        return pthread_cond_wait(c, m) == 0 ? 0 : -1;
    }
    static inline int cnd_signal(cnd_t *c) {
        return pthread_cond_signal(c) == 0 ? 0 : -1;
    }
    static inline int cnd_broadcast(cnd_t *c) {
        return pthread_cond_broadcast(c) == 0 ? 0 : -1;
    }
    static inline void cnd_destroy(cnd_t *c) { pthread_cond_destroy(c); }
    enum { mtx_plain = 0 };
#endif

/* ----------------------------------------------------------------
 * Internal types
 * ---------------------------------------------------------------- */
typedef struct {
    void (*function)(void* args, uint task_index, uint thread_id);
    void*       args;
    uint        task_count;
    atomic_uint next_task;
    atomic_uint tasks_completed;
} TaskBatch;

typedef struct {
    struct ThreadPool* pool;
    uint               thread_id;
} WorkerContext;

struct ThreadPool {
    thrd_t*        threads;
    WorkerContext*  contexts;
    uint           thread_count;
    TaskBatch*     current_batch;
    mtx_t          wake_mutex;
    cnd_t          wake_cond;
    cnd_t          done_cond;
    atomic_int     shutdown;
    atomic_int     active;   /* 1 when a batch is submitted */
};

/* ----------------------------------------------------------------
 * Worker function
 * ---------------------------------------------------------------- */
static int worker_main(void* arg) {
    WorkerContext* ctx = (WorkerContext*)arg;
    ThreadPool*    pool = ctx->pool;
    const uint     tid  = ctx->thread_id;

    for (;;) {
        mtx_lock(&pool->wake_mutex);
        while (!atomic_load(&pool->active) && !atomic_load(&pool->shutdown)) {
            cnd_wait(&pool->wake_cond, &pool->wake_mutex);
        }
        mtx_unlock(&pool->wake_mutex);

        if (atomic_load(&pool->shutdown))
            break;

        TaskBatch* batch = pool->current_batch;
        uint task;
        while ((task = atomic_fetch_add(&batch->next_task, 1)) < batch->task_count) {
            batch->function(batch->args, task, tid);
        }

        /* Last thread to finish signals completion */
        uint prev = atomic_fetch_add(&batch->tasks_completed, 1);
        if (prev + 1 == pool->thread_count) {
            mtx_lock(&pool->wake_mutex);
            atomic_store(&pool->active, 0);
            cnd_signal(&pool->done_cond);
            mtx_unlock(&pool->wake_mutex);
        }
    }
    return 0;
}

/* ----------------------------------------------------------------
 * Public API
 * ---------------------------------------------------------------- */
ThreadPool* pool_create(uint num_threads) {
    if (num_threads == 0) num_threads = 1;

    ThreadPool* pool = (ThreadPool*)calloc(1, sizeof(ThreadPool));
    if (!pool) return NULL;

    pool->thread_count  = num_threads;
    pool->current_batch = NULL;
    atomic_init(&pool->shutdown, 0);
    atomic_init(&pool->active, 0);

    mtx_init(&pool->wake_mutex, mtx_plain);
    cnd_init(&pool->wake_cond);
    cnd_init(&pool->done_cond);

    pool->threads  = (thrd_t*)calloc(num_threads, sizeof(thrd_t));
    pool->contexts = (WorkerContext*)calloc(num_threads, sizeof(WorkerContext));
    if (!pool->threads || !pool->contexts) {
        free(pool->threads);
        free(pool->contexts);
        free(pool);
        return NULL;
    }

    for (uint i = 0; i < num_threads; i++) {
        pool->contexts[i].pool      = pool;
        pool->contexts[i].thread_id = i;
        thrd_create(&pool->threads[i], worker_main, &pool->contexts[i]);
    }
    return pool;
}

void pool_destroy(ThreadPool* pool) {
    if (!pool) return;

    /* Signal shutdown */
    mtx_lock(&pool->wake_mutex);
    atomic_store(&pool->shutdown, 1);
    cnd_broadcast(&pool->wake_cond);
    mtx_unlock(&pool->wake_mutex);

    /* Join all workers */
    for (uint i = 0; i < pool->thread_count; i++) {
        thrd_join(pool->threads[i], NULL);
    }

    mtx_destroy(&pool->wake_mutex);
    cnd_destroy(&pool->wake_cond);
    cnd_destroy(&pool->done_cond);

    free(pool->threads);
    free(pool->contexts);
    free(pool);
}

void pool_dispatch_and_wait(ThreadPool* pool,
                            void (*fn)(void*, uint, uint),
                            void* args,
                            uint task_count) {
    /* Edge case: no tasks */
    if (task_count == 0) return;

    TaskBatch batch;
    batch.function = fn;
    batch.args     = args;
    batch.task_count = task_count;
    atomic_init(&batch.next_task, 0);
    atomic_init(&batch.tasks_completed, 0);

    mtx_lock(&pool->wake_mutex);
    pool->current_batch = &batch;
    atomic_store(&pool->active, 1);
    cnd_broadcast(&pool->wake_cond);

    /* Wait until all workers have finished this batch */
    while (atomic_load(&pool->active)) {
        cnd_wait(&pool->done_cond, &pool->wake_mutex);
    }
    pool->current_batch = NULL;
    mtx_unlock(&pool->wake_mutex);
}
