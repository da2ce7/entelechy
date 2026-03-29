/* cpu_threads.h — Persistent thread pool for CPU kernel dispatch (CPU_BACKEND.md).
 *
 * Provides pool_create / pool_destroy / pool_dispatch_and_wait.
 * Workers use atomic task claiming for natural load balancing.
 */
#ifndef CPU_THREADS_H
#define CPU_THREADS_H

#include <stdint.h>

typedef uint32_t uint;
typedef struct ThreadPool ThreadPool;

/* Create a persistent thread pool with num_threads worker threads. */
ThreadPool* pool_create(uint num_threads);

/* Destroy the thread pool, joining all worker threads. */
void pool_destroy(ThreadPool* pool);

/* Submit task_count independent tasks and block until all complete.
 * fn(args, task_index, thread_id) is called for task_index in [0, task_count). */
void pool_dispatch_and_wait(ThreadPool* pool,
                            void (*fn)(void*, uint, uint),
                            void* args,
                            uint task_count);

#endif /* CPU_THREADS_H */
