"""Bound EGL resources when generating many independently compiled scenes."""

import mujoco
import atexit

CACHES = []
REGISTERED = False


def close_renderers():
    for cache in CACHES:
        for renderer in list(cache.values()):
            renderer.close()
        cache.clear()


def renderer_for(cache, model, size):
    global REGISTERED
    if not any(c is cache for c in CACHES):
        CACHES.append(cache)
    key = (id(model), size)
    if key not in cache:
        while len(cache) >= 1:
            old = next(iter(cache))
            cache.pop(old).close()
        cache[key] = mujoco.Renderer(model, size, size)
        if not REGISTERED:
            # Register after EGL's own termination callback (LIFO at exit).
            atexit.register(close_renderers)
            REGISTERED = True
    return cache[key]
