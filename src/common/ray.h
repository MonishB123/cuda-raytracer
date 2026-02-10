#ifndef RAY_H
#define RAY_H

#include "vec3.h"

class ray {
public:
    HOST_DEVICE ray() {}
    HOST_DEVICE ray(const point3& origin, const vec3& direction)
        : orig(origin), dir(direction) {}

    HOST_DEVICE point3 origin() const { return orig; }
    HOST_DEVICE vec3 direction() const { return dir; }

    HOST_DEVICE point3 at(float t) const {
        return orig + t*dir;
    }

public:
    point3 orig;
    vec3 dir;
};

#endif
