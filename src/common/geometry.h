#ifndef GEOMETRY_H
#define GEOMETRY_H

#include "vec3.h"

struct Triangle {
    point3 v0, v1, v2;
    vec3 normal; // Geometric normal
    int material_index;
};

struct MaterialData {
    color diffuse;
    color specular;
    float shininess;
    int type; // 0=Lambertian
};

struct CameraData {
    point3 pos;
    point3 look_at;
    float fov;
    // Precomputed values could be added here later (viewport width/height etc)
};

struct LightData {
    point3 pos;
    color intensity;
    int type; // 0=Point
};

#endif
