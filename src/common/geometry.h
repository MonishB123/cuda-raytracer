#ifndef GEOMETRY_H
#define GEOMETRY_H

#include "vec3.h"

enum MaterialType { LAMBERTIAN = 0, METAL = 1, DIELECTRIC = 2 };

struct MaterialData {
    color albedo;
    color emit;       // Added for emissive materials
    float fuzz;       // For Metal
    float ref_idx;    // For Dielectric
    MaterialType type;
};

struct Triangle {
    point3 v0, v1, v2;
    vec3 normal;
    int material_index;
};

struct Sphere {
    point3 center;
    float radius;
    int material_index;
};

struct CameraData {
    point3 pos;
    point3 look_at;
    float fov;
    float aperture;
    float focus_dist;
};

struct LightData {
    point3 pos;
    color intensity;
    int type; // 0=Point
};

#endif
