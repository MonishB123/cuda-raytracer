#ifndef RAYTRACER_H
#define RAYTRACER_H

#include "../common/vec3.h"
#include "../common/geometry.h"

// Launch the raytracing kernel
// Now takes scene data
void launch_render_kernel(vec3* framebuffer, int width, int height,
                          Triangle* triangles, int num_triangles,
                          MaterialData* materials, int num_materials,
                          LightData* lights, int num_lights,
                          CameraData cam);

#endif
