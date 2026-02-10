#include <cuda_runtime.h>
#include <iostream>
#include <cfloat>
#include "../common/vec3.h"
#include "../common/ray.h"
#include "../common/geometry.h"

#define MAX_DEPTH 10000.0f
#define EPSILON 0.0001f

__device__ bool hit_triangle(const ray& r, const Triangle& tri, float& t, float& u, float& v) {
    vec3 v0v1 = tri.v1 - tri.v0;
    vec3 v0v2 = tri.v2 - tri.v0;
    vec3 pvec = cross(r.direction(), v0v2);
    float det = dot(v0v1, pvec);

    if (det < EPSILON && det > -EPSILON) return false;
    float invDet = 1.0f / det;

    vec3 tvec = r.origin() - tri.v0;
    u = dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return false;

    vec3 qvec = cross(tvec, v0v1);
    v = dot(r.direction(), qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return false;

    t = dot(v0v2, qvec) * invDet;
    if (t < EPSILON) return false;

    return true;
}

__global__ void render_kernel(vec3* framebuffer, int width, int height,
                              Triangle* triangles, int num_triangles,
                              MaterialData* materials, int num_materials,
                              LightData* lights, int num_lights,
                              CameraData cam) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= width || j >= height) return;

    // Camera setup
    float aspect_ratio = float(width) / float(height);
    float theta = cam.fov * 3.14159f / 180.0f;
    float h = tan(theta/2);
    float viewport_height = 2.0f * h;
    float viewport_width = aspect_ratio * viewport_height;

    // Assume camera looks at -Z by default if no look_at logic (or use look_at)
    // Simple look_at implementation:
    vec3 w = unit_vector(cam.pos - cam.look_at);
    vec3 u = unit_vector(cross(vec3(0,1,0), w));
    vec3 v = cross(w, u);

    vec3 origin = cam.pos;
    vec3 horizontal = viewport_width * u;
    vec3 vertical = viewport_height * v;
    vec3 lower_left_corner = origin - horizontal/2 - vertical/2 - w;

    // Ray Generation
    // Note: j=0 is usually top or bottom. 
    // If output.ppm writes rows from top, let's say j=0 is TOP.
    // Standard Math: y increases up. Image coordinates: y increases down (or scanlines).
    // Let's assume standard UV layout (0,0 bottom left).
    float s = float(i) / (width - 1);
    float t_coord = float(j) / (height - 1); // Flip this if image is upside down

    ray r(origin, lower_left_corner + s*horizontal + t_coord*vertical - origin);
    
    // Intersection
    float closest_t = MAX_DEPTH;
    int hit_idx = -1;
    float hit_u, hit_v;

    // Naive loop over all triangles
    // Optimization: BVH would be here
    for (int k = 0; k < num_triangles; ++k) {
        float t_val, u_val, v_val;
        if (hit_triangle(r, triangles[k], t_val, u_val, v_val)) {
            if (t_val < closest_t) {
                closest_t = t_val;
                hit_idx = k;
                hit_u = u_val;
                hit_v = v_val;
            }
        }
    }

    // Shading
    color pixel_color(0,0,0);
    if (hit_idx != -1) {
        Triangle tri = triangles[hit_idx];
        MaterialData mat = materials[tri.material_index];
        
        point3 hit_point = r.at(closest_t);
        vec3 normal = unit_vector(tri.normal); // Flat shading
        // OR interpolate vertex normals if available:
        // vec3 normal = unit_vector((1-hit_u-hit_v)*tri.v0_n + hit_u*tri.v1_n + hit_v*tri.v2_n);
        
        // Ambient
        float ambientStrength = 0.1f;
        pixel_color += mat.diffuse * ambientStrength;

        // Lights
        for (int l = 0; l < num_lights; ++l) {
            LightData light = lights[l];
            vec3 lightDir = unit_vector(light.pos - hit_point);
            
            // Diffuse
            float diff = fmaxf(dot(normal, lightDir), 0.0f);
            vec3 diffuse = diff * mat.diffuse * light.intensity;
            
            // Specular (Phong)
            vec3 viewDir = unit_vector(cam.pos - hit_point);
            vec3 reflectDir = reflect(-lightDir, normal);
            float spec = powf(fmaxf(dot(viewDir, reflectDir), 0.0f), mat.shininess);
            vec3 specular = spec * mat.specular * light.intensity;
            
            // Shadow Ray
            bool in_shadow = false;
            ray shadow_ray(hit_point + normal * EPSILON, lightDir);
            float light_dist = (light.pos - hit_point).length();
            
            for (int k = 0; k < num_triangles; ++k) {
                float t_shadow, u_s, v_s;
                if (hit_triangle(shadow_ray, triangles[k], t_shadow, u_s, v_s)) {
                    if (t_shadow < light_dist) {
                        in_shadow = true;
                        break;
                    }
                }
            }

            if (!in_shadow) {
                pixel_color += diffuse + specular;
            }
        }
    }

    int pixel_index = j * width + i;
    framebuffer[pixel_index] = pixel_color;
}

void launch_render_kernel(vec3* d_framebuffer, int width, int height,
                          Triangle* d_triangles, int num_triangles,
                          MaterialData* d_materials, int num_materials,
                          LightData* d_lights, int num_lights,
                          CameraData cam) 
{
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    render_kernel<<<numBlocks, threadsPerBlock>>>(
        d_framebuffer, width, height,
        d_triangles, num_triangles,
        d_materials, num_materials,
        d_lights, num_lights,
        cam
    );
}
