#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <cfloat>
#include "../common/vec3.h"
#include "../common/ray.h"
#include "../common/geometry.h"

#define MAX_DEPTH 10
#define EPSILON 0.001f

__device__ float schlick(float cosine, float ref_idx) {
    float r0 = (1-ref_idx) / (1+ref_idx);
    r0 = r0*r0;
    return r0 + (1-r0)*pow((1 - cosine), 5);
}

__device__ bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) {
    vec3 uv = unit_vector(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt*ni_over_nt*(1-dt*dt);
    if (discriminant > 0) {
        refracted = ni_over_nt*(uv - n*dt) - n*sqrt(discriminant);
        return true;
    }
    else
        return false;
}

__device__ vec3 random_in_unit_sphere(curandState* local_rand_state) {
    vec3 p;
    do {
        p = 2.0f * vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state)) - vec3(1,1,1);
    } while (p.length_squared() >= 1.0f);
    return p;
}

__device__ bool hit_triangle(const ray& r, const Triangle& tri, float t_min, float t_max, float& t, float& u, float& v) {
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
    return (t < t_max && t > t_min);
}

__device__ bool hit_sphere(const ray& r, const Sphere& s, float t_min, float t_max, float& t) {
    vec3 oc = r.origin() - s.center;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - s.radius*s.radius;
    float discriminant = b*b - a*c;
    if (discriminant > 0) {
        float temp = (-b - sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            t = temp;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            t = temp;
            return true;
        }
    }
    return false;
}

struct hit_record {
    float t;
    point3 p;
    vec3 normal;
    int mat_idx;
};

__device__ bool world_hit(const ray& r, float t_min, float t_max, hit_record& rec,
                          Triangle* triangles, int num_triangles,
                          Sphere* spheres, int num_spheres) {
    bool hit_anything = false;
    float closest_so_far = t_max;

    for (int i = 0; i < num_triangles; i++) {
        float t, u, v;
        if (hit_triangle(r, triangles[i], t_min, closest_so_far, t, u, v)) {
            hit_anything = true;
            closest_so_far = t;
            rec.t = t;
            rec.p = r.at(t);
            rec.normal = triangles[i].normal;
            rec.mat_idx = triangles[i].material_index;
        }
    }

    for (int i = 0; i < num_spheres; i++) {
        float t;
        if (hit_sphere(r, spheres[i], t_min, closest_so_far, t)) {
            hit_anything = true;
            closest_so_far = t;
            rec.t = t;
            rec.p = r.at(t);
            rec.normal = (rec.p - spheres[i].center) / spheres[i].radius;
            rec.mat_idx = spheres[i].material_index;
        }
    }

    return hit_anything;
}

__device__ color ray_color(ray& r, Triangle* triangles, int num_triangles,
                           Sphere* spheres, int num_spheres,
                           MaterialData* materials, int num_materials,
                           LightData* lights, int num_lights,
                           curandState* local_rand_state) {
    color cur_attenuation(1, 1, 1);
    color total_emit(0, 0, 0);
    ray cur_ray = r;
    for (int i = 0; i < MAX_DEPTH; i++) {
        hit_record rec;
        if (world_hit(cur_ray, EPSILON, FLT_MAX, rec, triangles, num_triangles, spheres, num_spheres)) {
            ray scattered;
            color attenuation;
            MaterialData mat = materials[rec.mat_idx];

            total_emit += cur_attenuation * mat.emit;

            if (mat.type == LAMBERTIAN) {
                vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
                scattered = ray(rec.p, target - rec.p);
                attenuation = mat.albedo;
                cur_attenuation = cur_attenuation * attenuation;
                cur_ray = scattered;
            }
            else if (mat.type == METAL) {
                vec3 reflected = reflect(unit_vector(cur_ray.direction()), rec.normal);
                scattered = ray(rec.p, reflected + mat.fuzz * random_in_unit_sphere(local_rand_state));
                attenuation = mat.albedo;
                if (dot(scattered.direction(), rec.normal) > 0) {
                    cur_attenuation = cur_attenuation * attenuation;
                    cur_ray = scattered;
                }
                else return total_emit;
            }
            else if (mat.type == DIELECTRIC) {
                vec3 outward_normal;
                vec3 reflected = reflect(cur_ray.direction(), rec.normal);
                float ni_over_nt;
                attenuation = color(1.0, 1.0, 1.0);
                vec3 refracted;
                float reflect_prob;
                float cosine;
                if (dot(cur_ray.direction(), rec.normal) > 0) {
                    outward_normal = -rec.normal;
                    ni_over_nt = mat.ref_idx;
                    cosine = mat.ref_idx * dot(cur_ray.direction(), rec.normal) / cur_ray.direction().length();
                }
                else {
                    outward_normal = rec.normal;
                    ni_over_nt = 1.0f / mat.ref_idx;
                    cosine = -dot(cur_ray.direction(), rec.normal) / cur_ray.direction().length();
                }
                if (refract(cur_ray.direction(), outward_normal, ni_over_nt, refracted)) {
                    reflect_prob = schlick(cosine, mat.ref_idx);
                }
                else {
                    reflect_prob = 1.0f;
                }
                if (curand_uniform(local_rand_state) < reflect_prob) {
                    scattered = ray(rec.p, reflected);
                }
                else {
                    scattered = ray(rec.p, refracted);
                }
                cur_attenuation = cur_attenuation * attenuation;
                cur_ray = scattered;
            }
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            color sky_color = (1.0f - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
            return total_emit + cur_attenuation * sky_color;
        }
    }
    return total_emit; // exceeded recursion depth
}

__global__ void render_kernel(vec3* framebuffer, int width, int height, int samples,
                              Triangle* triangles, int num_triangles,
                              Sphere* spheres, int num_spheres,
                              MaterialData* materials, int num_materials,
                              LightData* lights, int num_lights,
                              CameraData cam) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= width || j >= height) return;

    int pixel_index = j * width + i;
    curandState local_rand_state;
    curand_init(1984 + pixel_index, 0, 0, &local_rand_state);

    // Camera setup
    float aspect_ratio = float(width) / float(height);
    float theta = cam.fov * 3.14159f / 180.0f;
    float h = tan(theta/2);
    float viewport_height = 2.0f * h;
    float viewport_width = aspect_ratio * viewport_height;

    vec3 w = unit_vector(cam.pos - cam.look_at);
    vec3 u = unit_vector(cross(vec3(0,1,0), w));
    vec3 v = cross(w, u);

    vec3 origin = cam.pos;
    vec3 horizontal = viewport_width * u;
    vec3 vertical = viewport_height * v;
    vec3 lower_left_corner = origin - horizontal/2 - vertical/2 - w;

    color pixel_color(0, 0, 0);
    for (int s = 0; s < samples; s++) {
        float u_coord = float(i + curand_uniform(&local_rand_state)) / (width - 1);
        float v_coord = float(j + curand_uniform(&local_rand_state)) / (height - 1);
        ray r(origin, lower_left_corner + u_coord*horizontal + v_coord*vertical - origin);
        pixel_color += ray_color(r, triangles, num_triangles, spheres, num_spheres, materials, num_materials, lights, num_lights, &local_rand_state);
    }

    framebuffer[pixel_index] = pixel_color / float(samples);
}

void launch_render_kernel(vec3* d_framebuffer, int width, int height, int samples,
                          Triangle* d_triangles, int num_triangles,
                          Sphere* d_spheres, int num_spheres,
                          MaterialData* d_materials, int num_materials,
                          LightData* d_lights, int num_lights,
                          CameraData cam) 
{
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    render_kernel<<<numBlocks, threadsPerBlock>>>(
        d_framebuffer, width, height, samples,
        d_triangles, num_triangles,
        d_spheres, num_spheres,
        d_materials, num_materials,
        d_lights, num_lights,
        cam
    );
}
