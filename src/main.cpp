#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <array>

#include <json/value.h>
#include <json/reader.h>

#include <cuda_runtime.h>
#include "common/vec3.h"
#include "common/geometry.h"
#include "cuda/raytracer.h"

void checkCudaErrors(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(result) << std::endl;
        exit(-1);
    }
}

void save_ppm(const std::string& filename, const std::vector<vec3>& framebuffer, int width, int height) {
    std::ofstream file(filename);
    if (!file) return;
    file << "P3\n" << width << " " << height << "\n255\n";
    for (size_t i = 0; i < framebuffer.size(); ++i) {
        auto& pixel = framebuffer[i];
        // Gamma correction 2.0
        int r = int(255.99 * sqrt(fmaxf(0.0f, pixel.x())));
        int g = int(255.99 * sqrt(fmaxf(0.0f, pixel.y())));
        int b = int(255.99 * sqrt(fmaxf(0.0f, pixel.z())));
        file << r << " " << g << " " << b << "\n";
    }
    file.close();
    std::cout << "Saved " << filename << std::endl;
}

MaterialData parse_material(const Json::Value& matJson) {
    MaterialData mat;
    mat.albedo = color(matJson["albedo"][0].asFloat(), matJson["albedo"][1].asFloat(), matJson["albedo"][2].asFloat());
    mat.emit = matJson.isMember("emit") ? 
               color(matJson["emit"][0].asFloat(), matJson["emit"][1].asFloat(), matJson["emit"][2].asFloat()) : 
               color(0,0,0);
    std::string type = matJson["type"].asString();
    if (type == "metal") {
        mat.type = METAL;
        mat.fuzz = matJson.get("fuzz", 0.0f).asFloat();
    } else if (type == "dielectric") {
        mat.type = DIELECTRIC;
        mat.ref_idx = matJson.get("ref_idx", 1.5f).asFloat();
    } else {
        mat.type = LAMBERTIAN;
    }
    return mat;
}

int main(int argc, char* argv[]){
    if (argc < 5) {
        std::cout << "Usage: " << argv[0] << " <scene.json> <width> <height> <samples>\n";
        return 1;
    }

    std::string scene_source_file = std::string(argv[1]);
    int scene_width = std::stoi(argv[2]);
    int scene_height = std::stoi(argv[3]);
    int num_samples = std::stoi(argv[4]);

    // Parse JSON
    std::ifstream sceneFile(scene_source_file, std::ifstream::binary);
    if (!sceneFile.is_open()) { std::cerr << "No scene file\n"; return 1; }
    Json::Value Scene;
    Json::Reader reader;
    if (!reader.parse(sceneFile, Scene)) { std::cerr << "Bad JSON\n"; return 1; }

    // Prepare Scene Data
    std::vector<Triangle> triangles;
    std::vector<Sphere> spheres;
    std::vector<MaterialData> materials;
    std::vector<LightData> lights;

    // Materials
    if (Scene.isMember("materials") && Scene["materials"].isArray()) {
        for (const auto& m : Scene["materials"]) {
            materials.push_back(parse_material(m));
        }
    } else if (Scene.isMember("materials")) { // Fallback for old format
         MaterialData mat;
         mat.albedo = color(Scene["materials"]["diffuse"][0].asFloat(), Scene["materials"]["diffuse"][1].asFloat(), Scene["materials"]["diffuse"][2].asFloat());
         mat.type = LAMBERTIAN;
         materials.push_back(mat);
    }
    if (materials.empty()) {
        MaterialData mat;
        mat.albedo = color(0.5, 0.5, 0.5);
        mat.type = LAMBERTIAN;
        materials.push_back(mat);
    }

    // Lights
    if (Scene.isMember("lights") && Scene["lights"].isArray()) {
        for (const auto& l : Scene["lights"]) {
            LightData lit;
            lit.pos = point3(l["position"][0].asFloat(), l["position"][1].asFloat(), l["position"][2].asFloat());
            lit.intensity = color(l["intensity"][0].asFloat(), l["intensity"][1].asFloat(), l["intensity"][2].asFloat());
            lit.type = 0;
            lights.push_back(lit);
        }
    } else if (Scene.isMember("lights")) {
        LightData lit;
        lit.pos = point3(Scene["lights"]["position"][0].asFloat(), Scene["lights"]["position"][1].asFloat(), Scene["lights"]["position"][2].asFloat());
        lit.intensity = color(Scene["lights"]["intensity"][0].asFloat(), Scene["lights"]["intensity"][1].asFloat(), Scene["lights"]["intensity"][2].asFloat());
        lit.type = 0;
        lights.push_back(lit);
    }

    // Objects
    auto parse_object = [&](const Json::Value& obj) {
        std::string type = obj["type"].asString();
        int mat_idx = obj.get("material_index", 0).asInt();
        if (type == "mesh") {
            std::string file = obj["file"].asString();
            std::string obj_filename = "assets/objects/" + file + "/" + file + ".obj";
            std::ifstream objFile(obj_filename);
            if (!objFile.is_open()) { std::cerr << "No OBJ file: " << obj_filename << "\n"; return; }

            std::vector<vec3> positions;
            std::string line;
            while (std::getline(objFile, line)){
                std::stringstream ss(line);
                std::string t;
                ss >> t;
                if (t == "v") {
                    float x, y, z;
                    ss >> x >> y >> z;
                    positions.push_back(vec3(x, y, z));
                } else if (t == "f") {
                    std::vector<int> v_indices;
                    std::string token;
                    while (ss >> token) {
                        size_t slash = token.find('/');
                        std::string v_str = (slash != std::string::npos) ? token.substr(0, slash) : token;
                        v_indices.push_back(std::stoi(v_str));
                    }
                    if (v_indices.size() >= 3) {
                        for (size_t i = 1; i < v_indices.size() - 1; ++i) {
                            Triangle tri;
                            tri.v0 = positions[v_indices[0] - 1];
                            tri.v1 = positions[v_indices[i] - 1];
                            tri.v2 = positions[v_indices[i+1] - 1];
                            tri.material_index = mat_idx;
                            tri.normal = unit_vector(cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
                            triangles.push_back(tri);
                        }
                    }
                }
            }
        } else if (type == "sphere") {
            Sphere s;
            s.center = point3(obj["center"][0].asFloat(), obj["center"][1].asFloat(), obj["center"][2].asFloat());
            s.radius = obj["radius"].asFloat();
            s.material_index = mat_idx;
            spheres.push_back(s);
        }
    };

    if (Scene.isMember("objects") && Scene["objects"].isArray()) {
        for (const auto& obj : Scene["objects"]) parse_object(obj);
    } else if (Scene.isMember("objects")) {
        parse_object(Scene["objects"]);
    }

    // Camera
    CameraData camData;
    camData.pos = point3(Scene["camera"]["position"][0].asFloat(), Scene["camera"]["position"][1].asFloat(), Scene["camera"]["position"][2].asFloat());
    camData.look_at = point3(Scene["camera"]["look_at"][0].asFloat(), Scene["camera"]["look_at"][1].asFloat(), Scene["camera"]["look_at"][2].asFloat());
    camData.fov = Scene["camera"]["fov"].asFloat();
    camData.aperture = Scene["camera"].get("aperture", 0.0f).asFloat();
    camData.focus_dist = Scene["camera"].get("focus_dist", 10.0f).asFloat();

    // CUDA
    size_t num_pixels = scene_width * scene_height;
    std::vector<vec3> h_framebuffer(num_pixels);
    vec3* d_framebuffer;
    Triangle* d_triangles = nullptr;
    Sphere* d_spheres = nullptr;
    MaterialData* d_materials = nullptr;
    LightData* d_lights = nullptr;
    
    checkCudaErrors(cudaMalloc(&d_framebuffer, num_pixels * sizeof(vec3)));
    if (!triangles.empty()) {
        checkCudaErrors(cudaMalloc(&d_triangles, triangles.size() * sizeof(Triangle)));
        checkCudaErrors(cudaMemcpy(d_triangles, triangles.data(), triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice));
    }
    if (!spheres.empty()) {
        checkCudaErrors(cudaMalloc(&d_spheres, spheres.size() * sizeof(Sphere)));
        checkCudaErrors(cudaMemcpy(d_spheres, spheres.data(), spheres.size() * sizeof(Sphere), cudaMemcpyHostToDevice));
    }
    if (!materials.empty()) {
        checkCudaErrors(cudaMalloc(&d_materials, materials.size() * sizeof(MaterialData)));
        checkCudaErrors(cudaMemcpy(d_materials, materials.data(), materials.size() * sizeof(MaterialData), cudaMemcpyHostToDevice));
    }
    if (!lights.empty()) {
        checkCudaErrors(cudaMalloc(&d_lights, lights.size() * sizeof(LightData)));
        checkCudaErrors(cudaMemcpy(d_lights, lights.data(), lights.size() * sizeof(LightData), cudaMemcpyHostToDevice));
    }

    std::cout << "Launching Kernel (" << num_samples << " samples)...\n";
    launch_render_kernel(d_framebuffer, scene_width, scene_height, num_samples,
                         d_triangles, (int)triangles.size(),
                         d_spheres, (int)spheres.size(),
                         d_materials, (int)materials.size(),
                         d_lights, (int)lights.size(),
                         camData);
    
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(h_framebuffer.data(), d_framebuffer, num_pixels * sizeof(vec3), cudaMemcpyDeviceToHost));

    cudaFree(d_framebuffer);
    if (d_triangles) cudaFree(d_triangles);
    if (d_spheres) cudaFree(d_spheres);
    if (d_materials) cudaFree(d_materials);
    if (d_lights) cudaFree(d_lights);

    save_ppm("output.ppm", h_framebuffer, scene_width, scene_height);
    return 0;
}