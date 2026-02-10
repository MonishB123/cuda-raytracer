#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstdio>
#include <cmath>

#include <json/value.h>
#include <json/reader.h>

#include <cuda_runtime.h>
#include "common/vec3.h"
#include "common/geometry.h"
#include "cuda/raytracer.h"

// --- Pre-existing structs for JSON parsing ---
struct Camera{
    std::array<int, 3> pos;
    std::array<int, 3> look_at;
    int fov;
    Camera(const Json::Value& scene){
        std::transform(scene["camera"]["position"].begin(), scene["camera"]["position"].end(), pos.begin(), [](const auto& x){ return x.asInt(); });
        std::transform(scene["camera"]["look_at"].begin(), scene["camera"]["look_at"].end(), look_at.begin(), [](const auto& x){ return x.asInt(); });
        fov = scene["camera"]["fov"].asInt();
    }
};

struct Object{
    std::string type;
    std::string file;
    std::string material;
    Object(const Json::Value& scene){
        type = scene["objects"]["type"].asString();
        file = scene["objects"]["file"].asString();
        material = scene["objects"]["material"].asString();
    }
};

struct Material{
    std::array<float, 3> diffuse;
    std::array<float, 3> specular;
    int shininess;
    Material(const Json::Value& scene){
        std::transform(scene["material"]["diffuse"].begin(), scene["material"]["diffuse"].end(), diffuse.begin(), [](const auto& x){ return x.asFloat(); });
        if (scene["material"].isMember("specular")) {
             std::transform(scene["material"]["specular"].begin(), scene["material"]["specular"].end(), specular.begin(), [](const auto& x){ return x.asFloat(); });
        } else { specular = {0,0,0}; } // default black specular if missing
        shininess = scene["material"]["shininess"].asInt();        
    }
};

struct Light{
    std::array<int, 3> pos;
    std::array<float, 3> intense;
    std::string type;
    Light(const Json::Value& scene){
        std::transform(scene["lights"]["position"].begin(), scene["lights"]["position"].end(), pos.begin(), [](const auto& x){ return x.asInt(); });
        std::transform(scene["lights"]["intensity"].begin(), scene["lights"]["intensity"].end(), intense.begin(), [](const auto& x){ return x.asFloat(); });
        type = scene["lights"]["type"].asString();        
    }
};

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
        int r = int(255.99 * sqrt(fmax(0.0f, pixel.x()))); // Gamma correction
        int g = int(255.99 * sqrt(fmax(0.0f, pixel.y())));
        int b = int(255.99 * sqrt(fmax(0.0f, pixel.z())));
        file << r << " " << g << " " << b << "\n";
    }
    file.close();
    std::cout << "Saved " << filename << std::endl;
}

int main(int argc, char* argv[]){
    if (argc < 5) return 1;

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

    Camera sceneCamera(Scene);
    Object sceneObject(Scene);
    Material sceneMaterial(Scene);
    Light sceneLight(Scene);

    // Prepare GPU Data Structures
    std::vector<Triangle> triangles;
    std::vector<MaterialData> materials;
    std::vector<LightData> lights;

    // Convert Material
    MaterialData mat;
    mat.diffuse = vec3(sceneMaterial.diffuse[0], sceneMaterial.diffuse[1], sceneMaterial.diffuse[2]);
    mat.specular = vec3(sceneMaterial.specular[0], sceneMaterial.specular[1], sceneMaterial.specular[2]);
    mat.shininess = (float)sceneMaterial.shininess;
    mat.type = 0;
    materials.push_back(mat);

    // Convert Light (Single light support currently)
    LightData lit;
    lit.pos = point3((float)sceneLight.pos[0], (float)sceneLight.pos[1], (float)sceneLight.pos[2]);
    lit.intensity = color(sceneLight.intense[0], sceneLight.intense[1], sceneLight.intense[2]);
    lit.type = 0;
    lights.push_back(lit);

    // Parse OBJ
    std::string obj_filename = "assets/objects/" + sceneObject.file + "/" + sceneObject.file + ".obj";
    std::ifstream objFile(obj_filename);
    if (!objFile.is_open()) { std::cerr << "No OBJ file: " << obj_filename << "\n"; return 1; }

    std::vector<vec3> positions;
    std::string line;
    while (std::getline(objFile, line)){
        std::stringstream ss(line);
        std::string type;
        ss >> type;
        if (type == "v") {
            float x, y, z;
            ss >> x >> y >> z;
            positions.push_back(vec3(x, y, z));
        } else if (type == "f") {
            std::vector<int> v_indices;
            std::string token;
            while (ss >> token) {
                size_t slash = token.find('/');
                std::string v_str = (slash != std::string::npos) ? token.substr(0, slash) : token;
                v_indices.push_back(std::stoi(v_str));
            }
            
            // Triangulate fan
            if (v_indices.size() >= 3) {
                for (size_t i = 1; i < v_indices.size() - 1; ++i) {
                    Triangle t;
                    // OBJ is 1-based
                    int idx0 = v_indices[0] - 1;
                    int idx1 = v_indices[i] - 1;
                    int idx2 = v_indices[i+1] - 1;
                    
                    if(idx0 < 0 || idx0 >= positions.size() || 
                       idx1 < 0 || idx1 >= positions.size() || 
                       idx2 < 0 || idx2 >= positions.size()) continue;

                    t.v0 = positions[idx0];
                    t.v1 = positions[idx1];
                    t.v2 = positions[idx2];
                    t.material_index = 0; // Usage of first material
                    t.normal = unit_vector(cross(t.v1 - t.v0, t.v2 - t.v0));
                    triangles.push_back(t);
                }
            }
        }
    }
    
    std::cout << "Loaded " << triangles.size() << " triangles.\n";

    // --- CUDA Execution ---
    size_t num_pixels = scene_width * scene_height;
    std::vector<vec3> h_framebuffer(num_pixels);
    vec3* d_framebuffer;
    
    Triangle* d_triangles;
    MaterialData* d_materials;
    LightData* d_lights;
    
    checkCudaErrors(cudaMalloc(&d_framebuffer, num_pixels * sizeof(vec3)));
    checkCudaErrors(cudaMalloc(&d_triangles, triangles.size() * sizeof(Triangle)));
    checkCudaErrors(cudaMalloc(&d_materials, materials.size() * sizeof(MaterialData)));
    checkCudaErrors(cudaMalloc(&d_lights, lights.size() * sizeof(LightData)));

    checkCudaErrors(cudaMemcpy(d_triangles, triangles.data(), triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_materials, materials.data(), materials.size() * sizeof(MaterialData), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_lights, lights.data(), lights.size() * sizeof(LightData), cudaMemcpyHostToDevice));

    CameraData camData;
    camData.pos = point3((float)sceneCamera.pos[0], (float)sceneCamera.pos[1], (float)sceneCamera.pos[2]);
    camData.look_at = point3((float)sceneCamera.look_at[0], (float)sceneCamera.look_at[1], (float)sceneCamera.look_at[2]);
    camData.fov = (float)sceneCamera.fov;

    std::cout << "Launching Kernel...\n";
    launch_render_kernel(d_framebuffer, scene_width, scene_height,
                         d_triangles, (int)triangles.size(),
                         d_materials, (int)materials.size(),
                         d_lights, (int)lights.size(),
                         camData);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(h_framebuffer.data(), d_framebuffer, num_pixels * sizeof(vec3), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_framebuffer));
    checkCudaErrors(cudaFree(d_triangles));
    checkCudaErrors(cudaFree(d_materials));
    checkCudaErrors(cudaFree(d_lights));

    save_ppm("output.ppm", h_framebuffer, scene_width, scene_height);
    return 0;
}