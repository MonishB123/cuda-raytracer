#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <json/value.h>
#include <json/reader.h>
#include <array>
#include <algorithm>

struct Camera{
    std::array<int, 3> pos;
    std::array<int, 3> look_at;
    int fov;
    Camera(const Json::Value& scene){
        std::transform(scene["camera"]["position"].begin(), scene["camera"]["position"].end(), pos.begin(), [](const auto& x){
            return x.asInt();
        });
        std::transform(scene["camera"]["look_at"].begin(), scene["camera"]["look_at"].end(), look_at.begin(), [](const auto& x){
            return x.asInt();
        });
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
        std::transform(scene["material"]["diffuse"].begin(), scene["material"]["diffuse"].end(), diffuse.begin(), [](const auto& x){
            return x.asFloat();
        });
        std::transform(scene["material"]["look_at"].begin(), scene["material"]["look_at"].end(), specular.begin(), [](const auto& x){
            return x.asFloat();
        });
        shininess = scene["material"]["shininess"].asInt();        
    }
};

struct Light{
    std::array<int, 3> pos;
    std::array<float, 3> intense;
    std::string type;
    Light(const Json::Value& scene){
        std::transform(scene["lights"]["position"].begin(), scene["lights"]["position"].end(), pos.begin(), [](const auto& x){
            return x.asInt();
        });
        std::transform(scene["lights"]["intensity"].begin(), scene["lights"]["intensity"].end(), intense.begin(), [](const auto& x){
            return x.asFloat();
        });
        type = scene["lights"]["type"].asString();        
    }
};

int main(int argv, char* argc[]){
    std::string scence_source_file = std::string(argc[1]);
    int scene_width = std::stoi(argc[2]);
    int scence_height = std::stoi(argc[3]);
    int num_samples = std::stoi(argc[4]);

    //Read json file
    std::ifstream objFile(scence_source_file, std::ifstream::binary);
    Json::Value Scene;
    Json::Reader reader;
    reader.parse(objFile, Scene);

    //load scene context
    Camera sceneCamera(Scene);
    Object sceneObject(Scene);
    Material sceneMaterial(Scene);
    Light sceneLight(Scene);

    
    std::cout << scene_width << " " << scence_height << " " << num_samples << std::endl;
    std::cout << sceneLight.type << std::endl;



    return 0;
}