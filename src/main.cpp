#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <json/value.h>
#include <json/reader.h>
#include <array>
#include <algorithm>
#include <cstdio>

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
    std::ifstream sceneFile(scence_source_file, std::ifstream::binary);
    Json::Value Scene;
    Json::Reader reader;
    reader.parse(sceneFile, Scene);

    //load scene context
    Camera sceneCamera(Scene);
    Object sceneObject(Scene);
    Material sceneMaterial(Scene);
    Light sceneLight(Scene);

    //read .obj file
    std::vector<std::array<float, 3>> vertices;
    std::vector<std::array<float, 3>> normals;
    std::vector<std::array<float, 3>> texcoords;
    std::vector<std::array<std::array<int, 3>, 4>> faces;
    
    std::string filename = "assets/objects/" + sceneObject.file + "/" + sceneObject.file + ".obj";
    std::ifstream objFile(filename);
        if (!objFile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return 1;
    }

    std::string line;

    //read first 9 filler lines
    for(int i = 0; i < 9; i++){
        std::getline(objFile, line);
    }

    while (std::getline(objFile, line)){
        std::stringstream ss(line);
        std::string type;
        ss >> type;

        //check data type
        if (type == "v"){
            std::array<float, 3> vertice;
            for (int i = 0; i < 3; i++){
                std::string value;
                ss >> value;
                vertice[i] = std::stof(value);
            }
            vertices.push_back(vertice);
        }
        else if (type == "vn"){
            std::array<float, 3> normal;
            for (int i = 0; i < 3; i++){
                std::string value;
                ss >> value;
                normal[i] = std::stof(value);
            }
            normals.push_back(normal);
        }
        else if (type == "vt"){
            std::array<float, 3> texcoord;
            for (int i = 0; i < 3; i++){
                std::string value;
                ss >> value;
                texcoord[i] = std::stof(value);
            }
            texcoords.push_back(texcoord);
        }
        else if (type == "f"){
            std::array<std::array<int, 3>, 4> quad;  
            for (int i = 0; i < 4; i++){
                std::string strVertice;
                ss >> strVertice;

                std::array<int, 3> completeVertice;
                for (int j = 0; j < 3; j++){
                    std::stringstream delimiterstream(strVertice);
                    std::string individualVertice;
                    std::getline(delimiterstream, individualVertice, '/');
                    completeVertice[j] = std::stoi(individualVertice);
                }
                quad[i] = completeVertice;
            }
            faces.push_back(quad);
        }
    }
    objFile.close();

    std::cout << faces.at(100).at(2).at(1) << std::endl;
    
    return 0;
}