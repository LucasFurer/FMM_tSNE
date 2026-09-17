#pragma once

#include <FMM_tSNE/cameras/camera.h>
#include <FMM_tSNE/opengl_interaction/buffer.h>
#include <FMM_tSNE/opengl_interaction/shader.h>
#include <FMM_tSNE/opengl_interaction/texture.h>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <string>

struct Renderable
{
    GLenum renderType;
    glm::mat4 model;
    Buffer* buffer;
    Shader* shader;
    Texture* texture;

    Renderable(GLenum initRenderType,
               glm::mat4 initModel,
               Buffer* initbuffer,
               Shader* initShader,
               Texture* initTexture)
    {
        renderType = initRenderType;
        model = initModel;
        buffer = initbuffer;
        shader = initShader;
        texture = initTexture;
    }
};

class Scene
{
  public:
    std::string sceneName{};
    Camera* camera{};
    std::vector<Renderable> renderables{};

    Scene() = default;

    Scene(std::string initSceneName, Camera* initCamera, std::vector<Renderable> initRenderables);

    ~Scene() = default;

    void Render();

  private:
};
