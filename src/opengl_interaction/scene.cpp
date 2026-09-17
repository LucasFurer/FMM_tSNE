#include <FMM_tSNE/opengl_interaction/scene.h>

#include <FMM_tSNE/cameras/camera.h>
#include <FMM_tSNE/opengl_interaction/buffer.h>
#include <FMM_tSNE/opengl_interaction/shader.h>
#include <FMM_tSNE/opengl_interaction/texture.h>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <iostream>
#include <string>

Scene::Scene(std::string initSceneName, Camera* initCamera, std::vector<Renderable> initRenderables)
{
    sceneName = initSceneName;
    camera = initCamera;
    renderables = initRenderables;
}

void Scene::Render()
{
    for (int i = 0; i < renderables.size(); i++)
    {
        renderables[i].shader->use();

        switch (renderables[i].renderType)
        {
        case GL_POINTS:
            renderables[i].shader->setMat4("view", camera->getViewMatrix());
            renderables[i].shader->setMat4("projection", camera->getProjectionMatrix());
            renderables[i].shader->setMat4("model", renderables[i].model);

            renderables[i].buffer->bind_VAO();

            glDrawArrays(GL_POINTS, 0, renderables[i].buffer->get_elem_count_());
            break;
        case GL_LINES:
            renderables[i].shader->setMat4("view", camera->getViewMatrix());
            renderables[i].shader->setMat4("projection", camera->getProjectionMatrix());
            renderables[i].shader->setMat4("model", renderables[i].model);

            renderables[i].buffer->bind_VAO();

            glDrawArrays(GL_LINES, 0, renderables[i].buffer->get_elem_count_());
            break;
        case GL_TRIANGLES:
            renderables[i].shader->setInt("planeTexture", 0);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, renderables[i].texture->TEX);

            // renderables[i].shader->use();

            renderables[i].buffer->bind_VAO();

            glDrawElements(GL_TRIANGLES, renderables[i].buffer->get_elem_count_(), GL_UNSIGNED_INT, 0);
            break;
        default:
            std::cout << "invalid render type" << std::endl;
        }
    }
}
