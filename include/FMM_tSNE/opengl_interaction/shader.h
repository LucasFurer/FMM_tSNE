#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <string>
// #include <glm/gtc/type_ptr.hpp>

class Shader
{
  public:
    // the program ID
    unsigned int ID = 0;

    Shader() = default;

    // constructor reads and builds the shader
    Shader(const char* vertexPath, const char* fragmentPath); // change const char* to std::string

    ~Shader();

    // use/activate the shader
    void use();

    // utility uniform functions
    void setBool(const std::string& name, bool value) const;
    void setInt(const std::string& name, int value) const;
    void setFloat(const std::string& name, float value) const;
    void setMat4(const std::string& name, glm::mat4 value) const;
    void setVec3(const std::string& name, float x, float y, float z) const;
    void setVec3(const std::string& name, glm::vec3 value) const;

  private:
    // utility function for checking shader compilation/linking errors.
    // ------------------------------------------------------------------------
    void checkCompileErrors(unsigned int shader, std::string type);
};
