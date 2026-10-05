#pragma once

#include <glad/glad.h>

class Texture
{
  public:
    unsigned int TEX{};
    int width{};
    int height{};
    int nrChannels{};
    unsigned char* data{};

    Texture() = default;

    Texture(const char* fileName);

    Texture(int initWidth, int initHeight);

    ~Texture();

    void updateData();

  private:
};
