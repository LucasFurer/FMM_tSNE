#include <FMM_tSNE/opengl_interaction/texture.h>

#include <glad/glad.h>
#include <iostream>
#include <stb_image/stb_image.h>

Texture::Texture(const char* fileName)
{
    glGenTextures(1, &TEX);
    glBindTexture(GL_TEXTURE_2D, TEX);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // int stbiWidth, stbiHeight, stbiNrChannels;
    stbi_set_flip_vertically_on_load(true);
    data = stbi_load(fileName, &width, &height, &nrChannels, 0);

    if (data)
    {
        switch (nrChannels)
        {
        case 4:
            glTexImage2D(
                GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
            glGenerateMipmap(GL_TEXTURE_2D);
            break;
        case 3:
            glTexImage2D(
                GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
            glGenerateMipmap(GL_TEXTURE_2D);
            break;
        default:
            std::cout << "unsupported number of channels for texture" << std::endl;
        }
    }
    else
    {
        std::cout << "Failed to load texture" << std::endl;
    }
}

Texture::Texture(int initWidth, int initHeight)
{
    width = initWidth;
    height = initHeight;
    nrChannels = 4;

    glGenTextures(1, &TEX);
    glBindTexture(GL_TEXTURE_2D, TEX);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    data = new unsigned char[initWidth * initHeight * 4];
    for (int i = 0; i < initWidth * initHeight * 4; i++)
    {
        data[i] = (unsigned char) 0;
    }

    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGBA, initWidth, initHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    // glGenerateMipmap(GL_TEXTURE_2D);
}

Texture::~Texture()
{
    stbi_image_free(data);

    if (TEX != 0)
    {
        glDeleteTextures(1, &TEX);
        TEX = 0;
    }
}

void Texture::updateData()
{
    glBindTexture(GL_TEXTURE_2D, TEX);

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, data);
}
