#pragma once

#include <cstddef>
#include <glad/glad.h>
#include <iostream>
#include <vector>

enum class DataLayout
{
    Float2_Float3,
    Float2_Float2_Int1,
    Float2_Float2_Int1_Int1,
    Float2_Float2_Int1_Int1_Int32t1,
    Double2_Double2_Int1,
    Double2_Double2_Int1_Int1_Int32t1
};

class Buffer
{
  public:
    Buffer()
        : elem_count_{0}
    {
        glGenBuffers(1, &VBO_);
        glGenVertexArrays(1, &VAO_);
        glGenBuffers(1, &EBO_);
    }

    template <typename T>
    Buffer(const std::vector<T>& elems_to_buffer, DataLayout data_layout, GLenum buffer_type)
        : elem_count_{elems_to_buffer.size()}
    {
        glGenBuffers(1, &VBO_);
        glGenVertexArrays(1, &VAO_);

        create_vertex_buffer(elems_to_buffer, data_layout, buffer_type);
    }

    ~Buffer()
    {
        if (VAO_ != 0) glDeleteVertexArrays(1, &VAO_);
        if (VBO_ != 0) glDeleteBuffers(1, &VBO_);
        if (EBO_ != 0) glDeleteBuffers(1, &EBO_);
    }

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept
        : VAO_{other.VAO_},
          VBO_{other.VBO_},
          EBO_{other.EBO_},
          elem_count_{other.elem_count_}
    {
        other.VAO_ = 0;
        other.VBO_ = 0;
        other.EBO_ = 0;
        other.elem_count_ = 0;
    }
    Buffer& operator=(Buffer&& other) noexcept
    {
        if (this != &other)
        {
            if (VAO_ != 0) glDeleteVertexArrays(1, &VAO_);
            if (VBO_ != 0) glDeleteBuffers(1, &VBO_);
            if (EBO_ != 0) glDeleteBuffers(1, &EBO_);
            VAO_ = other.VAO_;
            VBO_ = other.VBO_;
            EBO_ = other.EBO_;
            elem_count_ = other.elem_count_;

            other.VAO_ = 0;
            other.VBO_ = 0;
            other.EBO_ = 0;
            other.elem_count_ = 0;
        }
        return *this;
    }

    inline void bind_VAO() const noexcept { glBindVertexArray(VAO_); }
    [[nodiscard]] std::size_t get_elem_count_() const noexcept { return elem_count_; }

    // change this so gl dynamic draw is a parameter!
    template <typename T>
    void update_vertex_buffer(const std::vector<T>& elems_to_buffer, DataLayout data_layout)
    {
        glBindVertexArray(VAO_);
        glBindBuffer(GL_ARRAY_BUFFER, VBO_);

        std::size_t elems_byte_size = elems_to_buffer.size() * sizeof(T);

        // check if new data is more then current buffer
        if (elems_to_buffer.size() > elem_count_)
        {
            // buffer data is for updating buffer to larger size
            glBufferData(GL_ARRAY_BUFFER, elems_byte_size, elems_to_buffer.data(), GL_DYNAMIC_DRAW);
        }
        else
        {
            // buffer sub data is for updating partial/whole buffer
            glBufferSubData(GL_ARRAY_BUFFER, 0, elems_byte_size, elems_to_buffer.data());
        }

        elem_count_ = elems_to_buffer.size();
    }

    template <typename T>
    void create_vertex_buffer(const std::vector<T>& elems_to_buffer, DataLayout data_layout, GLenum buffer_type)
    {
        glBindVertexArray(VAO_);

        glBindBuffer(GL_ARRAY_BUFFER, VBO_);

        std::size_t elems_byte_size = elems_to_buffer.size() * sizeof(T);
        glBufferData(GL_ARRAY_BUFFER, elems_byte_size, elems_to_buffer.data(), buffer_type);

        // clang-format off
        switch (data_layout)
        {
        case DataLayout::Float2_Float3:
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(T), (void*) 0);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(T), (void*) (2 * sizeof(float)));
            glEnableVertexAttribArray(1);
            break;
        case DataLayout::Float2_Float2_Int1:
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(T), (void*) 0);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(T), (void*) (2 * sizeof(float)));
            glEnableVertexAttribArray(1);
            glVertexAttribIPointer(2, 1, GL_INT,            sizeof(T), (void*) (4 * sizeof(float)));
            glEnableVertexAttribArray(2);
            break;
        case DataLayout::Float2_Float2_Int1_Int1:
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(T), (void*) 0);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(T), (void*) (2 * sizeof(float)));
            glEnableVertexAttribArray(1);
            glVertexAttribIPointer(2, 1, GL_INT,            sizeof(T), (void*) (4 * sizeof(float)));
            glEnableVertexAttribArray(2);
            glVertexAttribIPointer(3, 1, GL_INT,            sizeof(T), (void*) (4 * sizeof(float) + 1 * sizeof(int)));
            glEnableVertexAttribArray(3);
            break;
        case DataLayout::Float2_Float2_Int1_Int1_Int32t1:
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(T), (void*) 0);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(T), (void*) (2 * sizeof(float)));
            glEnableVertexAttribArray(1);
            glVertexAttribIPointer(2, 1, GL_INT,            sizeof(T), (void*) (4 * sizeof(float)));
            glEnableVertexAttribArray(2);
            glVertexAttribIPointer(3, 1, GL_INT,            sizeof(T), (void*) (4 * sizeof(float) + 1 * sizeof(int)));
            glEnableVertexAttribArray(3);
            glVertexAttribIPointer(4, 1,  GL_UNSIGNED_INT,  sizeof(T), (void*) (4 * sizeof(float) + 2 * sizeof(int)));
            glEnableVertexAttribArray(4);
            break;
        case DataLayout::Double2_Double2_Int1:
            glVertexAttribPointer(0, 2, GL_DOUBLE, GL_FALSE, sizeof(T), (void*) 0);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(1, 2, GL_DOUBLE, GL_FALSE, sizeof(T), (void*) (2 * sizeof(double)));
            glEnableVertexAttribArray(1);
            glVertexAttribIPointer(2, 1, GL_INT,             sizeof(T), (void*) (4 * sizeof(double)));
            glEnableVertexAttribArray(2);
            break;
        case DataLayout::Double2_Double2_Int1_Int1_Int32t1:
            glVertexAttribPointer(0, 2, GL_DOUBLE, GL_FALSE, sizeof(T), (void*) 0);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(1, 2, GL_DOUBLE, GL_FALSE, sizeof(T), (void*) (2 * sizeof(double)));
            glEnableVertexAttribArray(1);
            glVertexAttribIPointer(2, 1, GL_INT,             sizeof(T), (void*) (4 * sizeof(double)));
            glEnableVertexAttribArray(2);
            glVertexAttribIPointer(3, 1, GL_INT,             sizeof(T), (void*) (4 * sizeof(double) + 1 * sizeof(int)));
            glEnableVertexAttribArray(3);
            glVertexAttribIPointer(4, 1, GL_UNSIGNED_INT,    sizeof(T), (void*) (4 * sizeof(double) + 2 * sizeof(int)));
            glEnableVertexAttribArray(4);
            break;
        default:
            std::cout << "invalid BufferType given\n";
            // clang-format on
        }
    }

  private:
    GLuint VBO_;
    GLuint VAO_;
    GLuint EBO_;
    std::size_t elem_count_;
};
