#pragma once

#include <cstdint>
#include <glm/glm.hpp>

class TsnePoint2D
{
  public:
    glm::dvec2 position{0.0};
    glm::dvec2 derivative{0.0};
    int label{0};
#ifdef INDEX_TRACKER
    int ID{0};
    std::uint32_t morton_code{0};
#endif

    TsnePoint2D(glm::dvec2 initPosition,
                glm::dvec2 initDerivative,
                int initLabel
#ifdef INDEX_TRACKER
                ,
                int initID,
                std::uint32_t init_morton_code
#endif
                )
        : position(initPosition),
          derivative(initDerivative),
          label(initLabel)
#ifdef INDEX_TRACKER
          ,
          ID(initID),
          morton_code(init_morton_code)
#endif
    {
    }

    TsnePoint2D() = default;
};

#ifdef INDEX_TRACKER
struct TsnePoint2DLessthan
{
    inline bool operator()(const TsnePoint2D& x, const TsnePoint2D& y) const { return x.morton_code < y.morton_code; }
};

struct TsnePoint2DRightshift
{
    inline uint32_t operator()(const TsnePoint2D& x, const unsigned offset) const { return x.morton_code >> offset; }
};
#endif
