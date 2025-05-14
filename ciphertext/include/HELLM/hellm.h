#ifndef HELLM_H
#define HELLM_H

#include <cstdint>
#include <vector>
#include <string>

namespace HELLM {

class FHEContext {
public:
    FHEContext() = default;
    ~FHEContext() = default;
    
    bool initialize(const std::string& key_path);
    bool encrypt(const float* input, size_t size, float* output);
    bool decrypt(const float* input, size_t size, float* output);
};

} // namespace HELLM

#endif // HELLM_H
