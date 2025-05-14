#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <cstdint>
#include <cstdlib>

int main() {
    const char* env_key_path = std::getenv("HELLM_KEY_PATH");
    std::string key_path = env_key_path ? env_key_path : "./key";
    
    std::cout << "Generating encryption keys in " << key_path << std::endl;
    
    // Create random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> dist(0, 255);
    
    // Generate encryption key
    std::vector<uint8_t> enc_key(1024);
    for (auto& byte : enc_key) {
        byte = dist(gen);
    }
    
    // Generate decryption key
    std::vector<uint8_t> dec_key(1024);
    for (auto& byte : dec_key) {
        byte = dist(gen);
    }
    
    // Write keys to files
    std::ofstream enc_file(key_path + "/encryption.key", std::ios::binary);
    if (!enc_file) {
        std::cerr << "Failed to open encryption.key for writing" << std::endl;
        return 1;
    }
    enc_file.write(reinterpret_cast<const char*>(enc_key.data()), enc_key.size());
    
    std::ofstream dec_file(key_path + "/decryption.key", std::ios::binary);
    if (!dec_file) {
        std::cerr << "Failed to open decryption.key for writing" << std::endl;
        return 1;
    }
    dec_file.write(reinterpret_cast<const char*>(dec_key.data()), dec_key.size());
    
    std::cout << "Keys generated successfully" << std::endl;
    return 0;
}
