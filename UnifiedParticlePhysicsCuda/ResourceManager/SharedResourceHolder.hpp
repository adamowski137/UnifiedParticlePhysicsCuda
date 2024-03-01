#pragma once

#include <unordered_map>
#include <string>
#include <memory>

template<typename Resource>
class SharedResourceManager
{
public:
    SharedResourceManager() {};
    void load(const std::string& name, const std::string path)
    {
        m_resources.insert(std::make_pair(name, std::make_shared<Resource>()));
        m_resources[name].get()->createFromFile(path);
    }

    std::shared_ptr<Resource>& operator[](const std::string& name)
    {
        return m_resources.at(name);
    }

private:
    bool exists(const std::string& name) const
    {
        return (m_resources.find(name) != m_resources.end());
    }
    std::unordered_map<std::string, std::shared_ptr<Resource>> m_resources;
};