#include "KeyInput.h"
#include <algorithm>

std::vector<KeyInput*> KeyInput::_instances;


KeyInput::KeyInput(std::vector<int> keysToListen, std::vector<int> keysToListenRelease)
{
    for (int x : keysToListen)
        keymap.insert(std::make_pair(x, false));

    for (int x : keysToListenRelease)
        releasemap.insert(std::make_pair(x, false));

    KeyInput::_instances.push_back(this);

    scrollOffset = 0;
}

KeyInput::~KeyInput()
{
    //_instances.erase(std::remove(_instances.begin(), _instances.end(), this), _instances.end());
}

void KeyInput::setupKeyCallback(GLFWwindow* window)
{
    glfwSetKeyCallback(window, KeyInput::key_callback);
	glfwSetMouseButtonCallback(window, KeyInput::mouse_button_callback);
    glfwSetScrollCallback(window, KeyInput::scroll_callback);
}

void KeyInput::setKeyDown(int key, bool flag)
{
    auto it = keymap.find(key);
    if (it != keymap.end())
        keymap[key] = flag;
}

bool KeyInput::getKeyDown(int key)
{
    bool result = false;
    auto it = keymap.find(key);
    if (it != keymap.end())
        result = keymap[key];
    return result;
}

bool KeyInput::getMouseButtonDown(int key)
{
    bool result = getKeyDown(key);
    if (result)
    {
        if (getKeyReleased(key))
        {
            setKeyReleased(key, false);
            return true;
        }
        return false;
    }
    else
    {
        if (!getKeyReleased(key))
            setKeyReleased(key, true);
        return false;
    }
}

void KeyInput::setKeyReleased(int key, bool flag)
{
    auto it = releasemap.find(key);
    if (it != releasemap.end())
        releasemap[key] = flag;
}

bool KeyInput::getKeyReleased(int key)
{
    bool result = false;
    auto it = releasemap.find(key);
    if (it != releasemap.end())
        result = releasemap[key];
    return result;
}

int KeyInput::getScrollOffset()
{
    int ret = this->scrollOffset;
    this->scrollOffset = 0;
    return ret;
}

void KeyInput::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    for (KeyInput* instance : _instances)
    {
        instance->setKeyDown(key, action != GLFW_RELEASE);
		instance->setKeyReleased(key, action == GLFW_RELEASE);
    }
}

void KeyInput::mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    for (KeyInput* instance : _instances)
        instance->setKeyDown(button, action == GLFW_PRESS);
}

void KeyInput::scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    for (KeyInput* instance : _instances)
    {
        if (yoffset < 0)
            instance->scrollOffset++;
        else if (yoffset > 0)
            instance->scrollOffset--;
    }
}
