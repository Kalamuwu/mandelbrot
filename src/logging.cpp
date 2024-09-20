#include "logging.hpp"

// _checkSDLError
#include <SDL2/SDL.h>

// _checkTTFError
#include <SDL2/SDL_ttf.h>

// _checkHIPError
#include "hip-commons.hpp"

// _checkLibVAError
extern "C" {
    #include <libavutil/opt.h>
}


void _checkSDLError(
    int const code,
    char const *const func,
    const char *const file,
    int const line)
{
    if (code != 0)
    {
        fprintf(stderr, "\nSDL Error: %s\n...at %s:%d '%s'\n",
            SDL_GetError(), file, line, func);
        fflush(stderr);

        #if KILL_ON_ERR
            exit(code);
        #endif
    }
}

void _checkTTFError(
    int const code,
    char const *const func,
    const char *const file,
    int const line)
{
    if (code != 0)
    {
        fprintf(stderr, "\nTTF Error: %s\n...at %s:%d '%s'\n",
            TTF_GetError(), file, line, func);
        fflush(stderr);

        #if KILL_ON_ERR
            exit(code);
        #endif
    }
}


void _checkHIPError(
    hipError_t const code,
    char const *const func,
    const char *const file,
    int const line)
{
    if (code != 0)
    {
        fprintf(stderr, "\nHIP Error: %s\n...at %s:%d '%s'\n",
            hipGetErrorString(code), file, line, func);
        fflush(stderr);

        #if KILL_ON_ERR
            hipDeviceReset();
            exit(static_cast<unsigned int>(code));
        #endif
    }
}


void _checkLibVAError(
    int const code,
    char const *const func,
    const char *const file,
    int const line)
{
    if (code != 0)
    {
        char* buf = new char[1024];
        av_make_error_string(buf, 1024, code);

        fprintf(stderr, "\nlibva Error: %s\n...at %s:%d '%s'\n",
            buf, file, line, func);
        fflush(stderr);

        #if KILL_ON_ERR
            exit(code);
        #endif
    }
}
