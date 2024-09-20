#include "video.hpp"
#include "logging.hpp"


// font path and size
#define FONT "NotoSansMono-Regular.ttf", 14


VideoScreen::VideoScreen(const uint32_t w, const uint32_t h)
:
    IVideoOutput(w, h)
{
    // init SDL2
    constexpr Uint32 initFlags =
        SDL_INIT_TIMER | SDL_INIT_VIDEO | SDL_INIT_EVENTS;
    //constexpr int imgFlags = IMG_INIT_PNG;
                                          // ah, SDL2...
    checkSDLError(SDL_Init(initFlags));      // >0 on fail
  //checkSDLError(IMG_Init(imgFlags) == 0);  //  0 on fail
    checkTTFError(TTF_Init(/* no flags*/));  // -1 on fail
    // SDL_gfx needs no init

    // create the window
    //constexpr Uint32 windowFlags = SDL_WINDOW_RESIZABLE;
    constexpr Uint32 windowFlags = 0;
    mp_window = SDL_CreateWindow(
        "Mandelbrot",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        w, h,
        windowFlags);
    //SDL_SetWindowMinimumSize(mp_window, 800, 700);
    checkSDLError(mp_window == NULL);

    // create the renderer
    //constexpr Uint32 renderFlags = SDL_RENDERER_ACCELERATED;
    constexpr Uint32 renderFlags = 0;
    mp_renderer = SDL_CreateRenderer(mp_window, -1, renderFlags);
    checkSDLError(mp_renderer == NULL);
    // SDL_BlendMode blend = SDL_BlendMode::SDL_BLENDMODE_BLEND;
    // checkSDLError(SDL_SetRenderDrawBlendMode(mp_renderer, blend));

    // create the texture that we will draw on
    mp_display = SDL_CreateTexture(
        mp_renderer,
        SDL_PIXELFORMAT_RGBA8888,
        SDL_TEXTUREACCESS_STATIC,
        m_width, m_height);
    checkSDLError(mp_display == NULL);

    // ...and its custom pixel buffer
    mp_textureBuffer = new uint32_t[ m_width * m_height ];

    // create our font
    mp_font = TTF_OpenFont(FONT);
    checkTTFError(mp_font == NULL);

    // for fps tracking+limiting
    const Uint32 ticksNow = SDL_GetTicks();
    m_fpsData.framesSinceLastUpd = 0;
    m_fpsData.lastFrameTicks = ticksNow;
    m_fpsData.lastUpdTicks = ticksNow;

    // all OK!
    DEBUG("screen: initialized OK\n");
}

VideoScreen::~VideoScreen(void)
{
    DEBUG_FILELOC();

    TTF_CloseFont(mp_font);
    mp_font = nullptr;

    delete[] mp_textureBuffer;
    mp_textureBuffer = nullptr;

    SDL_DestroyTexture(mp_display);
    mp_display = nullptr;

    SDL_DestroyRenderer(mp_renderer);
    mp_renderer = nullptr;

    SDL_DestroyWindow(mp_window);
    mp_window = nullptr;

    // quit SDL2 subsystems
    TTF_Quit();
  //IMG_Quit();
    SDL_Quit();

    DEBUG("screen: quit OK\n");
}


void VideoScreen::pushFrame(void)
{
    DEBUG_FILELOC();

    // copy display texture to window and render
    checkSDLError(SDL_UpdateTexture(
        mp_display, NULL, mp_textureBuffer, m_width * sizeof(uint32_t)));
    checkSDLError(SDL_RenderCopy(mp_renderer, mp_display, NULL, NULL));
    SDL_RenderPresent(mp_renderer);

    // calculate fps
    const Uint32 ticksNow = SDL_GetTicks();
    m_fpsData.framesSinceLastUpd++;
    if (ticksNow - m_fpsData.lastUpdTicks > 1000)
    {
        m_fps = m_fpsData.framesSinceLastUpd;
        m_fpsData.framesSinceLastUpd = 0;
        m_fpsData.lastUpdTicks = ticksNow;
    }

    // calculate delay to reach target fps
    if (m_fpsData.targetFps)
    {
        // note: uses SDL2's Uint32 instead of stdint.h's uint32_t (even
        // though they're identical)
        const Uint32 ticksNow = SDL_GetTicks();
        const Uint32 targetDelay = 1000 / m_fpsData.targetFps;
        const Uint32 ticksElapsed = ticksNow - m_fpsData.lastFrameTicks;
        if (ticksElapsed > targetDelay)
            DEBUG("Can't keep up! Running %u ms behind "
                "(target delay %u ms, elapsed %u ms)\n",
                ticksElapsed - targetDelay, // te>td, so no risk of underflow
                targetDelay, ticksElapsed);
        else SDL_Delay(targetDelay - ticksElapsed); // td>=te, no underflow
        m_fpsData.lastFrameTicks = SDL_GetTicks();
    }

    m_frameIdx++;
}


void VideoScreen::pixel(
    const int32_t x, const int32_t y, const Color c)
{
    if (x < 0) return;
    else if ((uint32_t)x >= m_width) return;
    if (y < 0) return;
    else if ((uint32_t)y >= m_height) return;

    const uint64_t i = (uint64_t)y * m_width + x;
    mp_textureBuffer[i] = c.col();
}


// specialized drawing functions

void VideoScreen::drawText(
    const char* s,
    const int32_t x, const int32_t y,
    const Color c)
{
    SDL_Surface* surf = TTF_RenderText_Solid(mp_font, s, {255,255,255,255});
    checkTTFError(surf == NULL);
    if ((surf != NULL) && (surf->w > 0) && (surf->h > 0))
    {
        SDL_LockSurface(surf);
        Uint8* const pixels = (Uint8*)surf->pixels;
        // blit each pixel to screen
        for (int32_t dy = 0; dy < surf->h; dy++)
            for (int32_t dx = 0; dx < surf->w; dx++)
                if (pixels[dy * surf->pitch + dx])
                    pixel(x+dx, y+dy, c);
        SDL_UnlockSurface(surf);
    }
    // safe to call SDL_FreeSurface with NULL
    SDL_FreeSurface(surf);
}

void VideoScreen::drawText(
    const char* s,
    const int32_t x, const int32_t y,
    const Color fg, const Color bg)
{
    SDL_Surface* surf = TTF_RenderText_Solid(mp_font, s, {255,255,255,255});
    checkTTFError(surf == NULL);
    if ((surf != NULL) && (surf->w > 0) && (surf->h > 0))
    {
        SDL_LockSurface(surf);
        Uint8* const pixels = (Uint8*)surf->pixels;
        // blit each pixel to screen
        for (int32_t dy = 0; dy < surf->h; dy++)
            for (int32_t dx = 0; dx < surf->w; dx++)
                pixel(x+dx, y+dy,
                    (pixels[dy * surf->pitch + dx])? fg : bg);
        SDL_UnlockSurface(surf);
    }
    // safe to call SDL_FreeSurface with NULL
    SDL_FreeSurface(surf);
}

