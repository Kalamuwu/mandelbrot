#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string>
#include <chrono>

#include <SDL2/SDL.h>
#include "hip-commons.hpp"

#include "complex.hpp"
#include "color.hpp"
#include "pattern.hpp"
#include "video.hpp"
#include "logging.hpp"


#define MAX_DEPTH 1024lu
#define N_BANDS 24u
#define INTERACTIVE true
#define USE_GPU true


#if INTERACTIVE

int main()
{
    // initialize SDL2 and a renderable window
    VideoScreen screen(1280, 720);
    screen.setTargetFps(50);
    viewsettings view;
    view.width  = screen.getWidth();
    view.height = screen.getHeight();
    const uint64_t npix = view.width*view.height;

    // set up cpu fractal pixel buffer
    // note that this is different than the screen buffer, so that we can draw
    // text onto the screen without losing the fractal
    uint32_t* fractalbuf = new uint32_t[npix];


#if USE_GPU
    // set up gpu pixel buffer
    uint32_t* d_buf;
    checkHIPError(hipMalloc((void**)&d_buf, sizeof(uint32_t) * npix));
    checkHIPError(hipMemset(d_buf, 0x00, sizeof(uint32_t) * npix));
    checkHIPError(hipDeviceSynchronize());

    // prepare kernel dims
    const dim3 blockDims(32,32);
    const dim3 gridDims(
        ceiling_div(view.width,  blockDims.x),
        ceiling_div(view.height, blockDims.y));
#endif

    // grab some system cursors
    SDL_Cursor* const cursorArrow =
        SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_ARROW);
    SDL_Cursor* const cursorArrowWait =
        SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_WAITARROW);
    checkSDLError(cursorArrow     == NULL);
    checkSDLError(cursorArrowWait == NULL);

    // timing for render
    uint64_t rendertime_ns = 0;

    // current exponent and threshhold information
    PRECISION_FP exponent = 2.0;
    PRECISION_FP threshhold = 2.0;

    // current view settings
    view.center = complex{ 0.0, 0.0 };
    view.zoom = 100.0;

    // determines if we need to recalculate the fractal or not
    // note that this is initially true so that the fractal gets drawn on the
    // first loop
    bool screenchanged = true;

    char rendertime_str[64] {0};
    char zoom_str[64] {0};
    char mathsettings_str[64] {0};

    // fetch keyboard state pointer
    const Uint8* const keyboard = SDL_GetKeyboardState(NULL);

    // main loop
    SDL_Event e;
    while (true)
    {
        DEBUG_FILELOC();

        // handle events
        while (SDL_PollEvent(&e))
            if (e.type == SDL_QUIT)
                goto quit;

        // handle keyboard
        const PRECISION_FP tfps = screen.getTargetFps();
        const PRECISION_FP lshift = keyboard[SDL_SCANCODE_LSHIFT]? 5.0 : 1.0;
        const PRECISION_FP deltacenter = lshift * 400.0 / (tfps * view.zoom);
        const PRECISION_FP deltazoom =   lshift * 1.0   /  tfps;
        const PRECISION_FP deltaexp =    lshift * 0.05  /  tfps;
        const PRECISION_FP deltathresh = lshift * 0.5   /  tfps;
        if (keyboard[SDL_SCANCODE_S])  // -imag
        {
            screenchanged = true;
            view.center.imag += deltacenter;
        }
        if (keyboard[SDL_SCANCODE_W])  // +imag
        {
            screenchanged = true;
            view.center.imag -= deltacenter;
        }
        if (keyboard[SDL_SCANCODE_A])  // -real
        {
            screenchanged = true;
            view.center.real -= deltacenter;
        }
        if (keyboard[SDL_SCANCODE_D])  // +real
        {
            screenchanged = true;
            view.center.real += deltacenter;
        }
        if (keyboard[SDL_SCANCODE_Q])  // -zoom
        {
            screenchanged = true;
            view.zoom -= view.zoom * deltazoom;
        }
        if (keyboard[SDL_SCANCODE_E])  // +zoom
        {
            screenchanged = true;
            view.zoom += view.zoom * deltazoom;
        }
        if (keyboard[SDL_SCANCODE_LEFTBRACKET])  // -exp
        {
            screenchanged = true;
            exponent -= deltaexp;
        }
        if (keyboard[SDL_SCANCODE_RIGHTBRACKET])  // +exp
        {
            screenchanged = true;
            exponent += deltaexp;
        }
        if (keyboard[SDL_SCANCODE_MINUS])  // -thresh
        {
            screenchanged = true;
            threshhold -= deltathresh;
        }
        if (keyboard[SDL_SCANCODE_EQUALS])  // +thresh
        {
            screenchanged = true;
            threshhold += deltathresh;
        }
        if (keyboard[SDL_SCANCODE_R])  // reset view
        {
            screenchanged = true;
            view.center = complex{ 0.0, 0.0 };
            view.zoom = 100;
        }

        // run calculations, if needed
        if (screenchanged)
        {
            screenchanged = false;
            SDL_SetCursor(cursorArrowWait);
            const auto renderstart = std::chrono::high_resolution_clock::now();

#if USE_GPU
            Mandelbrot::render_gpu<<<gridDims, blockDims>>>(
                d_buf, view,
                2.0, 2.0);

            // copy back to cpu for display
            checkHIPError(hipMemcpy(
                fractalbuf, d_buf,
                sizeof(uint32_t) * npix,
                hipMemcpyDeviceToHost));
            checkHIPError(hipDeviceSynchronize());
#else
            Mandelbrot::render_cpu(
                fractalbuf, view,
                2.0, 2.0);
#endif

            // calculate render time
            const auto renderstop = std::chrono::high_resolution_clock::now();
            SDL_SetCursor(cursorArrow);
            rendertime_ns =
                std::chrono::duration_cast<std::chrono::nanoseconds>
                (renderstop - renderstart)
                .count();

            // pretty-print render time: up to 9999 of smallest unit
            if (rendertime_ns < 9'999)
                sprintf(rendertime_str,
                    "render took: %lu ns", rendertime_ns);
            else if (rendertime_ns < 9'999'999)
                sprintf(rendertime_str,
                    "render took: %lu us", rendertime_ns / 1'000lu);
            else if (rendertime_ns < 9'999'999'999)
                sprintf(rendertime_str,
                    "render took: %lu ms", rendertime_ns / 1'000'000lu);
            else
                sprintf(rendertime_str,
                    "render took: %lu s",  rendertime_ns / 1'000'000'000lu);

            // update pos+zoom string
            sprintf(zoom_str,
                "pos: %+.5" PRIpFP "%+.5" PRIpFP "i zoom: %" PRIpFP "x",
                view.center.real, view.center.imag, view.zoom);

            // update exponent+threshhold string
            sprintf(mathsettings_str,
                "exp: %+" PRIpFP "i thresh: %" PRIpFP "x",
                exponent, threshhold);
        }

        // blit fractal to screen
        for (uint32_t y = 0; y < view.height; y++)
            for (uint32_t x = 0; x < view.width; x++)
                screen.pixel(x, y, fractalbuf[y*view.width + x]);

        // find mouse position and label that point
        if (keyboard[SDL_SCANCODE_LCTRL])
        {
            int x, y; SDL_GetMouseState(&x, &y);
            const complex c = pixToComplex(x, y, view);
            const std::size_t n = Mandelbrot::depth(c, exponent, threshhold);
            char s[64];
            if (n != MAX_DEPTH)
                sprintf(s,
                    "%+.5" PRIpFP "%+.5" PRIpFP "i : %lu",
                     c.real,       c.imag,           n);
            else
                sprintf(s,
                    "%+.5" PRIpFP "%+.5" PRIpFP "i : converges",
                     c.real,       c.imag);
            screen.drawText(s, x+15, y, WHITE, BLACK);
        }

        // show fps and render time in top-left
        char s[64] {0};
        sprintf(s, "FPS %3u (target %3u)",
            std::min(999u, screen.getFps()),
            std::min(999u, screen.getTargetFps()));
        screen.drawText(s,                5,  5, WHITE, BLACK);
        screen.drawText(rendertime_str,   5, 20, WHITE, BLACK);
        screen.drawText(zoom_str,         5, 35, WHITE, BLACK);
        screen.drawText(mathsettings_str, 5, 50, WHITE, BLACK);

        // crosshair
        screen.pixel(view.width/2, view.height/2, WHITE);

        // display
        screen.pushFrame();
    }

quit:

    DEBUG_FILELOC();

#if USE_GPU
    checkHIPError(hipFree(d_buf));
#endif

    delete[] fractalbuf;
    return 0;
}

#else

int main()
{
    // const char* filename = "/home/kalamari/projects/mandelbrot/out.mkv";
    // const char* encoder = "libvpx-vp9";
    //
    // std::vector<EncoderSetting> encoderSettings;
    // encoderSettings.emplace_back( "lossless", "1" );
    // encoderSettings.emplace_back( "row-mt",   "1" );

    const char* filename = "/home/kalamari/projects/mandelbrot/out.mp4";
    const char* encoder = "libx264";

    std::vector<EncoderSetting> encoderSettings;
    encoderSettings.emplace_back( "preset", "slower" );

    // initialize video output
    VideoFile video(
        1280, 720,
        40,
        filename,
        encoder,
        &encoderSettings);
    const uint32_t w = 1280;
    const uint32_t h = 720;
    const uint64_t npix = w*h;

    // set up gpu pixel buffer
    uint32_t* d_buf;
    checkHIPError(hipMalloc((void**)&d_buf, sizeof(uint32_t) * npix));
    checkHIPError(hipMemset(d_buf, 0x00, sizeof(uint32_t) * npix));
    checkHIPError(hipDeviceSynchronize());

    // set up cpu fractal pixel buffer
    // note that this is different than the screen buffer, so that we can draw
    // text onto the screen without losing the fractal
    uint32_t* fractalbuf = new uint32_t[npix];

    // prepare kernel dims
    const dim3 blockDims(32,32);
    const dim3 gridDims(
        ceiling_div(w, blockDims.x),
        ceiling_div(h, blockDims.y));

    // timing for render
    uint64_t rendertime_ns = 0;

    // current zoom and window location information
    complex zoomcenter { 0.0, 0.0 };
    PRECISION_FP zoomscale = 250.0;

    // current exponent and threshhold information
    PRECISION_FP exponent = 2.0;
    PRECISION_FP threshhold = 3.0;

    for (int i = 0; i < 40*10; i++)
    {
        // update exponent
        exponent -= 0.0175;

        const auto renderstart = std::chrono::high_resolution_clock::now();

        // run fractal generation kernel
        render<<<gridDims, blockDims>>>(
            d_buf, w, h,
            zoomcenter, zoomscale,
            exponent, threshhold);
        checkHIPError(hipGetLastError());
        checkHIPError(hipDeviceSynchronize());

        // copy back to cpu for display
        checkHIPError(hipMemcpy(
            fractalbuf, d_buf,
            sizeof(uint32_t) * npix,
            hipMemcpyDeviceToHost));
        checkHIPError(hipDeviceSynchronize());

        // blit fractal to screen
        for (uint32_t y = 0; y < h; y++)
            for (uint32_t x = 0; x < w; x++)
                video.pixel(x, y, fractalbuf[y*w + x]);

        // calculate render time
        const auto renderstop = std::chrono::high_resolution_clock::now();
        rendertime_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>
            (renderstop - renderstart)
            .count();
        // pretty-print render time: up to 9999 of smallest unit
        if (rendertime_ns < 9'999)
            fprintf(stderr,
                "render took: %lu ns\n", rendertime_ns);
        else if (rendertime_ns < 9'999'999)
            fprintf(stderr,
                "render took: %lu us\n", rendertime_ns / 1'000lu);
        else if (rendertime_ns < 9'999'999'999)
            fprintf(stderr,
                "render took: %lu ms\n", rendertime_ns / 1'000'000lu);
        else
            fprintf(stderr,
                "render took: %lu s\n",  rendertime_ns / 1'000'000'000lu);
        fflush(stderr);

        // display
        video.pushFrame();
    }

    DEBUG_FILELOC();
    // Screen destructor will handle SDL stuff
    delete[] fractalbuf;
    return 0;
}

#endif
