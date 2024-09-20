#ifndef VIDEOH
#define VIDEOH

#include <stdint.h>
#include <vector>

extern "C" {
    #include <libswscale/swscale.h>
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
}

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>

#include "color.hpp"


class IVideoOutput
{
public:
    IVideoOutput(const uint32_t width, const uint32_t height) :
        m_width(width), m_height(height), m_npix(width*height) {}
    virtual ~IVideoOutput(void) {}

    virtual void pushFrame(void) = 0;
    virtual void pixel(
        const int32_t x, const int32_t y,
        const Color c) = 0;

    uint32_t  getWidth(void)    const { return m_width;    }
    uint32_t  getHeight(void)   const { return m_height;   }
    uint64_t  getNumPix(void)   const { return m_npix;     }
    uint32_t  getFrameIdx(void) const { return m_frameIdx; }

protected:
    // video information
    uint32_t m_frameIdx = 0;

    // frame information
    const uint32_t m_width, m_height;
    const uint64_t m_npix;  // macro to w*h
};


struct EncoderSetting
{
    constexpr EncoderSetting(const char* k, const char* v)
    : key(k), val(v) {}
    constexpr EncoderSetting(void) : key(nullptr), val(nullptr) {}

    const char* key;
    const char* val;
};

/**
* Represents an object that manages frames and video formatting, such that
* frames can be written directly to video files.
*/
class VideoFile : public IVideoOutput
{
public:
    VideoFile(
        const uint32_t width, const uint32_t height,
        const uint32_t framerate,
        const char* path,
        const char* encoder,
        const std::vector<EncoderSetting>* encoderSettings);

    ~VideoFile(void);

    void pushFrame(void) override;
    void pixel(const int32_t x, const int32_t y, const Color c) override;

private:
    void encode(AVFrame* frame);

    // Video information
    const char* m_outfile;

    // Working frame
    AVFrame* mp_frameRGB, * mp_frameYUV;
    SwsContext* mp_swsCtx;  // for converting RGB to YUV

    // Contexts and encoders
    AVPacket* mp_pkt;
    const AVOutputFormat* mp_fmt;
    AVFormatContext* mp_fmtCtx;
    AVCodecContext* mp_codecCtx;
    AVStream* mp_videoStream;
};


/**
* Represents an object that manages frames and video formatting, such that
* frames can be written directly to the screen.
*/
class VideoScreen : public IVideoOutput
{
public:
    VideoScreen(const uint32_t width, const uint32_t height);
    ~VideoScreen(void);

    void pushFrame(void) override;
    void pixel(const int32_t x, const int32_t y, const Color c) override;

    // specialized drawing functions
    void drawText(
        const char* string,
        const int32_t x, const int32_t y,
        const Color color);
    void drawText(
        const char* string,
        const int32_t x, const int32_t y,
        const Color fg, const Color bg);

    uint32_t getFps(void) const { return m_fps; }
    uint32_t getTargetFps(void) const { return m_fpsData.targetFps; }
    void setTargetFps(const uint32_t fps) { m_fpsData.targetFps = fps; }

private:
    uint32_t* mp_textureBuffer;

    // for tracking fps
    struct {
        uint32_t targetFps = 0;
        uint32_t framesSinceLastUpd = 0;
        uint32_t lastFrameTicks;
        uint32_t lastUpdTicks;
    } m_fpsData;
    uint32_t m_fps = 0;

    // SDL2 render stuffs
    SDL_Renderer* mp_renderer = nullptr;
    SDL_Window* mp_window = nullptr;
    SDL_Texture* mp_display = nullptr;
    TTF_Font* mp_font = nullptr;
};

#endif  // VIDEOH
