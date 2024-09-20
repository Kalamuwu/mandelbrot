#include "video.hpp"
#include "logging.hpp"

#include <string>

extern "C" {
    #include <libswscale/swscale.h>
    #include <libavcodec/avcodec.h>
    #include <libavutil/mathematics.h>
    #include <libavutil/timestamp.h>
    #include <libavformat/avformat.h>
    #include <libavutil/opt.h>
}


VideoFile::VideoFile(
    const uint32_t w, const uint32_t h,
    const uint32_t fps,
    const char* path,
    const char* encoder,
    const std::vector<EncoderSetting>* encoderSettings)
:
    IVideoOutput(w, h),
    m_outfile(path)
{

    DEBUG_FILELOC();

    // extract file format...
    std::string strpath { m_outfile };
    auto const extIdx = strpath.find_last_of('.');
    if (extIdx == std::string::npos)
    {
        DEBUG("Could not extract file format");
        checkLibVAError(1);
    }
    std::string extension = strpath.substr(extIdx + 1);

    // ...and use it to set up output format
    mp_fmt = av_guess_format(extension.c_str(), NULL, NULL);
    checkLibVAError(
        avformat_alloc_output_context2(&mp_fmtCtx, NULL, NULL, m_outfile)
    );

    // set up output codex
    const AVCodec* codec = avcodec_find_encoder_by_name(encoder);
    if (!codec)
    {
        DEBUG("Encoder codec '%s' not found", encoder);
        checkLibVAError(1);
    }
    //AVDictionary* codec_options = NULL;
    //av_dict_set(&codec_options, "crf", "0", 0);

    // allocate video codec context
    mp_codecCtx = avcodec_alloc_context3(codec);
    if (!mp_codecCtx)
    {
        DEBUG("Could not allocate video codec context");
        checkLibVAError(1);
    }
    mp_pkt = av_packet_alloc();
    if (!mp_pkt)
    {
        DEBUG("Could not allocate video codec packet");
        checkLibVAError(1);
    }

    // set video params
    mp_codecCtx->width = m_width;
    mp_codecCtx->height = m_height;
    mp_codecCtx->pix_fmt = AV_PIX_FMT_YUV420P;

    // mp_codecCtx->time_base = (AVRational){ 1, m_framerate };
    // mp_codecCtx->framerate = (AVRational){ m_framerate, 1 };
    mp_codecCtx->time_base.num = 1;
    mp_codecCtx->time_base.den = fps;
    mp_codecCtx->framerate.num = fps;
    mp_codecCtx->framerate.den = 1;

    mp_codecCtx->gop_size = 12;  // emit I-frame every 12 frames at most
    mp_codecCtx->max_b_frames = 2;

    // some formats require a global header
    if (mp_fmtCtx->oformat->flags & AVFMT_GLOBALHEADER)
        mp_codecCtx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    // set encoder params
    if (encoderSettings != nullptr)
        for (EncoderSetting encSet : *encoderSettings)
        {
            if (encSet.key == nullptr || encSet.val == nullptr) continue;
            DEBUG("Encoder setting: %s = %s\n", encSet.key, encSet.val);
            checkLibVAError(
                av_opt_set(mp_codecCtx->priv_data, encSet.key, encSet.val, 0)
            );
        }

    // output streams -- just one: video
    mp_videoStream = avformat_new_stream(mp_fmtCtx, codec);
    mp_videoStream->time_base = mp_codecCtx->time_base;

    // open codec
    checkLibVAError(
        avcodec_open2(mp_codecCtx, codec, NULL /*&codec_options*/)
    );

    // initialize streams
    checkLibVAError(
        avcodec_parameters_from_context(mp_videoStream->codecpar, mp_codecCtx)
    );

    // stream 0: video
    av_dump_format(mp_fmtCtx, 0, m_outfile, 1);
    checkLibVAError(
        avio_open(&mp_fmtCtx->pb, m_outfile, AVIO_FLAG_WRITE)
    );

    // commit stream header data
    checkLibVAError(
        avformat_write_header(mp_fmtCtx, NULL/*&codec_options*/)
    );
    //av_dict_free(&codec_options);

    // allocate rgb and yuv frames
    mp_frameRGB = av_frame_alloc();
    mp_frameRGB->format = AV_PIX_FMT_RGBA;
    mp_frameRGB->width = m_width;
    mp_frameRGB->height = m_height;
    checkLibVAError(
        av_frame_get_buffer(mp_frameRGB, 0)
    );

    mp_frameYUV = av_frame_alloc();
    mp_frameYUV->format = AV_PIX_FMT_YUV420P;
    mp_frameYUV->width = m_width;
    mp_frameYUV->height = m_height;
    checkLibVAError(
        av_frame_get_buffer(mp_frameYUV, 0)
    );

    // for converting RGB to YUV
    mp_swsCtx = sws_getContext(
        m_width, m_height, AV_PIX_FMT_RGBA,    // source
        m_width, m_height, AV_PIX_FMT_YUV420P,  // dest
        SWS_FAST_BILINEAR,  // flags -- we aren't upscaling so this is moot
        NULL, NULL, // no filters necessary
        NULL);      // no params necessary

    DEBUG("videofile: initialized OK\n");
}


VideoFile::~VideoFile(void)
{
    DEBUG_FILELOC();

    // flush encoder
    encode(NULL);

    // write the end of the stream,
    av_write_trailer(mp_fmtCtx);
    // ... and close the file.
    avio_closep(&mp_fmtCtx->pb);

    // free all our video encoder stuffs

    avcodec_free_context(&mp_codecCtx);
    avformat_free_context(mp_fmtCtx);

    av_packet_free(&mp_pkt);

    av_frame_free(&mp_frameRGB);
    av_frame_free(&mp_frameYUV);
    sws_freeContext(mp_swsCtx);

    DEBUG("videofile: quit OK\n");
}


void VideoFile::encode(AVFrame* frame)
{
    DEBUG_FILELOC();

    // send frame to encoder
    if (frame) DEBUG("videofile: Sending frame %ld\n", frame->pts);

    checkLibVAError(avcodec_send_frame(mp_codecCtx, frame));

    int ret = 0;
    while (ret >= 0)
    {
        ret = avcodec_receive_packet(mp_codecCtx, mp_pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            return;
        else checkLibVAError(ret);

        // set packet PTS and DTS accounting for FPS and our format's timebase
        av_packet_rescale_ts(mp_pkt,
            mp_codecCtx->time_base,
            mp_videoStream->time_base);
        mp_pkt->stream_index = mp_videoStream->index;  // should be 0

        DEBUG("videofile: Writing packet %ld (size = %d)\n",
            mp_pkt->pts, mp_pkt->size);
        av_interleaved_write_frame(mp_fmtCtx, mp_pkt);
        av_packet_unref(mp_pkt);
    }
}


void VideoFile::pushFrame()
{
    DEBUG_FILELOC();

    checkLibVAError(
        av_frame_make_writable(mp_frameYUV)
    );

    // translate RGB frame to YUV frame
    // note: this isn't actually scaling anything, just encoding RGB to YUV
    sws_scale(mp_swsCtx,
        mp_frameRGB->data, mp_frameRGB->linesize,
        0, m_height,
        mp_frameYUV->data, mp_frameYUV->linesize);

    mp_frameYUV->pts = m_frameIdx;
    encode(mp_frameYUV);

    // // reset for next frame draw
    // // memset(p_pixelBuffer, 0x00, 3*m_nPix*sizeof(uint8_t));
    checkLibVAError(
        av_frame_make_writable(mp_frameRGB)
    );
    m_frameIdx++;
}


void VideoFile::pixel(
    const int32_t x, const int32_t y, const Color c)
{
    if (x < 0) return;
    else if ((uint32_t)x >= m_width) return;
    if (y < 0) return;
    else if ((uint32_t)y >= m_height) return;

    const uint64_t i = (uint64_t)y * m_width + x;
    mp_frameRGB->data[0][4 * i + 0] = c.r();
    mp_frameRGB->data[0][4 * i + 1] = c.g();
    mp_frameRGB->data[0][4 * i + 2] = c.b();
    mp_frameRGB->data[0][4 * i + 3] = c.a();
}

