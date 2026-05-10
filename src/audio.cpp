#include "audio.h"
#include <cstdio>

// Number of simultaneously-playing voices. Each voice is one SDL_AudioStream
// bound to the default playback device; SDL mixes their outputs for us.
//
// 8 is comfortably above what point-and-click navigation needs (transition
// whoosh, click feedback, maybe an ambient one-shot or two). If we ever
// actually run out, sfx_play() will steal a voice -- see comment there.
#define AUDIO_VOICE_COUNT 8

static SDL_AudioStream* g_voices[AUDIO_VOICE_COUNT];
static bool             g_audio_ready = false;

bool audio_init(void) {
    // SDL_INIT_AUDIO must already have been requested by the caller. If it
    // wasn't, SDL_OpenAudioDeviceStream below will fail and we bail.
    for (int i = 0; i < AUDIO_VOICE_COUNT; i++) {
        // spec   = NULL  -> the stream's *output* side adopts the device's
        //                   native format; per-Sfx input formats are set later
        //                   in sfx_play() and SDL converts on the fly.
        // cb     = NULL  -> pull mode; we just queue PCM with
        //                   SDL_PutAudioStreamData and SDL drains it on its
        //                   own audio thread. No callback to write.
        g_voices[i] = SDL_OpenAudioDeviceStream(
            SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, NULL, NULL, NULL);
        if (!g_voices[i]) {
            fprintf(stderr,
                    "audio_init: SDL_OpenAudioDeviceStream voice %d failed: %s\n",
                    i, SDL_GetError());
            // Tear down the streams we already opened so we don't leak.
            for (int j = 0; j < i; j++) {
                SDL_DestroyAudioStream(g_voices[j]);
                g_voices[j] = NULL;
            }
            return false;
        }
        // Streams returned by SDL_OpenAudioDeviceStream start *paused* on
        // their logical device. Resume so queued data actually plays.
        SDL_ResumeAudioStreamDevice(g_voices[i]);
    }
    g_audio_ready = true;
    return true;
}

void audio_shutdown(void) {
    for (int i = 0; i < AUDIO_VOICE_COUNT; i++) {
        if (g_voices[i]) {
            // Destroying the stream also unbinds it and closes the underlying
            // logical device opened by SDL_OpenAudioDeviceStream.
            SDL_DestroyAudioStream(g_voices[i]);
            g_voices[i] = NULL;
        }
    }
    g_audio_ready = false;
}

bool sfx_load(Sfx* out, const char* wav_path) {
    *out = (Sfx){};
    if (!SDL_LoadWAV(wav_path, &out->spec, &out->pcm, &out->bytes)) {
        fprintf(stderr, "sfx_load(%s): %s\n", wav_path, SDL_GetError());
        return false;
    }
    return true;
}

void sfx_free(Sfx* s) {
    if (!s) return;
    if (s->pcm) {
        SDL_free(s->pcm);   // matches the allocator used by SDL_LoadWAV
        s->pcm = NULL;
    }
    s->bytes = 0;
}

void sfx_play(const Sfx* s, float volume) {
    if (!g_audio_ready || !s || !s->pcm || s->bytes == 0) return;

    // Pick a voice. Preference order:
    //   1. The first stream whose queue is empty (truly free).
    //   2. Otherwise steal the stream with the *most* queued bytes. This is
    //      a crude heuristic -- it doesn't know which voice has been playing
    //      longest, only which has the most audio left to play -- but it's
    //      cheap and good enough for a handful of voices. Stealing is audible
    //      (the stolen sound cuts off mid-sample); raise AUDIO_VOICE_COUNT
    //      if it becomes a problem.
    int chosen = 0;
    int max_queued = -1;
    for (int i = 0; i < AUDIO_VOICE_COUNT; i++) {
        int q = (int)SDL_GetAudioStreamQueued(g_voices[i]);
        if (q == 0) { chosen = i; max_queued = 0; break; }
        if (q > max_queued) { max_queued = q; chosen = i; }
    }
    SDL_AudioStream* v = g_voices[chosen];

    // Tell the stream the *input* format for this batch of samples. The
    // output side stays at the device's native format, so SDL will resample
    // / re-channelize on the fly. Passing NULL for the second arg means
    // "leave the output spec alone".
    SDL_SetAudioStreamFormat(v, &s->spec, NULL);
    SDL_SetAudioStreamGain(v, volume);
    SDL_ClearAudioStream(v);                       // drop any stolen leftovers
    SDL_PutAudioStreamData(v, s->pcm, (int)s->bytes);
}

void sfx_stop_all(void) {
    if (!g_audio_ready) return;
    for (int i = 0; i < AUDIO_VOICE_COUNT; i++) {
        SDL_ClearAudioStream(g_voices[i]);
    }
}
