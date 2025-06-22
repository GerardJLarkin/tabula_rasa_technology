// pi1_capture.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zmq.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <stdint.h>
#include <alsa/asoundlib.h>

#define FEATURE_DIM 40
typedef struct {
    int frame_index;
    float features[FEATURE_DIM];
} SnapshotFeature;

// --- Audio Capture via ALSA (persistent) ---
static snd_pcm_t *audio_capture_handle = NULL;
#define AUDIO_SAMPLE_RATE 44100
#define FRAME_PERIOD_SECONDS (1.0/30.0)

int init_audio_capture(void) {
    int err;
    snd_pcm_hw_params_t *hw_params;
    if ((err = snd_pcm_open(&audio_capture_handle, "default", SND_PCM_STREAM_CAPTURE, 0)) < 0) {
        fprintf(stderr, "cannot open audio device (%s)\n", snd_strerror(err));
        return err;
    }
    snd_pcm_hw_params_malloc(&hw_params);
    snd_pcm_hw_params_any(audio_capture_handle, hw_params);
    if ((err = snd_pcm_hw_params_set_access(audio_capture_handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED)) < 0) {
        fprintf(stderr, "cannot set access type (%s)\n", snd_strerror(err));
        snd_pcm_hw_params_free(hw_params);
        snd_pcm_close(audio_capture_handle);
        return err;
    }
    if ((err = snd_pcm_hw_params_set_format(audio_capture_handle, hw_params, SND_PCM_FORMAT_S16_LE)) < 0) {
        fprintf(stderr, "cannot set sample format (%s)\n", snd_strerror(err));
        snd_pcm_hw_params_free(hw_params);
        snd_pcm_close(audio_capture_handle);
        return err;
    }
    unsigned int rate = AUDIO_SAMPLE_RATE;
    int dir = 0;
    if ((err = snd_pcm_hw_params_set_rate_near(audio_capture_handle, hw_params, &rate, &dir)) < 0) {
        fprintf(stderr, "cannot set sample rate (%s)\n", snd_strerror(err));
        snd_pcm_hw_params_free(hw_params);
        snd_pcm_close(audio_capture_handle);
        return err;
    }
    if ((err = snd_pcm_hw_params_set_channels(audio_capture_handle, hw_params, 1)) < 0) {
        fprintf(stderr, "cannot set channel count (%s)\n", snd_strerror(err));
        snd_pcm_hw_params_free(hw_params);
        snd_pcm_close(audio_capture_handle);
        return err;
    }
    if ((err = snd_pcm_hw_params(audio_capture_handle, hw_params)) < 0) {
        fprintf(stderr, "cannot set parameters (%s)\n", snd_strerror(err));
        snd_pcm_hw_params_free(hw_params);
        snd_pcm_close(audio_capture_handle);
        return err;
    }
    snd_pcm_hw_params_free(hw_params);
    if ((err = snd_pcm_prepare(audio_capture_handle)) < 0) {
        fprintf(stderr, "cannot prepare audio interface (%s)\n", snd_strerror(err));
        snd_pcm_close(audio_capture_handle);
        return err;
    }
    return 0;
}

void close_audio_capture(void) {
    if (audio_capture_handle) {
        snd_pcm_close(audio_capture_handle);
        audio_capture_handle = NULL;
    }
}

int read_audio_samples(int16_t *buffer, int sample_count) {
    int err = snd_pcm_readi(audio_capture_handle, buffer, sample_count);
    if (err < 0) {
        err = snd_pcm_recover(audio_capture_handle, err, 0);
    }
    return err;
}

// --- Feature Extraction Stub ---
// In your final system, you'd process video, audio, and sensor data into a feature vector.
// Here we simulate by generating random features.
void compute_feature_vector(SnapshotFeature *feat, int frame_index) {
    feat->frame_index = frame_index;
    for (int i = 0; i < FEATURE_DIM; i++) {
        feat->features[i] = (float)rand()/(float)RAND_MAX;
    }
}

// --- Main: Capture, Process, and Stream via ZeroMQ ---
#include <zmq.h>
int main(void) {
    // Initialize ALSA audio capture.
    if (init_audio_capture() < 0) {
        fprintf(stderr, "Audio capture initialization failed.\n");
        exit(EXIT_FAILURE);
    }
    
    // Initialize ZeroMQ context and publisher.
    void *context = zmq_ctx_new();
    void *publisher = zmq_socket(context, ZMQ_PUB);
    if (zmq_bind(publisher, "tcp://*:5555") != 0) {
        perror("zmq_bind");
        exit(EXIT_FAILURE);
    }
    
    int frame_index = 0;
    int audio_sample_count = (int)(AUDIO_SAMPLE_RATE * FRAME_PERIOD_SECONDS);
    int16_t *audio_buffer = malloc(audio_sample_count * sizeof(int16_t));
    if (!audio_buffer) {
        perror("Allocating audio buffer");
        exit(EXIT_FAILURE);
    }
    
    // For demonstration, we simulate a loop at 30 fps.
    while (1) {
        SnapshotFeature feat;
        // Simulate capturing video, audio, and sensor data then compute feature vector.
        // (Replace with real capture code as needed.)
        // For audio, read samples.
        read_audio_samples(audio_buffer, audio_sample_count);
        // (Similarly, sensor data would be captured here.)
        compute_feature_vector(&feat, frame_index);
        
        // (Optionally, you can incorporate audio and sensor info into the feature vector.)
        
        // Send the feature vector as a binary message.
        if (zmq_send(publisher, &feat, sizeof(feat), 0) == -1) {
            perror("zmq_send");
        } else {
            printf("Sent snapshot frame %d\n", frame_index);
        }
        
        frame_index++;
        usleep(33333); // Approximately 30 fps.
    }
    
    free(audio_buffer);
    zmq_close(publisher);
    zmq_ctx_destroy(context);
    close_audio_capture();
    return 0;
}
