/*
 * pi1_capture.c
 *
 * This program runs on Raspberry Pi 1 (Part A). It captures data (video, audio,
 * and sensor data – here audio is captured using ALSA and video/sensor data are simulated),
 * computes a feature vector (SnapshotFeature), and publishes the feature vector to a remote
 * endpoint via ZeroMQ.
 *
 * Compile with:
 * gcc -O3 -march=native -fopenmp -o pi1_capture pi1_capture.c -lm -lasound -lzmq
 */

#include <stdio.h>            // Standard I/O functions (printf, fprintf, etc.)
#include <stdlib.h>           // Standard library functions (malloc, free, exit)
#include <string.h>           // String functions (memcpy, memset)
#include <zmq.h>              // ZeroMQ messaging API
#include <time.h>             // Time functions (clock_gettime, usleep)
#include <unistd.h>           // Unix standard functions (usleep)
#include <math.h>             // Math functions (fabs, pow, etc.)
#include <stdint.h>           // Standard integer types (int16_t)
#include <alsa/asoundlib.h>   // ALSA library for audio capture

// Define the dimension of our feature vector.
#define FEATURE_DIM 40

// Define the frame period based on 30 frames per second.
#define FRAME_PERIOD_SECONDS (1.0/30.0)

// SnapshotFeature: Structure to hold one snapshot's feature vector.
typedef struct {
    int frame_index;              // The frame index associated with this snapshot.
    float features[FEATURE_DIM];  // A fixed-length feature vector (computed from video, audio, sensor data).
} SnapshotFeature;

// ---------- ALSA Audio Capture Functions ----------

// Global variable to hold the ALSA audio capture handle.
static snd_pcm_t *audio_capture_handle = NULL;

// init_audio_capture: Initializes the default audio capture device.
// It sets up the device for interleaved, 16-bit little-endian, mono capture.
int init_audio_capture(void) {
    int err;  // Variable to store error codes.
    snd_pcm_hw_params_t *hw_params;  // Pointer to hardware parameter structure.
    
    // Open the default audio device in capture mode.
    if ((err = snd_pcm_open(&audio_capture_handle, "default", SND_PCM_STREAM_CAPTURE, 0)) < 0) {
        fprintf(stderr, "cannot open audio device (%s)\n", snd_strerror(err));
        return err;
    }
    
    // Allocate the hardware parameter structure.
    snd_pcm_hw_params_malloc(&hw_params);
    // Initialize the hw_params with default values.
    snd_pcm_hw_params_any(audio_capture_handle, hw_params);
    
    // Set the access type to interleaved (samples are interleaved if multi-channel).
    if ((err = snd_pcm_hw_params_set_access(audio_capture_handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED)) < 0) {
        fprintf(stderr, "cannot set access type (%s)\n", snd_strerror(err));
        snd_pcm_hw_params_free(hw_params);
        snd_pcm_close(audio_capture_handle);
        return err;
    }
    
    // Set the audio format to 16-bit little-endian.
    if ((err = snd_pcm_hw_params_set_format(audio_capture_handle, hw_params, SND_PCM_FORMAT_S16_LE)) < 0) {
        fprintf(stderr, "cannot set sample format (%s)\n", snd_strerror(err));
        snd_pcm_hw_params_free(hw_params);
        snd_pcm_close(audio_capture_handle);
        return err;
    }
    
    // Set the sample rate near the defined AUDIO_SAMPLE_RATE.
    unsigned int rate = 44100;  // We'll use 44100 Hz as our sample rate.
    int dir = 0;
    if ((err = snd_pcm_hw_params_set_rate_near(audio_capture_handle, hw_params, &rate, &dir)) < 0) {
        fprintf(stderr, "cannot set sample rate (%s)\n", snd_strerror(err));
        snd_pcm_hw_params_free(hw_params);
        snd_pcm_close(audio_capture_handle);
        return err;
    }
    
    // Set the number of channels to 1 (mono).
    if ((err = snd_pcm_hw_params_set_channels(audio_capture_handle, hw_params, 1)) < 0) {
        fprintf(stderr, "cannot set channel count (%s)\n", snd_strerror(err));
        snd_pcm_hw_params_free(hw_params);
        snd_pcm_close(audio_capture_handle);
        return err;
    }
    
    // Apply the parameters to the audio device.
    if ((err = snd_pcm_hw_params(audio_capture_handle, hw_params)) < 0) {
        fprintf(stderr, "cannot set parameters (%s)\n", snd_strerror(err));
        snd_pcm_hw_params_free(hw_params);
        snd_pcm_close(audio_capture_handle);
        return err;
    }
    
    // Free the hardware parameter structure as we no longer need it.
    snd_pcm_hw_params_free(hw_params);
    
    // Prepare the audio interface to start capturing.
    if ((err = snd_pcm_prepare(audio_capture_handle)) < 0) {
        fprintf(stderr, "cannot prepare audio interface (%s)\n", snd_strerror(err));
        snd_pcm_close(audio_capture_handle);
        return err;
    }
    return 0;
}

// close_audio_capture: Closes the audio capture device.
void close_audio_capture(void) {
    if (audio_capture_handle) {
        snd_pcm_close(audio_capture_handle);
        audio_capture_handle = NULL;
    }
}

// read_audio_samples: Reads a specified number of audio samples (frames) into a buffer.
int read_audio_samples(int16_t *buffer, int sample_count) {
    int err = snd_pcm_readi(audio_capture_handle, buffer, sample_count);
    if (err < 0) {
        // Attempt to recover from an error.
        err = snd_pcm_recover(audio_capture_handle, err, 0);
    }
    return err;
}

// ---------- Feature Extraction ----------
// compute_feature_vector: This function simulates computing a feature vector for a snapshot.
// In a complete implementation, you would process your captured video, audio, and sensor data.
void compute_feature_vector(SnapshotFeature *feat, int frame_index) {
    feat->frame_index = frame_index;  // Save the frame index.
    for (int i = 0; i < FEATURE_DIM; i++) {
        // Fill each feature with a random float value between 0 and 1.
        feat->features[i] = (float)rand() / (float)RAND_MAX;
    }
}

// ---------- Main: Data Capture and Publishing ----------

#include <zmq.h>  // ZeroMQ for networking.

int main(void) {
    // Initialize the ALSA audio capture device.
    if (init_audio_capture() < 0) {
        fprintf(stderr, "Audio capture initialization failed.\n");
        exit(EXIT_FAILURE);
    }
    
    // Create a ZeroMQ context.
    void *context = zmq_ctx_new();
    // Create a ZeroMQ publisher socket.
    void *publisher = zmq_socket(context, ZMQ_PUB);
    // Bind the publisher on TCP port 5555 so other devices can connect.
    if (zmq_bind(publisher, "tcp://*:5555") != 0) {
        perror("zmq_bind");
        exit(EXIT_FAILURE);
    }
    
    int frame_index = 0; // Initialize frame index.
    // Calculate the number of audio samples to capture per frame based on the sample rate and frame period.
    int audio_sample_count = (int)(44100 * FRAME_PERIOD_SECONDS);
    // Allocate a buffer to hold audio samples.
    int16_t *audio_buffer = malloc(audio_sample_count * sizeof(int16_t));
    if (!audio_buffer) {
        perror("Allocating audio buffer");
        exit(EXIT_FAILURE);
    }
    
    // Main capture loop: target approximately 30 frames per second.
    while (1) {
        SnapshotFeature feat;  // Create a feature vector structure.
        
        // Capture audio samples during the current frame period.
        read_audio_samples(audio_buffer, audio_sample_count);
        
        // (In a full implementation, you would also capture and process video and sensor data here.)
        // For this demonstration, we simulate feature extraction.
        compute_feature_vector(&feat, frame_index);
        
        // (Optionally, you could incorporate audio and sensor information into feat.features.)
        
        // Publish the feature vector as a binary message to connected subscribers.
        if (zmq_send(publisher, &feat, sizeof(feat), 0) == -1) {
            perror("zmq_send");
        } else {
            printf("Sent snapshot frame %d\n", frame_index);
        }
        
        frame_index++;         // Increment frame index.
        usleep(33333);         // Sleep ~33.33 ms to maintain ~30 frames per second.
    }
    
    // Clean up allocated resources.
    free(audio_buffer);
    zmq_close(publisher);
    zmq_ctx_destroy(context);
    close_audio_capture();
    
    return 0;
}
