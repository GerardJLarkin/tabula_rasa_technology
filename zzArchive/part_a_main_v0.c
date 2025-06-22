#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#include <omp.h>
#include <emmintrin.h>
#include <alsa/asoundlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CLEAR(x) memset(&(x), 0, sizeof(x))

// Audio parameters
#define AUDIO_SAMPLE_RATE 44100
#define FRAME_PERIOD_SECONDS (1.0/30.0)

#define MAX_SNAPSHOTS 900

Snapshot *snapshot_deque[MAX_SNAPSHOTS];
int snapshot_count = 0;  // number of snapshots currently stored (<= MAX_SNAPSHOTS)
int snapshot_head = 0;   // index of the oldest snapshot

// Global frame index
static int global_frame_index = 0;

// Persistent audio capture handle
static snd_pcm_t *audio_capture_handle = NULL;

// -------------------------------------------------------------------------
// Snapshot Data Structure
// -------------------------------------------------------------------------
typedef struct {
    int frame_index;
    int num_groups;
    int *group_ids;     // Valid group IDs (with >=10 pixels)
    int *group_counts;  // Pixel counts for valid groups
    int audio_sample_count;
    int16_t *audio_samples; // Audio samples for this snapshot
    float gyro[3];      // Gyroscope data: x,y,z
    float acc[3];       // Accelerometer data: x,y,z
} Snapshot;

// -------------------------------------------------------------------------
// ALSA Audio Capture Functions (Persistent Device)
// -------------------------------------------------------------------------
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
    
    if ((err = snd_pcm_hw_params_set_channels(audio_capture_handle, hw_params, 1)) < 0) {  // Mono
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

// -------------------------------------------------------------------------
// SIMD/Vectorized YUYV to RGB Conversion
// -------------------------------------------------------------------------
static inline void convert_yuyv_to_rgb_simd(const unsigned char *yuyv, unsigned int *rgb, int numPixels) {
    for (int i = 0; i < numPixels; i += 2) {
        int j = i * 2; // 4 bytes per 2 pixels
        int y0 = yuyv[j] - 16;
        int u  = yuyv[j+1] - 128;
        int y1 = yuyv[j+2] - 16;
        int v  = yuyv[j+3] - 128;
        int c0 = 298 * y0;
        int c1 = 298 * y1;
        int r0 = (c0 + 409 * v + 128) >> 8;
        int g0 = (c0 - 100 * u - 208 * v + 128) >> 8;
        int b0 = (c0 + 516 * u + 128) >> 8;
        int r1 = (c1 + 409 * v + 128) >> 8;
        int g1 = (c1 - 100 * u - 208 * v + 128) >> 8;
        int b1 = (c1 + 516 * u + 128) >> 8;
        r0 = r0 < 0 ? 0 : (r0 > 255 ? 255 : r0);
        g0 = g0 < 0 ? 0 : (g0 > 255 ? 255 : g0);
        b0 = b0 < 0 ? 0 : (b0 > 255 ? 255 : b0);
        r1 = r1 < 0 ? 0 : (r1 > 255 ? 255 : r1);
        g1 = g1 < 0 ? 0 : (g1 > 255 ? 255 : g1);
        b1 = b1 < 0 ? 0 : (b1 > 255 ? 255 : b1);
        rgb[i]   = (r0 << 16) | (g0 << 8) | b0;
        rgb[i+1] = (r1 << 16) | (g1 << 8) | b1;
    }
}

// -------------------------------------------------------------------------
// Basic Helper Functions: Color Distance and Intensity
// -------------------------------------------------------------------------
static inline int color_distance_squared(unsigned int color1, unsigned int color2) {
    int r1 = (color1 >> 16) & 0xFF, g1 = (color1 >> 8) & 0xFF, b1 = color1 & 0xFF;
    int r2 = (color2 >> 16) & 0xFF, g2 = (color2 >> 8) & 0xFF, b2 = color2 & 0xFF;
    int dr = r1 - r2, dg = g1 - g2, db = b1 - b2;
    return dr*dr + dg*dg + db*db;
}

static inline int get_intensity(unsigned int color) {
    int r = (color >> 16) & 0xFF, g = (color >> 8) & 0xFF, b = color & 0xFF;
    return (r + g + b) / 3;
}

// -------------------------------------------------------------------------
// Function: identify_neighbors_within_threshold
// -------------------------------------------------------------------------
void identify_neighbors_within_threshold(const unsigned int *color_array, int width, int height, int threshold, unsigned char *neighbor_mask_array) {
    int num_pixels = width * height;
    for (int i = 0; i < num_pixels; i++){
        int x = i % width;
        int y = i / width;
        unsigned int center_color = color_array[i];
        unsigned char mask = 0;
        for (int dy = -1; dy <= 1; dy++){
            for (int dx = -1; dx <= 1; dx++){
                if (dx == 0 && dy == 0)
                    continue;
                int nx = x + dx, ny = y + dy;
                if (nx < 0 || nx >= width || ny < 0 || ny >= height)
                    continue;
                int neighbor_index = ny * width + nx;
                unsigned int neighbor_color = color_array[neighbor_index];
                int diff_sq = color_distance_squared(center_color, neighbor_color);
                if (diff_sq < threshold) {
                    int bit = -1;
                    if      (dx == -1 && dy == -1) bit = 0;
                    else if (dx ==  0 && dy == -1) bit = 1;
                    else if (dx ==  1 && dy == -1) bit = 2;
                    else if (dx == -1 && dy ==  0) bit = 3;
                    else if (dx ==  1 && dy ==  0) bit = 4;
                    else if (dx == -1 && dy ==  1) bit = 5;
                    else if (dx ==  0 && dy ==  1) bit = 6;
                    else if (dx ==  1 && dy ==  1) bit = 7;
                    if (bit >= 0)
                        mask |= (1 << bit);
                }
            }
        }
        neighbor_mask_array[i] = mask;
    }
}

// -------------------------------------------------------------------------
// Optimized Connected-Component Labeling Using Union-Find
// -------------------------------------------------------------------------
static inline int find_root(int *parent, int i) {
    while (parent[i] != i) {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    return i;
}

static inline void union_labels(int *parent, int i, int j) {
    int root_i = find_root(parent, i);
    int root_j = find_root(parent, j);
    if (root_i < root_j)
        parent[root_j] = root_i;
    else if (root_i > root_j)
        parent[root_i] = root_j;
}

int* label_connected_components_union_find(const unsigned int *color_array, int width, int height, int threshold) {
    int size = width * height;
    int *labels = malloc(size * sizeof(int));
    int *parent = malloc(size * sizeof(int));
    if (!labels || !parent) {
        perror("Allocating union-find arrays");
        free(labels);
        free(parent);
        return NULL;
    }
    for (int i = 0; i < size; i++) {
        labels[i] = i;
        parent[i] = i;
    }
    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){
            int idx = i * width + j;
            int curr_intensity = get_intensity(color_array[idx]);
            if (j > 0) {
                int left_idx = idx - 1;
                int left_intensity = get_intensity(color_array[left_idx]);
                if (abs(curr_intensity - left_intensity) <= threshold)
                    union_labels(parent, idx, left_idx);
            }
            if (i > 0) {
                int top_idx = idx - width;
                int top_intensity = get_intensity(color_array[top_idx]);
                if (abs(curr_intensity - top_intensity) <= threshold)
                    union_labels(parent, idx, top_idx);
            }
            if (i > 0 && j > 0) {
                int tl_idx = idx - width - 1;
                int tl_intensity = get_intensity(color_array[tl_idx]);
                if (abs(curr_intensity - tl_intensity) <= threshold)
                    union_labels(parent, idx, tl_idx);
            }
            if (i > 0 && j < width - 1) {
                int tr_idx = idx - width + 1;
                int tr_intensity = get_intensity(color_array[tr_idx]);
                if (abs(curr_intensity - tr_intensity) <= threshold)
                    union_labels(parent, idx, tr_idx);
            }
        }
    }
    for (int i = 0; i < size; i++){
        labels[i] = find_root(parent, i);
    }
    free(parent);
    return labels;
}

// -------------------------------------------------------------------------
// Stub for Gyroscope/Accelerometer Input
// -------------------------------------------------------------------------
void read_gyro_acc(float *gyro, float *acc) {
    gyro[0] = 0.1f; gyro[1] = 0.2f; gyro[2] = 0.3f;
    acc[0] = 9.8f; acc[1] = 0.0f; acc[2] = 0.0f;
}

// -------------------------------------------------------------------------
// Snapshot Creation: Combine Video Groups, Audio, and Sensor Data
// -------------------------------------------------------------------------
Snapshot create_snapshot(unsigned int *color_array, int width, int height, const int *labels) {
    Snapshot snap;
    snap.frame_index = global_frame_index;
    int num_pixels = width * height;
    int max_label = -1;
    for (int i = 0; i < num_pixels; i++) {
        if (labels[i] > max_label)
            max_label = labels[i];
    }
    int *all_group_counts = calloc(max_label + 1, sizeof(int));
    if (!all_group_counts) {
        perror("Allocating group_counts");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < num_pixels; i++) {
        all_group_counts[labels[i]]++;
    }
    int valid_count = 0;
    for (int g = 0; g <= max_label; g++) {
        if (all_group_counts[g] >= 10)
            valid_count++;
    }
    snap.num_groups = valid_count;
    snap.group_ids = malloc(valid_count * sizeof(int));
    snap.group_counts = malloc(valid_count * sizeof(int));
    if (!snap.group_ids || !snap.group_counts) {
        perror("Allocating snapshot group arrays");
        exit(EXIT_FAILURE);
    }
    int idx = 0;
    for (int g = 0; g <= max_label; g++) {
        if (all_group_counts[g] >= 10) {
            snap.group_ids[idx] = g;
            snap.group_counts[idx] = all_group_counts[g];
            idx++;
        }
    }
    free(all_group_counts);
    
    int audio_sample_count = (int)(AUDIO_SAMPLE_RATE * FRAME_PERIOD_SECONDS);
    snap.audio_sample_count = audio_sample_count;
    snap.audio_samples = malloc(audio_sample_count * sizeof(int16_t));
    if (!snap.audio_samples) {
        perror("Allocating audio buffer");
        exit(EXIT_FAILURE);
    }
    read_audio_samples(snap.audio_samples, audio_sample_count);
    
    read_gyro_acc(snap.gyro, snap.acc);
    
    return snap;
}

void free_snapshot(Snapshot *snap) {
    free(snap->group_ids);
    free(snap->group_counts);
    free(snap->audio_samples);
}

// -------------------------------------------------------------------------
// Process Frame: Create Snapshot and Print Summary
// -------------------------------------------------------------------------
void process_frame_example(unsigned int *color_array, int width, int height,
                           const int *labels, const unsigned char *neighbor_mask_array) {
    Snapshot snap = create_snapshot(color_array, width, height, labels);
    // printf("Snapshot for frame %d: %d valid groups, %d audio samples.\n",
    //        snap.frame_index, snap.num_groups, snap.audio_sample_count);
    // printf("Gyro: [%.2f, %.2f, %.2f] | Acc: [%.2f, %.2f, %.2f]\n",
    //        snap.gyro[0], snap.gyro[1], snap.gyro[2],
    //        snap.acc[0], snap.acc[1], snap.acc[2]);
    // for (int i = 0; i < snap.num_groups; i++) {
    //     printf("  Group %d has %d pixels.\n", snap.group_ids[i], snap.group_counts[i]);
    // }
    free_snapshot(&snap);
}

// -------------------------------------------------------------------------
// Capture Loop and Main - Integrating Video, Audio, and Sensor Snapshots
// -------------------------------------------------------------------------
struct buffer {
    void *start;
    size_t length;
};

typedef void (*frame_callback_t)(unsigned int *color_array, int width, int height,
                                  const int *labels, const unsigned char *neighbor_mask_array);

void capture_loop(frame_callback_t process_frame) {
    const char *dev_name = "/dev/video0";
    int fd = open(dev_name, O_RDWR);
    if (fd < 0) {
        perror("Opening video device");
        exit(EXIT_FAILURE);
    }
    struct v4l2_capability cap;
    if (ioctl(fd, VIDIOC_QUERYCAP, &cap) < 0) {
        perror("Querying Capabilities");
        close(fd);
        exit(EXIT_FAILURE);
    }
    struct v4l2_format fmt;
    CLEAR(fmt);
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = 640;
    fmt.fmt.pix.height = 480;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;
    if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
        perror("Setting Pixel Format");
        close(fd);
        exit(EXIT_FAILURE);
    }
    int width = fmt.fmt.pix.width, height = fmt.fmt.pix.height;
    struct v4l2_requestbuffers req;
    CLEAR(req);
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (ioctl(fd, VIDIOC_REQBUFS, &req) < 0) {
        perror("Requesting Buffer");
        close(fd);
        exit(EXIT_FAILURE);
    }
    struct buffer *buffers = calloc(req.count, sizeof(*buffers));
    if (!buffers) {
        perror("Out of memory");
        close(fd);
        exit(EXIT_FAILURE);
    }
    unsigned int n_buffers = 0;
    for (n_buffers = 0; n_buffers < req.count; n_buffers++) {
        struct v4l2_buffer buf;
        CLEAR(buf);
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = n_buffers;
        if (ioctl(fd, VIDIOC_QUERYBUF, &buf) < 0) {
            perror("Querying Buffer");
            close(fd);
            exit(EXIT_FAILURE);
        }
        buffers[n_buffers].length = buf.length;
        buffers[n_buffers].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
        if (buffers[n_buffers].start == MAP_FAILED) {
            perror("mmap");
            close(fd);
            exit(EXIT_FAILURE);
        }
    }
    for (unsigned int i = 0; i < n_buffers; i++) {
        struct v4l2_buffer buf;
        CLEAR(buf);
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        if (ioctl(fd, VIDIOC_QBUF, &buf) < 0) {
            perror("Queue Buffer");
            close(fd);
            exit(EXIT_FAILURE);
        }
    }
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_STREAMON, &type) < 0) {
        perror("Start Capture");
        close(fd);
        exit(EXIT_FAILURE);
    }
    
    // Initialize audio capture once.
    if (init_audio_capture() < 0) {
        fprintf(stderr, "Audio capture initialization failed.\n");
        exit(EXIT_FAILURE);
    }
    
    // Allocate frame buffers once.
    unsigned int *color_array = malloc(width * height * sizeof(unsigned int));
    unsigned char *neighbor_mask_array = malloc(width * height * sizeof(unsigned char));
    if (!color_array || !neighbor_mask_array) {
        perror("Allocating frame arrays");
        exit(EXIT_FAILURE);
    }
    
    while (1) {
        struct timespec loop_start, loop_end;
        clock_gettime(CLOCK_MONOTONIC, &loop_start);
        
        struct v4l2_buffer buf;
        CLEAR(buf);
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        if (ioctl(fd, VIDIOC_DQBUF, &buf) < 0) {
            perror("Retrieving Frame");
            break;
        }
        
        // Convert YUYV to RGB using SIMD conversion.
        convert_yuyv_to_rgb_simd((unsigned char*)buffers[buf.index].start, color_array, width * height);
        
        // Compute neighbor mask.
        int mask_threshold = 2500;
        identify_neighbors_within_threshold(color_array, width, height, mask_threshold, neighbor_mask_array);
        
        // Use union-find CCL.
        int group_threshold = 10;
        int *labels = label_connected_components_union_find(color_array, width, height, group_threshold);
        if (!labels) {
            free(color_array);
            free(neighbor_mask_array);
            break;
        }
        
        // Create snapshot and print summary.
        process_frame_example(color_array, width, height, labels, neighbor_mask_array);
        
        free(labels);
        
        if (ioctl(fd, VIDIOC_QBUF, &buf) < 0) {
            perror("Requeue Buffer");
            break;
        }
        
        clock_gettime(CLOCK_MONOTONIC, &loop_end);
        double elapsed = (loop_end.tv_sec - loop_start.tv_sec) + (loop_end.tv_nsec - loop_start.tv_nsec) / 1e9;
        printf("Iteration took %.6f seconds\n", elapsed);
        
        global_frame_index = (global_frame_index + 1) % 900;
    }
    
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_STREAMOFF, &type) < 0)
        perror("Stop Capture");
    
    for (unsigned int i = 0; i < n_buffers; i++)
        munmap(buffers[i].start, buffers[i].length);
    free(buffers);
    free(color_array);
    free(neighbor_mask_array);
    close_audio_capture();
    close(fd);
}

int main(void) {
    capture_loop(process_frame_example);
    return 0;
}
