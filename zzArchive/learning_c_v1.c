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

#define CLEAR(x) memset(&(x), 0, sizeof(x))

// Global frame index; resets to 0 after 900 frames.
static int global_frame_index = 0;
// Global file counter for naming output files (0 to 1000000).
static unsigned long file_counter = 0;

// -------------------------------------------------------------------------
// PART 1: Helper Functions for Color & Neighbors
// -------------------------------------------------------------------------

// Compute squared Euclidean distance between two 0xRRGGBB colors.
static inline int color_distance_squared(unsigned int color1, unsigned int color2) {
    int r1 = (color1 >> 16) & 0xFF;
    int g1 = (color1 >> 8)  & 0xFF;
    int b1 = color1 & 0xFF;
    int r2 = (color2 >> 16) & 0xFF;
    int g2 = (color2 >> 8)  & 0xFF;
    int b2 = color2 & 0xFF;
    int dr = r1 - r2, dg = g1 - g2, db = b1 - b2;
    return dr * dr + dg * dg + db * db;
}

/**
 * For each pixel in color_array (packed 0xRRGGBB), check its 8-connected neighbors
 * (in the 3x3 window) and set an 8-bit mask if the neighbor’s color difference is below threshold.
 *
 * Bit mapping:
 *   Bit 0: top-left, Bit 1: top, Bit 2: top-right, Bit 3: left,
 *   Bit 4: right, Bit 5: bottom-left, Bit 6: bottom, Bit 7: bottom-right.
 */
void identify_neighbors_within_threshold(const unsigned int *color_array, int width, int height, int threshold, unsigned char *neighbor_mask_array) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int index = y * width + x;
            unsigned int center_color = color_array[index];
            unsigned char mask = 0;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0)
                        continue; // skip center pixel
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
            neighbor_mask_array[index] = mask;
        }
    }
}

// -------------------------------------------------------------------------
// PART 2: Connected-Component Grouping (Flood Fill)
// -------------------------------------------------------------------------

// Compute intensity as the average of R, G, B.
static inline int get_intensity(unsigned int color) {
    int r = (color >> 16) & 0xFF, g = (color >> 8) & 0xFF, b = color & 0xFF;
    return (r + g + b) / 3;
}

/**
 * Label connected components based on pixel intensity.
 * Two pixels are connected if they are 8-connected and the absolute difference of intensities is ≤ threshold.
 * Returns a dynamically allocated label array (size width*height). Caller must free it.
 */
int* label_connected_components(const unsigned int *color_array, int width, int height, int threshold) {
    int size = width * height;
    int *labels = malloc(size * sizeof(int));
    if (!labels) {
        perror("Allocating labels");
        return NULL;
    }
    for (int i = 0; i < size; i++)
        labels[i] = -1;

    int *stack = malloc(size * sizeof(int));
    if (!stack) {
        perror("Allocating stack");
        free(labels);
        return NULL;
    }
    int stack_top = 0, current_label = 0;

    for (int i = 0; i < size; i++) {
        if (labels[i] != -1)
            continue;
        labels[i] = current_label;
        stack[stack_top++] = i;

        while (stack_top > 0) {
            int index = stack[--stack_top];
            int x = index % width;
            int y = index / width;
            int curr_intensity = get_intensity(color_array[index]);
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0)
                        continue;
                    int nx = x + dx, ny = y + dy;
                    if (nx < 0 || nx >= width || ny < 0 || ny >= height)
                        continue;
                    int n_index = ny * width + nx;
                    if (labels[n_index] != -1)
                        continue;
                    int n_intensity = get_intensity(color_array[n_index]);
                    if (abs(curr_intensity - n_intensity) <= threshold) {
                        labels[n_index] = current_label;
                        stack[stack_top++] = n_index;
                    }
                }
            }
        }
        current_label++;
    }
    free(stack);
    return labels;
}

// -------------------------------------------------------------------------
// PART 3: Direct Binary File Output for Each Group (with Normalization & Segmentation)
// -------------------------------------------------------------------------

/**
 * Save an individual group’s pixel data to a binary file.
 *
 * The file header consists of:
 *    - frame index (int)
 *    - group id (int)
 *    - number of pixels in this group (int)
 *
 * Each pixel record then contains the following fields:
 *    - normalized x coordinate (float) in [-1, 1]
 *    - normalized y coordinate (float) in [-1, 1]
 *    - pixel color (unsigned int)
 *    - neighbor mask (unsigned char)
 *    - segment index (int) [0-7] based on the unit circle split into 8 segments
 *    - segment count (int) for that segment in the group
 */
void save_group_to_binary(const char *filename, int frame_index, int group_id, int num_pixels,
                          int width, int height, const unsigned int *color_array,
                          const int *labels, const unsigned char *neighbor_mask_array) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        perror("Failed to open file for binary output");
        return;
    }
    
    // Write header: frame index, group id, and number of pixels.
    if (fwrite(&frame_index, sizeof(frame_index), 1, fp) != 1 ||
        fwrite(&group_id, sizeof(group_id), 1, fp) != 1 ||
        fwrite(&num_pixels, sizeof(num_pixels), 1, fp) != 1) {
        perror("Failed to write group header");
        fclose(fp);
        return;
    }
    
    // First pass: Compute segment counts for this group.
    int seg_counts[8] = {0};
    for (int i = 0; i < width * height; i++) {
        if (labels[i] == group_id) {
            int x = i % width;
            int y = i / width;
            float norm_x = 2.0f * x / (width - 1) - 1.0f;
            float norm_y = 2.0f * y / (height - 1) - 1.0f;
            double angle = atan2(norm_y, norm_x);
            int seg = (int)floor((angle + M_PI) / (M_PI / 4));
            if (seg < 0) seg = 0;
            if (seg > 7) seg = 7;
            seg_counts[seg]++;
        }
    }
    
    // Second pass: Write pixel records.
    // Each record: norm_x (float), norm_y (float), pixel color (unsigned int),
    // neighbor mask (unsigned char), segment index (int), segment count (int)
    for (int i = 0; i < width * height; i++) {
        if (labels[i] == group_id) {
            int x = i % width;
            int y = i / width;
            float norm_x = 2.0f * x / (width - 1) - 1.0f;
            float norm_y = 2.0f * y / (height - 1) - 1.0f;
            double angle = atan2(norm_y, norm_x);
            int seg = (int)floor((angle + M_PI) / (M_PI / 4));
            if (seg < 0) seg = 0;
            if (seg > 7) seg = 7;
            int seg_count = seg_counts[seg];
            
            if (fwrite(&norm_x, sizeof(float), 1, fp) != 1 ||
                fwrite(&norm_y, sizeof(float), 1, fp) != 1 ||
                fwrite(&color_array[i], sizeof(unsigned int), 1, fp) != 1 ||
                fwrite(&neighbor_mask_array[i], sizeof(unsigned char), 1, fp) != 1 ||
                fwrite(&seg, sizeof(int), 1, fp) != 1 ||
                fwrite(&seg_count, sizeof(int), 1, fp) != 1) {
                perror("Failed to write pixel record");
                fclose(fp);
                return;
            }
        }
    }
    
    fclose(fp);
}

// -------------------------------------------------------------------------
// PART 4: Capture Loop and Frame Processing
// -------------------------------------------------------------------------

// Structure for memory-mapped buffers.
struct buffer {
    void   *start;
    size_t length;
};

// Frame callback signature updated to include neighbor_mask_array.
typedef void (*frame_callback_t)(unsigned int *color_array, int width, int height,
                                 const int *labels, const unsigned char *neighbor_mask_array);

/**
 * Process a frame: compute connected groups and, for each group with at least 10 pixels,
 * save the group's pixel data (with normalized coordinates, segmentation, and segment counts)
 * to a separate binary file.
 */
void process_frame_example(unsigned int *color_array, int width, int height,
                           const int *labels, const unsigned char *neighbor_mask_array) {
    int num_pixels = width * height;
    int max_label = -1;
    // Determine the maximum group id.
    for (int i = 0; i < num_pixels; i++) {
        if (labels[i] > max_label)
            max_label = labels[i];
    }
    
    // For each group, count the number of pixels and save the group's data if it has at least 10 pixels.
    for (int group = 0; group <= max_label; group++) {
        int count = 0;
        for (int i = 0; i < num_pixels; i++) {
            if (labels[i] == group)
                count++;
        }
        if (count < 10)  // Skip groups with fewer than 10 pixels.
            continue;
        
        char filename[128];
        // Generate file name using the global file counter in the folder "c_data"
        snprintf(filename, sizeof(filename), "c_data/%06lu.bin", file_counter++);
        save_group_to_binary(filename, global_frame_index, group, count, width, height,
                             color_array, labels, neighbor_mask_array);
    }
}

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
        
        // Allocate and fill a color array from the YUYV data.
        unsigned int *color_array = malloc(width * height * sizeof(unsigned int));
        if (!color_array) {
            perror("Allocating color array");
            break;
        }
        unsigned char *yuyv = buffers[buf.index].start;
        int pixel_index = 0;
        for (int i = 0; i < width * height * 2; i += 4) {
            unsigned char y0 = yuyv[i], u = yuyv[i+1], y1 = yuyv[i+2], v = yuyv[i+3];
            int c = y0 - 16, d = u - 128, e = v - 128;
            int r = (298 * c + 409 * e + 128) >> 8;
            int g = (298 * c - 100 * d - 208 * e + 128) >> 8;
            int b = (298 * c + 516 * d + 128) >> 8;
            r = (r < 0) ? 0 : (r > 255 ? 255 : r);
            g = (g < 0) ? 0 : (g > 255 ? 255 : g);
            b = (b < 0) ? 0 : (b > 255 ? 255 : b);
            color_array[pixel_index++] = (r << 16) | (g << 8) | b;
            
            c = y1 - 16;
            r = (298 * c + 409 * e + 128) >> 8;
            g = (298 * c - 100 * d - 208 * e + 128) >> 8;
            b = (298 * c + 516 * d + 128) >> 8;
            r = (r < 0) ? 0 : (r > 255 ? 255 : r);
            g = (g < 0) ? 0 : (g > 255 ? 255 : g);
            b = (b < 0) ? 0 : (b > 255 ? 255 : b);
            color_array[pixel_index++] = (r << 16) | (g << 8) | b;
        }
        
        // Allocate neighbor mask array.
        unsigned char *neighbor_mask_array = malloc(width * height * sizeof(unsigned char));
        if (!neighbor_mask_array) {
            perror("Allocating neighbor mask array");
            free(color_array);
            break;
        }
        int mask_threshold = 2500;
        identify_neighbors_within_threshold(color_array, width, height, mask_threshold, neighbor_mask_array);
        
        // Label connected components with an intensity threshold.
        int group_threshold = 10;
        int *labels = label_connected_components(color_array, width, height, group_threshold);
        if (!labels) {
            free(color_array);
            free(neighbor_mask_array);
            break;
        }
        
        // Process the frame: save each group's data to its own binary file.
        process_frame_example(color_array, width, height, labels, neighbor_mask_array);
        
        free(labels);
        free(color_array);
        free(neighbor_mask_array);
        
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
    close(fd);
}

int main(void) {
    capture_loop(process_frame_example);
    return 0;
}
