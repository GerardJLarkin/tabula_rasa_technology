#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <time.h>
#include <math.h>  // for abs()

#define CLEAR(x) memset(&(x), 0, sizeof(x))
#define _POSIX_C_SOURCE 200809L


// Structure for memory-mapped buffers.
struct buffer {
    void   *start;
    size_t length;
};

// -------------------------------------------------------------------------
// PART 1: Neighbor Masking
// -------------------------------------------------------------------------

// Compute the squared Euclidean distance between two colors in 0xRRGGBB format.
static inline int color_distance_squared(unsigned int color1, unsigned int color2) {
    int r1 = (color1 >> 16) & 0xFF;
    int g1 = (color1 >> 8)  & 0xFF;
    int b1 = color1 & 0xFF;
    int r2 = (color2 >> 16) & 0xFF;
    int g2 = (color2 >> 8)  & 0xFF;
    int b2 = color2 & 0xFF;
    int dr = r1 - r2, dg = g1 - g2, db = b1 - b2;
    return dr*dr + dg*dg + db*db;
}

/**
 * For each pixel in the frame (represented in a flattened array of packed 0xRRGGBB),
 * check its 8-connected neighbors (the 3×3 window around the pixel) and set bits
 * in an 8-bit mask when a neighbor’s color difference is below the given threshold.
 *
 * Bit mapping for the mask:
 *   Bit 0: top-left
 *   Bit 1: top
 *   Bit 2: top-right
 *   Bit 3: left
 *   Bit 4: right
 *   Bit 5: bottom-left
 *   Bit 6: bottom
 *   Bit 7: bottom-right
 *
 * The output neighbor_mask_array (one byte per pixel) will contain the mask.
 */
void identify_neighbors_within_threshold(const unsigned int *color_array, int width, int height, int threshold, unsigned char *neighbor_mask_array) {
    for (int y = 0; y < height; y++){
        for (int x = 0; x < width; x++){
            int index = y * width + x;
            unsigned int center_color = color_array[index];
            unsigned char mask = 0;
            for (int dy = -1; dy <= 1; dy++){
                for (int dx = -1; dx <= 1; dx++){
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
                        if      (dx == -1 && dy == -1) bit = 0;  // top-left
                        else if (dx ==  0 && dy == -1) bit = 1;  // top
                        else if (dx ==  1 && dy == -1) bit = 2;  // top-right
                        else if (dx == -1 && dy ==  0) bit = 3;  // left
                        else if (dx ==  1 && dy ==  0) bit = 4;  // right
                        else if (dx == -1 && dy ==  1) bit = 5;  // bottom-left
                        else if (dx ==  0 && dy ==  1) bit = 6;  // bottom
                        else if (dx ==  1 && dy ==  1) bit = 7;  // bottom-right
                        if (bit >= 0)
                            mask |= (1 << bit);
                    }
                }
            }
            neighbor_mask_array[index] = mask;
        }
    }
}

void print_neighbor_masks(const unsigned int *color_array, int width, int height, const unsigned char *neighbor_mask_array) {
    // For demonstration, print the mask for the first 10 pixels.
    for (int i = 0; i < 10 && i < width * height; i++){
        printf("Pixel %d (color 0x%06X): neighbor mask = 0x%02X\n", 
               i, color_array[i] & 0xFFFFFF, neighbor_mask_array[i]);
    }
}

// -------------------------------------------------------------------------
// PART 2: Connected-Component Grouping (Flood Fill)
// -------------------------------------------------------------------------

// Compute intensity as the average of the R, G, and B components.
static inline int get_intensity(unsigned int color) {
    int r = (color >> 16) & 0xFF, g = (color >> 8) & 0xFF, b = color & 0xFF;
    return (r + g + b) / 3;
}

/**
 * Perform connected-component labeling based on pixel intensity.
 * Two pixels are connected if they are 8-connected and the absolute difference
 * of their intensities is ≤ threshold.
 *
 * Returns an array of labels (one per pixel) where connected pixels share the same label.
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
    int stack_top = 0;
    int current_label = 0;

    // Loop over every pixel.
    for (int i = 0; i < size; i++) {
        if (labels[i] != -1)
            continue;  // already labeled
        labels[i] = current_label;
        stack[stack_top++] = i;

        while (stack_top > 0) {
            int index = stack[--stack_top];
            int x = index % width;
            int y = index / width;
            int current_intensity = get_intensity(color_array[index]);

            // Check all 8 neighbors.
            for (int dy = -1; dy <= 1; dy++){
                for (int dx = -1; dx <= 1; dx++){
                    if (dx == 0 && dy == 0)
                        continue;
                    int nx = x + dx, ny = y + dy;
                    if (nx < 0 || nx >= width || ny < 0 || ny >= height)
                        continue;
                    int neighbor_index = ny * width + nx;
                    if (labels[neighbor_index] != -1)
                        continue;
                    int neighbor_intensity = get_intensity(color_array[neighbor_index]);
                    if (abs(current_intensity - neighbor_intensity) <= threshold) {
                        labels[neighbor_index] = current_label;
                        stack[stack_top++] = neighbor_index;
                    }
                }
            }
        }
        current_label++;
    }
    free(stack);
    return labels;
}

void print_connected_components(const int *labels, int width, int height) {
    int size = width * height;
    int max_label = -1;
    for (int i = 0; i < size; i++) {
        if (labels[i] > max_label)
            max_label = labels[i];
    }
    printf("Number of connected groups: %d\n", max_label + 1);
}

// Helper: Print the pixel groups with their coordinates and pixel color.
void print_pixel_groups(const int *labels, const unsigned int *color_array, int width, int height) {
    int size = width * height;
    int max_label = -1;
    for (int i = 0; i < size; i++) {
        if (labels[i] > max_label)
            max_label = labels[i];
    }
    int num_groups = max_label + 1;
    printf("Number of groups: %d\n", num_groups);

    // For each group, print the pixel coordinates along with its color.
    for (int group = 0; group < num_groups; group++) {
        printf("Group %d:\n", group);
        for (int i = 0; i < size; i++) {
            if (labels[i] == group) {
                int x = i % width;
                int y = i / width;
                unsigned int color = color_array[i];
                // Print coordinate and color value in hexadecimal.
                printf("  (%d,%d) -> 0x%06X\n", x, y, color & 0xFFFFFF);
            }
        }
        printf("\n");
    }
}


// -------------------------------------------------------------------------
// PART 3: V4L2 Capture Loop and Integration with Timing
// -------------------------------------------------------------------------

typedef void (*frame_callback_t)(unsigned int *color_array, int width, int height);

// Example callback: prints first pixel color and connected group information.
void process_frame_example(unsigned int *color_array, int width, int height) {
    if (!color_array)
        return;
    unsigned int color = color_array[0];
    // printf("Frame first pixel color: 0x%06X\n", color & 0xFFFFFF);

    // Perform connected-component labeling using an intensity threshold (e.g., 10).
    int group_threshold = 10;
    int *labels = label_connected_components(color_array, width, height, group_threshold);
    if (labels) {
        print_connected_components(labels, width, height);
        print_pixel_groups(labels, color_array, width, height);
        free(labels);
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
    // printf("Driver: %s\nCard: %s\n", cap.driver, cap.card);
    
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
        // Start timing this iteration.
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
        
        // Allocate a packed color array for this frame.
        unsigned int *color_array = malloc(width * height * sizeof(unsigned int));
        if (!color_array) {
            perror("Allocating color array");
            break;
        }
        
        // Convert YUYV data to packed RGB.
        // Every 4 bytes from YUYV represent 2 pixels.
        unsigned char *yuyv = buffers[buf.index].start;
        int pixel_index = 0;
        for (int i = 0; i < width * height * 2; i += 4) {
            unsigned char y0 = yuyv[i], u = yuyv[i+1], y1 = yuyv[i+2], v = yuyv[i+3];
            int c = y0 - 16, d = u - 128, e = v - 128;
            int r = (298 * c + 409 * e + 128) >> 8;
            int g = (298 * c - 100 * d - 208 * e + 128) >> 8;
            int b = (298 * c + 516 * d + 128) >> 8;
            if (r < 0) r = 0; if (r > 255) r = 255;
            if (g < 0) g = 0; if (g > 255) g = 255;
            if (b < 0) b = 0; if (b > 255) b = 255;
            color_array[pixel_index++] = (r << 16) | (g << 8) | b;
            
            c = y1 - 16;
            r = (298 * c + 409 * e + 128) >> 8;
            g = (298 * c - 100 * d - 208 * e + 128) >> 8;
            b = (298 * c + 516 * d + 128) >> 8;
            if (r < 0) r = 0; if (r > 255) r = 255;
            if (g < 0) g = 0; if (g > 255) g = 255;
            if (b < 0) b = 0; if (b > 255) b = 255;
            color_array[pixel_index++] = (r << 16) | (g << 8) | b;
        }
        
        // Allocate neighbor mask array.
        unsigned char *neighbor_mask_array = malloc(width * height * sizeof(unsigned char));
        if (!neighbor_mask_array) {
            perror("Allocating neighbor mask array");
            free(color_array);
            break;
        }
        int mask_threshold = 2500; // e.g. squared difference threshold.
        identify_neighbors_within_threshold(color_array, width, height, mask_threshold, neighbor_mask_array);
        
        // Process the frame (callback prints first-pixel info and group counts).
        process_frame(color_array, width, height);
        // Also print some neighbor mask info.
        // print_neighbor_masks(color_array, width, height, neighbor_mask_array);
        
        free(color_array);
        free(neighbor_mask_array);
        
        if (ioctl(fd, VIDIOC_QBUF, &buf) < 0) {
            perror("Requeue Buffer");
            break;
        }
        
        // End timing for this iteration.
        clock_gettime(CLOCK_MONOTONIC, &loop_end);
        double elapsed = (loop_end.tv_sec - loop_start.tv_sec) +
                         (loop_end.tv_nsec - loop_start.tv_nsec) / 1e9;
        printf("Iteration took %.6f seconds\n", elapsed);
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
    // Start the capture loop using our example frame-processing callback.
    capture_loop(process_frame_example);
    return 0;
}
