#define _POSIX_C_SOURCE 200809L
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
#include <sqlite3.h>

#define CLEAR(x) memset(&(x), 0, sizeof(x))
#define _USE_MATH_DEFINES

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

// -------------------------------------------------------------------------
// Structures & Global Frame Index
// -------------------------------------------------------------------------

struct buffer {
    void   *start;
    size_t length;
};

static int global_frame_index = 0; // Frame index; resets to 0 after 900 frames.

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
    return dr*dr + dg*dg + db*db;
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
            for (int dy = -1; dy <= 1; dy++){
                for (int dx = -1; dx <= 1; dx++){
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
// PART 3: Database Saving Functionality
// -------------------------------------------------------------------------

/**
 * Save group output to a SQLite database.
 * For each group, this function computes:
 *   - The group centroid in original (i, j) and normalized (i, j) coordinates.
 *   - For each pixel, its normalized coordinate (norm = 2*(coord/(max_coord)) - 1)
 *     and its segment (computed via atan2 on normalized coordinates; unit circle split into 8 segments).
 * It then writes pixel-level and group-level summaries to two tables.
 */
void save_group_output_to_db(int frame_index, int width, int height, const unsigned int *color_array, const int *labels) {
    int size = width * height;
    int max_label = -1;
    for (int i = 0; i < size; i++) {
        if (labels[i] > max_label)
            max_label = labels[i];
    }
    int num_groups = max_label + 1;

    // Allocate arrays to accumulate per-group sums and counts.
    int *group_count = calloc(num_groups, sizeof(int));
    double *sum_orig_i = calloc(num_groups, sizeof(double));
    double *sum_orig_j = calloc(num_groups, sizeof(double));
    double *sum_norm_i = calloc(num_groups, sizeof(double));
    double *sum_norm_j = calloc(num_groups, sizeof(double));
    int seg_counts[num_groups][8];
    memset(seg_counts, 0, sizeof(seg_counts));

    // Loop over all pixels.
    for (int i = 0; i < size; i++) {
        int group = labels[i];
        int x = i % width;
        int y = i / width;
        group_count[group]++;
        sum_orig_i[group] += x;
        sum_orig_j[group] += y;
        // Normalize coordinates to [-1, 1]:
        double norm_i = 2.0 * x / (width - 1) - 1.0;
        double norm_j = 2.0 * y / (height - 1) - 1.0;
        sum_norm_i[group] += norm_i;
        sum_norm_j[group] += norm_j;
        double angle = atan2(norm_j, norm_i);  // angle in radians, range [-pi, pi]
        int seg = (int)floor((angle + M_PI) / (M_PI / 4));
        if (seg < 0) seg = 0;
        if (seg > 7) seg = 7;
        seg_counts[group][seg]++;
    }

    // Compute per-group centroids.
    double *centroid_orig_i = malloc(num_groups * sizeof(double));
    double *centroid_orig_j = malloc(num_groups * sizeof(double));
    double *centroid_norm_i = malloc(num_groups * sizeof(double));
    double *centroid_norm_j = malloc(num_groups * sizeof(double));
    for (int g = 0; g < num_groups; g++) {
        if (group_count[g] > 0) {
            centroid_orig_i[g] = sum_orig_i[g] / group_count[g];
            centroid_orig_j[g] = sum_orig_j[g] / group_count[g];
            centroid_norm_i[g] = sum_norm_i[g] / group_count[g];
            centroid_norm_j[g] = sum_norm_j[g] / group_count[g];
        } else {
            centroid_orig_i[g] = centroid_orig_j[g] = 0;
            centroid_norm_i[g] = centroid_norm_j[g] = 0;
        }
    }

    // Open (or create) the SQLite database.
    sqlite3 *db;
    int rc = sqlite3_open("group_output.db", &db);
    if (rc) {
        fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(db));
        goto cleanup;
    }

    char *errMsg = NULL;
    // Create table for pixel-level output.
    const char *sql_pixel = "CREATE TABLE IF NOT EXISTS pixel_output ("
                            "frame_index INTEGER, group_id INTEGER, pixel_color INTEGER, "
                            "orig_i INTEGER, orig_j INTEGER, norm_i REAL, norm_j REAL, "
                            "group_centroid_orig_i REAL, group_centroid_orig_j REAL, "
                            "group_centroid_norm_i REAL, group_centroid_norm_j REAL, "
                            "segment INTEGER);";
    rc = sqlite3_exec(db, sql_pixel, 0, 0, &errMsg);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "SQL error (pixel_output): %s\n", errMsg);
        sqlite3_free(errMsg);
    }

    // Create table for group-level summary.
    const char *sql_group = "CREATE TABLE IF NOT EXISTS group_summary ("
                            "frame_index INTEGER, group_id INTEGER, "
                            "group_centroid_orig_i REAL, group_centroid_orig_j REAL, "
                            "group_centroid_norm_i REAL, group_centroid_norm_j REAL, "
                            "seg0_count INTEGER, seg1_count INTEGER, seg2_count INTEGER, "
                            "seg3_count INTEGER, seg4_count INTEGER, seg5_count INTEGER, "
                            "seg6_count INTEGER, seg7_count INTEGER);";
    rc = sqlite3_exec(db, sql_group, 0, 0, &errMsg);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "SQL error (group_summary): %s\n", errMsg);
        sqlite3_free(errMsg);
    }

    // Prepare insert statement for pixel_output.
    sqlite3_stmt *stmt_pixel;
    const char *sql_insert_pixel = "INSERT INTO pixel_output ("
                                   "frame_index, group_id, pixel_color, orig_i, orig_j, "
                                   "norm_i, norm_j, group_centroid_orig_i, group_centroid_orig_j, "
                                   "group_centroid_norm_i, group_centroid_norm_j, segment) "
                                   "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);";
    rc = sqlite3_prepare_v2(db, sql_insert_pixel, -1, &stmt_pixel, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Cannot prepare pixel statement: %s\n", sqlite3_errmsg(db));
        sqlite3_close(db);
        goto cleanup;
    }

    // Insert a row for each pixel.
    for (int i = 0; i < size; i++) {
        int group = labels[i];
        int x = i % width;
        int y = i / width;
        double norm_i = 2.0 * x / (width - 1) - 1.0;
        double norm_j = 2.0 * y / (height - 1) - 1.0;
        double angle = atan2(norm_j, norm_i);
        int seg = (int)floor((angle + M_PI) / (M_PI / 4));
        if (seg < 0) seg = 0;
        if (seg > 7) seg = 7;

        sqlite3_bind_int(stmt_pixel, 1, frame_index);
        sqlite3_bind_int(stmt_pixel, 2, group);
        sqlite3_bind_int(stmt_pixel, 3, color_array[i]);
        sqlite3_bind_int(stmt_pixel, 4, x);
        sqlite3_bind_int(stmt_pixel, 5, y);
        sqlite3_bind_double(stmt_pixel, 6, norm_i);
        sqlite3_bind_double(stmt_pixel, 7, norm_j);
        sqlite3_bind_double(stmt_pixel, 8, centroid_orig_i[group]);
        sqlite3_bind_double(stmt_pixel, 9, centroid_orig_j[group]);
        sqlite3_bind_double(stmt_pixel, 10, centroid_norm_i[group]);
        sqlite3_bind_double(stmt_pixel, 11, centroid_norm_j[group]);
        sqlite3_bind_int(stmt_pixel, 12, seg);

        rc = sqlite3_step(stmt_pixel);
        if (rc != SQLITE_DONE) {
            fprintf(stderr, "Pixel insert failed: %s\n", sqlite3_errmsg(db));
        }
        sqlite3_reset(stmt_pixel);
    }
    sqlite3_finalize(stmt_pixel);

    // Prepare insert statement for group_summary.
    sqlite3_stmt *stmt_group;
    const char *sql_insert_group = "INSERT INTO group_summary ("
                                   "frame_index, group_id, group_centroid_orig_i, group_centroid_orig_j, "
                                   "group_centroid_norm_i, group_centroid_norm_j, "
                                   "seg0_count, seg1_count, seg2_count, seg3_count, seg4_count, seg5_count, seg6_count, seg7_count) "
                                   "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);";
    rc = sqlite3_prepare_v2(db, sql_insert_group, -1, &stmt_group, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Cannot prepare group statement: %s\n", sqlite3_errmsg(db));
        sqlite3_close(db);
        goto cleanup;
    }

    for (int g = 0; g < num_groups; g++) {
        sqlite3_bind_int(stmt_group, 1, frame_index);
        sqlite3_bind_int(stmt_group, 2, g);
        sqlite3_bind_double(stmt_group, 3, centroid_orig_i[g]);
        sqlite3_bind_double(stmt_group, 4, centroid_orig_j[g]);
        sqlite3_bind_double(stmt_group, 5, centroid_norm_i[g]);
        sqlite3_bind_double(stmt_group, 6, centroid_norm_j[g]);
        for (int s = 0; s < 8; s++) {
            sqlite3_bind_int(stmt_group, 7 + s, seg_counts[g][s]);
        }
        rc = sqlite3_step(stmt_group);
        if (rc != SQLITE_DONE) {
            fprintf(stderr, "Group insert failed: %s\n", sqlite3_errmsg(db));
        }
        sqlite3_reset(stmt_group);
    }
    sqlite3_finalize(stmt_group);
    sqlite3_close(db);

cleanup:
    free(group_count);
    free(sum_orig_i);
    free(sum_orig_j);
    free(sum_norm_i);
    free(sum_norm_j);
    free(centroid_orig_i);
    free(centroid_orig_j);
    free(centroid_norm_i);
    free(centroid_norm_j);
}

// -------------------------------------------------------------------------
// PART 4: Capture Loop (with Timing & New Output)
// -------------------------------------------------------------------------

typedef void (*frame_callback_t)(unsigned int *color_array, int width, int height, const int *labels);

void process_frame_example(unsigned int *color_array, int width, int height, const int *labels) {
    if (!color_array)
        return;
    unsigned int first_color = color_array[0];
    printf("Frame first pixel color: 0x%06X\n", first_color & 0xFFFFFF);

    // For demonstration, print number of connected groups.
    int size = width * height;
    int max_label = -1;
    for (int i = 0; i < size; i++) {
        if (labels[i] > max_label)
            max_label = labels[i];
    }
    printf("Number of connected groups in frame: %d\n", max_label + 1);

    // Save the detailed output to the database.
    save_group_output_to_db(global_frame_index, width, height, color_array, labels);
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
    printf("Driver: %s, Card: %s\n", cap.driver, cap.card);
    
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
        // Timing start for this loop iteration.
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
        
        // Allocate and fill a packed color array from the YUYV data.
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
        
        // Optionally, allocate a neighbor mask array (if needed).
        unsigned char *neighbor_mask_array = malloc(width * height * sizeof(unsigned char));
        if (!neighbor_mask_array) {
            perror("Allocating neighbor mask array");
            free(color_array);
            break;
        }
        int mask_threshold = 2500;
        identify_neighbors_within_threshold(color_array, width, height, mask_threshold, neighbor_mask_array);
        
        // Label connected components with an intensity threshold (e.g., 10).
        int group_threshold = 10;
        int *labels = label_connected_components(color_array, width, height, group_threshold);
        if (!labels) {
            free(color_array);
            free(neighbor_mask_array);
            break;
        }
        
        // Process the frame (print some info and save to database).
        process_frame_example(color_array, width, height, labels);
        
        free(labels);
        free(color_array);
        free(neighbor_mask_array);
        
        if (ioctl(fd, VIDIOC_QBUF, &buf) < 0) {
            perror("Requeue Buffer");
            break;
        }
        
        // Timing end for this iteration.
        clock_gettime(CLOCK_MONOTONIC, &loop_end);
        double elapsed = (loop_end.tv_sec - loop_start.tv_sec) + (loop_end.tv_nsec - loop_start.tv_nsec) / 1e9;
        printf("Iteration took %.6f seconds\n", elapsed);
        
        // Update global frame index (reset to 0 at 900).
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
